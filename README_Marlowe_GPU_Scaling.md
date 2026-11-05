# Marlowe H100 strong scaling: 1, 2, 4, 8 GPUs (one node) and 16 GPUs (two nodes)

This is the Marlowe counterpart of the Sherlock A40 scaling suite
(`benchmarks/gpu_scaling/sherlock_a40_scaling.sbatch`). It uses the same solver
path: BIPN, Kokkos fluid assembly, Schur AMG with reused transfers, and the
explicit `BᵀB` product, with one MPI rank per GPU. It runs the same two cases:

| Case | Steps | Momentum GMRES preconditioner | Same as on Sherlock |
|---|---:|---|---|
| `iliac`: iliac artery, 76k tets | 20 | diagonal | `scale` suite |
| `pipe`: Lasso's 1M-tet pipe, 181,875 nodes | 10 | resistance | `pipe` suite |

Each GPU count runs every combination of a **binding mode** and a
**communication mode**:

| Binding | mpirun | What it tests |
|---|---|---|
| `core` | `--map-by ppr:N:node:PE=1 --bind-to core` | The recipe of every earlier Marlowe job. Open MPI places ranks on consecutive cores, so with 8 GPUs the ranks driving GPUs 4–7 are expected to sit on the other socket. The binding probe records this. |
| `numa` | `--bind-to none`, then `taskset` to the GPU's NUMA-local CPUs (from sysfs) | Each rank stages host buffers on the socket attached to its GPU and NIC. |
| `none` | `--bind-to none` | No pinning at all; the OS scheduler decides. |

| Communication | Settings | Purpose |
|---|---|---|
| `ucx-best` | 1 node: `UCX_TLS=self,sm,cuda_copy`; 2 nodes: `UCX_TLS=rc,sm,self,cuda_copy`; `UCX_MEMTYPE_CACHE=n`, `UCX_IB_GPU_DIRECT_RDMA=n`; `TPETRA_ASSUME_GPU_AWARE_MPI` unset | The fastest repeatable settings from the August Marlowe UCX sweeps (pipe RCR): 14.0 s vs 61.9 s with UCX defaults on 4 GPUs, 15.1 vs 109.1 s on 8, and 21.4 vs 96.5 s on 16. |
| `gpuaware-ipc` | `TPETRA_ASSUME_GPU_AWARE_MPI=1`; `cuda_ipc` added to `UCX_TLS` | The NVLink path. Tpetra hands device buffers to MPI, and UCX may copy them GPU-to-GPU over NVLink/NVSwitch inside a node. |
| `ucx-cache-on`, `host-staged`, `gpuaware`, `ucx-default` | optional, see the sbatch header | `ucx-cache-on`: two-node candidate (single run in August). `host-staged`: Tpetra=0. `gpuaware`: Tpetra=1 without `cuda_ipc`. `ucx-default`: reference only. |

Forced GPUDirect RDMA is never used: in August it took 295 s against 21 s.

**NVLink is not automatic.** In `ucx-best` the UCX transport list has no
`cuda_ipc`, so any GPU buffer that reaches MPI is copied through host memory
(`cuda_copy` + `sm`). That is the same PCIe path an A40 node uses. Only
`gpuaware-ipc` can move halo data over NVLink. The August sweep's `cuda_ipc`
variants did not set `TPETRA_ASSUME_GPU_AWARE_MPI`, so it is not known whether
device buffers ever reached UCX in them. Comparing `ucx-best` with
`gpuaware-ipc` at 4 and 8 GPUs answers that.

## 1. Build on Marlowe

Use the build kit in `buildSVMPonMarloweHPC` (modules `gcc/13.1.0`,
`cmake/3.30.3`, `nvhpc/24.7`, HPC-X 2.19) with this fork and branch:

```bash
sbatch --export=ALL,INSTALL_ROOT=/projects/m000218/Sujal/svmp_gpuassembly,\
SVMP_REPO=https://github.com/sujaldave/svMultiPhysics.git,SVMP_BRANCH=gpuAssemblyTest \
  buildLibrariesAndSVMP.sh
```

For parity with Sherlock `build3` (Trilinos 18.0 with vendor kernels), add
these flags to `build_trilinos` in that script. Delete
`$INSTALL_ROOT/stamps/trilinos.done` (or use a new `INSTALL_ROOT`) so Trilinos
is rebuilt:

```text
-DKokkos_ARCH_HOPPER90=ON            (already there)
-DTPL_ENABLE_CUDA=ON -DTPL_ENABLE_CUSPARSE=ON -DTPL_ENABLE_CUBLAS=ON -DTPL_ENABLE_CUSOLVER=ON
-DTpetra_INST_CUDA=ON -DKokkosKernels_INST_LAYOUTLEFT=ON
```

The cuSPARSE/cuBLAS kernels made pressure CG and GMRES 9–18% cheaper per
iteration on the A40. Pin the same Trilinos source as Sherlock
(`TRILINOS_GIT_REF=...`), or the two machines will not run the same library.
The job refuses a binary whose Kokkos does not target `HOPPER90`. It records
the Trilinos version and the KokkosKernels TPLs in
`config/attempt-*/trilinos.txt`.

## 2. Pipe case (once)

The pipe generator needs NumPy. Either copy the Sherlock case:

```bash
rsync -a sherlock:/scratch/users/sdave03/pipe-case/ /projects/m000218/Sujal/pipe-case/
```

or generate it with any Python that has NumPy:

```bash
benchmarks/large_pipe/prepare_case.sh /projects/m000218/Sujal/pipe-case /path/to/python3
```

## 3. Submit

From the repository root, i.e. `$INSTALL_ROOT/svMultiPhysics` with the build
kit:

```bash
export SVMP_BUILD_DIR=$PWD/build                              # contains svMultiPhysics-build/bin/svmultiphysics
export SVMP_ENV=/projects/m000218/Sujal/svmp_gpuassembly/svmp_marlowe_env.sh
export SVMP_PIPE_CASE=/projects/m000218/Sujal/pipe-case
SVMP_DRY_RUN=1 benchmarks/gpu_scaling/submit_marlowe.sh       # check the four sbatch commands
benchmarks/gpu_scaling/submit_marlowe.sh                      # submit them
```

That submits four independent jobs into one campaign root:

| Job | Nodes | GPU counts | Configurations |
|---|---:|---|---|
| `iliac-nodes1`, `pipe-nodes1` | 1 (exclusive) | 1, 2, 4, 8 | 1 GPU × 3 bindings, plus {2, 4, 8} × 3 bindings × 2 comm modes = 21 |
| `iliac-nodes2`, `pipe-nodes2` | 2 (exclusive) | 16 | 3 bindings × 2 comm modes = 6 |

Each configuration runs `SVMP_REPEAT=3` independent solver processes. Each
(case, GPU count) also gets one untimed 2-step diagnostic run that prints the
Kokkos configuration and AMG hierarchy details.

A single job can also be submitted directly:

```bash
SVMP_BUILD_DIR=... SVMP_ENV=... SVMP_CASES=iliac sbatch benchmarks/gpu_scaling/marlowe_h100_scaling.sbatch
SVMP_BUILD_DIR=... SVMP_ENV=... SVMP_CASES=iliac sbatch --nodes=2 benchmarks/gpu_scaling/marlowe_h100_scaling.sbatch
```

Useful overrides (all `SVMP_*` variables pass through `submit_marlowe.sh`):

| Variable | Default | Example |
|---|---|---|
| `SVMP_BIND_MODES` | `core numa none` | `numa` once the winner is known |
| `SVMP_COMM_MODES` | `ucx-best gpuaware-ipc` | `"ucx-best ucx-cache-on"` on two nodes |
| `SVMP_GPU_COUNTS` | `1 2 4 8` (1 node), `16` (2 nodes) | `8` on 2 nodes = 4+4 GPUs, isolates the inter-node cost |
| `SVMP_SCHUR_COARSE` | KLU2 | `chebyshev`: device coarse solve, tests the KLU2-gather hypothesis |
| `SVMP_MPIRUN_EXTRA` | none | `"--mca coll ^ucc,hcoll"` if a GPU-aware run fails with UCC `Message truncated` |
| `SVMP_REPEAT`, `SVMP_RUN_TIMEOUT` | 3, 900 s | |
| `SVMP_ACCOUNT`, `SVMP_PARTITION`, `SVMP_TIME` | header: `marlowe-m000218`, `preempt`, 4 h | |

**Preemption and failures.** The jobs run on `preempt` with `--requeue`. A
requeued job keeps its ID and output directory, skips finished configurations,
and sets partial ones aside as `*.incomplete-*`. A run that exceeds
`SVMP_RUN_TIMEOUT` is killed together with all of its ranks. A timeout is the
form the A40 4-GPU stall would take here (542 s). The configuration is then
listed in `failed.txt` and the sweep continues; set `SVMP_RETRY_FAILED=1` to
retry such configurations on the next submission.

## 4. Results

```text
<campaign>/<case>-nodes<N>/
  sweep.json                          settings; a resubmission must match them
  config/attempt-<job>-<restart>/      commit, diff, ldd, modules, CMake caches, Trilinos/Kokkos,
                                       nvidia-smi -q / topo -m / nvlink, lscpu, numactl -H,
                                       ompi_info, ucx_info -d, IB devices, environment
  config/gpu-map/<host>.tsv            GPU index -> PCI bus -> NUMA node -> local CPUs
  config/binding-<bind>-<N>.log        mpirun --report-bindings
  binding/<bind>/gpus-<N>/rank-*.json  per rank: host, GPU bus ID, CPU affinity, NUMA locality
  <case>/warmup/gpus-<N>/              untimed 2-step diagnostics
  <case>/<bind>__<comm>/gpus-<N>/run-<k>/   stdout.log, solver.xml, measurement.json, <N>-procs/
  failed.txt                           failed or timed-out configurations
  marlowe_scaling.{md,csv,json}        summary written at the end of each job
```

`measurement.json` now also records the communication environment of the run
(all `UCX_*`, `OMPI_MCA_*` and `TPETRA_*` variables, and the binding and
communication modes) and whether it timed out.

Combined summary across all four jobs, on Marlowe or after copying the campaign
back:

```bash
python3 benchmarks/gpu_scaling/summarize_marlowe.py <campaign> --expect-cuda
```

With Python VTK available, for example locally, the same command also compares
every run's final fields with the first 1-GPU run of its case.

The summary contains one table per case, with these columns:

- binding and communication mode, and GPU count;
- median wall time with min–max;
- **pressure-CG ms per iteration**, the A40 diagnostic: 0.40 → 0.75 → 11.3 ms on 1/2/4 A40s;
- CG iterations;
- GMRES time;
- speedup and efficiency against the 1-GPU run with the same binding;
- speedup against the fastest 1-GPU run;
- binding probe result, e.g. `4/8 ranks NUMA-local` for `core`.

## 5. How to read it

1. **1 GPU, all bindings:** the direct H100 vs A40 comparison. Compare with
   35.7 s for iliac and 82.0 s for the pipe on one A40 (`build3`).
2. **2, 4 and 8 GPUs, `ucx-best` vs `gpuaware-ipc`:** whether NVLink helps. If
   CG ms/iteration stays flat from 1 to 4 GPUs here but blew up on the A40,
   the A40 stall was the PCIe/cross-socket path.
3. **`core` vs `numa` at 8 and 16 GPUs:** whether the old binding recipe costs
   time. Check each run's binding probe first.
4. **16 GPUs vs 8 GPUs:** the inter-node cost. Rerun `SVMP_GPU_COUNTS=8` on 2
   nodes (4+4) to compare at equal GPU count.

Two cautions about problem size:

- **The iliac case is far too small for 8–16 H100s:** about 1–2k nodes per
  GPU. A flat or rising wall time there says little about the code.
- **Even the pipe is small:** about 23k nodes per GPU at 8 GPUs and about 11k
  at 16. An H100 needs on the order of 10⁶ unknowns to be busy.

So read these runs as tests of communication and latency. Do not quote their
efficiency as the code's scalability. That needs a larger mesh, for example the
1.58M-element AAA model or a refined pipe.
