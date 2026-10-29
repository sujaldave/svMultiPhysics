# A40 multiphysics audit and AMG/communication experiments

## What the Sherlock run establishes

Audited `oct27-tests/multiphysics-50018693`, its sibling configuration directory,
and batch stdout/stderr. The batch stderr is empty; individual solver logs contain
errors. Re-ran the multiphysics summary locally with VTK to finish field checks.

Of 60 attempted runs, **48 completed, eight timed out, and four failed**. The
completed, strictly converged configurations agree with their references to
**7.70e-14 relative L2** across saved point fields. `Validated=False` in the original
cluster summary also reflected unavailable Python VTK; that alone was not a solver
failure. The regenerated local summary distinguishes those conditions.

| Policy | CPU median wall s | A40 median wall s | Interpretation |
|---|---:|---:|---|
| Scalar GMRES/diagonal, FSILS assembly | 103.970 | 158.660 | Fastest completed scalar policy; host element integration still dominates |
| Scalar GMRES/diagonal, Trilinos assembly | 116.664 | 175.649 | Direct host Trilinos assembly remains slower for this case |
| Solid CG/diagonal, Trilinos assembly | 1.074 | 5.276 | Best solid policy; simulation alone is 0.619 vs 1.524 s |
| Solid GMRES/MueLu, Trilinos assembly | 3.694 | 28.116 | Converges, but loses heavily to CG on this tiny beam |
| FSI GMRES/RILUK0, mesh CG, Trilinos assembly | 26.293 | 49.562 | Fastest FSI policy meeting every linear tolerance on A40 |
| FSI GMRES/diagonal, mesh CG, Trilinos assembly | 12.297 | 23.664 | Matching nonlinear solution/fields, but 135 linear-limit warnings per run |

The A40 timings use one GPU and one host rank; the CPU measurements use one
rank/thread on an Apple M4 Pro. They are not a full CPU-node versus GPU-node
comparison. These scalar, solid and FSI equations **do not invoke the new fluid
Kokkos element kernel**. The device solves are too small/fragmented to overcome
host integration and launch/setup costs in these tests.

FSI/MueLu fails immediately with `ParallelReduce<Cuda> requested too much L0
scratch memory`. Its approximately five-second exit time is not a simulation
result. The logs do not identify which MueLu/Kokkos kernel requested the scratch;
the later Tpetra-finalization messages follow the exception and do not establish
the original cause. This is the general coupled-system MueLu implementation,
separate from the Schur AMG path. Scalar RILUK0 and MueLu hit the 180-second launch
timeout: their partial histories and missing final states cannot establish either
full-run speed or final convergence. A larger timeout is needed to rank them.

## Why pressure AMG and communication are the next target

The previous 1/2/4-GPU iliac study measured assembly about
1.477/0.757/0.371 s, while pressure CG rose from 19.42 to 33.97 to 432.49 s.
Iteration counts remained comparable. That is strong evidence to investigate the
cost per iteration: AMG applications, halo exchange, global reductions, the
coarse solve, and synchronization. It does not prove which communication path
or coarse level caused the four-GPU regression.

Priorities are: (1) measure MPI/CUDA behavior on two and four GPUs; (2) isolate
host staging from GPU-aware MPI; (3) replace the CPU coarse solve experimentally;
(4) use the trace to decide whether to change coarse rank participation, aggregation,
CG reductions, or fuse small kernels. A pipelined CG implementation or manual MPI
rewrite would change numerical/communication behavior substantially and needs
trace evidence and dedicated numerical tests before adoption.

## Implemented experiment

`SV_TRILINOS_SCHUR_COARSE=klu|chebyshev` selects the **Schur** coarse solver.
The default remains KLU. The optional Chebyshev coarse solve uses eight polynomial
steps, eigenvalue ratio 30, and a zero initial guess. It avoids the CPU KLU
factorization/application path, but retains distributed matrix applications and
outer CG reductions. It does not promise fewer MPI messages or fewer iterations.

A deterministic upper bound of 200 on the diagonally scaled SPD coarse matrix's
largest eigenvalue follows from its positive eigenvalues and trace equal to its
row count. The code explicitly rejects a polynomial hierarchy with more than
200 coarse rows. This conservative bound avoids random spectral estimates and
preserves numerical scaling when the matrix is refreshed. It may be slower
than an accurately tuned polynomial. The existing RP refresh policy updates
numerical levels; the hierarchy cache also invalidates when coarse policy changes.
The pressure operator, outlet physics, precision and stopping tolerances are retained.

Existing Schur tests were run with the new option: residual reduction, symmetry,
distributed products, exact low-rank corrections, and numerical scaling across
hierarchy reuse. All seven pass on one rank and on each of two MPI ranks. The
initial random-estimate polynomial failed the numerical scaling test and was
replaced before acceptance. The full default test-suite result and matched CPU
benchmark results are recorded below.

## Run on Sherlock

Rebuild the solver after syncing the C++ changes; benchmark scripts alone cannot
activate the new coarse option. Use the existing successful CUDA/Trilinos module
and CMake setup, then build the inner target:

```bash
cmake --build build/svMultiPhysics-build -j4
sbatch benchmarks/amg_application/sherlock_a40.sbatch
```

The job reserves one node with four A40s for two hours. It runs matched **five-step,
two-repeat** iliac tests on 1, 2 and 4 GPUs. Momentum uses resistance preconditioning.
For each count, it compares:

| Group | Coarse solver | Tpetra GPU-aware MPI setting |
|---|---|---|
| `klu-auto` | KLU | Unset: library default |
| `klu-staged` | KLU | `TPETRA_ASSUME_GPU_AWARE_MPI=0` |
| `klu-device` | KLU | `TPETRA_ASSUME_GPU_AWARE_MPI=1` |
| `polynomial-auto` | Chebyshev | Unset: library default |

The forced-device test requires the CUDA-aware MPI installation used for the
previous successful Sherlock runs. Setting the variable does not add CUDA support
to an MPI library. No `UCX_TLS` or Open MPI transport is forced. Binding validation,
verbose CPU affinity, GPU topology, MPI configuration, source diff, binary hash,
per-Jacobian timings, and field outputs are retained. One GPU provides a control:
a speed change seen there cannot be attributed to inter-rank traffic.

For a first two/four-GPU diagnostic including separate two-step Nsight MPI/CUDA
profiles (keep Nsight and MPI modules available):

```bash
SVMP_GPU_COUNTS="2 4" SVMP_PROFILE=1 \
  sbatch benchmarks/amg_application/sherlock_a40.sbatch
```

Profiles use `--trace=cuda,nvtx,mpi,osrt`; they are outside timing samples.
Inspect MPI call duration/counts, CUDA memcpy/synchronization around MPI, coarse
solver work, and rank imbalance, correlating these with `cg_s` and iteration
counts. MPI API time can overlap device work: do not sum trace categories as an
end-to-end runtime. Only after a stable winner, confirm with `SVMP_STEPS=20` and
an increased Slurm time allocation (the old four-GPU run exceeded 30 minutes).

Copy the output directory, batch logs and profiles back locally. For each count:

```bash
python3 benchmarks/solver_optimization/summarize.py \
  /path/to/amg-application-JOBID/gpus-4 \
  --reference-group klu-auto --expect-cuda
```

Python VTK completes the field checks. Compare wall, CG time, iterations,
convergence/true-residual failures, and solution fields. A faster failed inner
solve is not an accepted improvement. CUDA-aware MPI's local CPU effect cannot
be measured on this CPU-only Mac; GPU timing claims require this job.

## Completed local coarse-solver comparison

Iliac artery, 20 steps, two MPI ranks, one thread each, two repeats, native CPU
Kokkos assembly, momentum resistance preconditioning and Schur transfer reuse:

| Coarse solver | Run 1 wall s | Run 2 wall s | Median wall s |
|---|---:|---:|---:|
| KLU (default) | 80.807 | 81.037 | 80.922 |
| Chebyshev (optional) | 113.351 | 114.348 | 113.849 |

The polynomial candidate is **40.7% slower on this CPU**. In the first matched
run, pressure iterations rise from 51,254 to 80,883 and pressure CG time from
56.47 to 89.99 s. Both policies complete 184 Newton iterations, with zero CG,
true-residual or outer-solve failures. Their six saved fields agree within
3.4e-13 relative L2. There is no local speedup to claim, and KLU remains the default.
The A40 experiment tests whether avoiding its host coarse solve compensates for
the additional iterations on that hardware; it might also lose there.

Full default verification: **363 passed, one expected single-rank skip** out of
364 tests. Seven Schur tests also pass with the polynomial option on one rank
and on each of two ranks. Three benchmark compatibility tests pass; both shell
scripts pass syntax checks.

Raw CPU results: `build/benchmarks/amg-application/cpu/`. Compact results are in
[`benchmarks/amg_application/cpu_results.json`](benchmarks/amg_application/cpu_results.json).
The A40 multiphysics audit is in
[`benchmarks/amg_application/a40_multiphysics_audit.json`](benchmarks/amg_application/a40_multiphysics_audit.json).

To reproduce locally with a new output directory:

```bash
SVMP_OUTPUT="$PWD/build/benchmarks/amg-application/cpu-new" \
  bash benchmarks/amg_application/run_cpu.sh
SV_TRILINOS_SCHUR_COARSE=chebyshev \
  build/svMultiPhysics-build/bin/run_all_unit_tests --gtest_filter='TrilinosSchurAMG.*'
```

No CUDA execution or GPU-aware MPI speedup has been validated on this Mac.
The general FSI/MueLu scratch-memory failure is still unresolved and is separate
from this optional Schur coarse-solver implementation.
