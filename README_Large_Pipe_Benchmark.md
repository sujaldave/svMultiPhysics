# Large-mesh benchmark: Lasso's 1M-tetrahedra pipe

The iliac artery (15,334 nodes) is too small for one GPU to beat a CPU
workstation: on one A40 every solver kernel is latency-bound. This benchmark
runs the current branch, unchanged, on the pipe Andras Lasso used for his
37× GPU result, so CPU and GPU can be compared at a size where GPU throughput
matters.

## Case

`benchmarks/large_pipe/make_pipe.py` is Lasso's generator
([fc37b11](https://github.com/lassoan/svMultiPhysics/blob/fc37b11/Code/Source/gpu/make_pipe.py)):
a pipe with radius 1 cm and length 24 cm, 181,875 nodes and 1,002,240 TET4s.
One line differs. The six tetrahedra of each hexahedron are numbered
consecutively, instead of in six blocks that each span the whole pipe. The mesh
is the same apart from element numbering. With the original numbering, the
bundled ParMETIS crashes in `Match_Global` (bus error) on 4, 8, 10 and 12
ranks every time; 2 ranks work. With the new numbering, partitioning succeeds.

Lasso did not commit a solver input. `benchmarks/large_pipe/solver.xml` uses the
iliac physics, BC types and values, time step and tolerances: parabolic inflow
of 100 mL/s, one resistance outlet (16,000) with backflow stabilization, and a
no-slip wall. It runs 10 time steps, as Lasso's did, with output written only at
the final step.

```sh
# Generate once (needs NumPy); ~4 s, 53 MB of ASCII VTK
benchmarks/large_pipe/prepare_case.sh build/benchmarks/large-pipe/case python3
# Any case directory now works with the existing runner
python3 benchmarks/schur_amg/run_benchmark.py --case build/benchmarks/large-pipe/case ...
```

## M4 Pro CPU results

Commit `244b697` (this change adds only benchmark tooling). Apple M4 Pro,
10 performance + 4 efficiency cores, Trilinos 16.2 / Kokkos Serial,
`RelWithDebInfo`, one thread per rank. Configuration: Schur AMG pressure CG with
resistance-preconditioned GMRES and native Kokkos assembly. All runs were
sequential.
Reproduce with `benchmarks/large_pipe/run_cpu.sh`; compact results are in
[`benchmarks/large_pipe/cpu_results.json`](benchmarks/large_pipe/cpu_results.json).

**Pipe, 10 steps, 85 Newton iterations at every rank count:**

| Ranks | Wall s | CG iters (per call) | CG s | µs/CG iter | Schur setup s | AMG setup s | GMRES s | Assembly s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 701.1 | 21,568 (56.6) | 484.7 | 22,474 | 17.06 | 3.24 | 55.16 | 95.61 |
| 2 | 391.1 | 21,890 (57.5) | 261.8 | 11,960 | 12.59 | 1.80 | 33.84 | 50.14 |
| 4 | 205.6 | 22,256 (58.4) | 135.1 | 6,072 | 6.88 | 1.02 | 16.39 | 26.63 |
| 8 | 135.8 | 21,574 (56.6) | 83.5 | 3,872 | 4.57 | 0.70 | 14.59 | 15.84 |
| 10 (2 repeats) | **128.1** | 21,142 (55.5) | 77.9 | 3,683 | 4.30 | 0.71 | 14.94 | 13.58 |
| 10, classical diagonal CG | 718.7 | 444,900 (300.0) | 608.6 | 1,368 | — | — | 58.18 | 18.15 |

The pressure CG has zero inner-solve failures at every rank count, and final
fields agree with the 10-rank run to at most 2.7e-13 (velocity) and 6.1e-15
(pressure) relative L2. From 1 to 10 ranks the speedup is 5.5×; from 8 to 10
ranks it is only 6%, as the run approaches the M4's memory-bandwidth limit.

The **classical BIPN fails on this mesh**. It records 89 failed RI solves and
1,483 capped pressure solves; Newton reaches its 12-iteration cap in several
steps (106 Newton iterations in total). Its final pressure differs from the AMG
solution by 1.3% relative L2. Schur AMG is needed at this size, not just faster.

For context, Lasso reports 1,631 s on 10 CPU processes (FSILS, i9-10900X) and
43.9 s of solver time on one RTX 3090 for this mesh. The current CPU Schur-AMG
path takes 122.5 s of simulation time on 10 M4 cores. Most of his 37× therefore
comes from the weak FSILS baseline; the remaining GPU gap must be measured on
Sherlock.

**Iliac control, 20 steps, 184 Newton iterations:**

| Ranks | 1 | 2 | 4 | 8 | 10 |
|---|---:|---:|---:|---:|---:|
| Wall s | 138.8 | 76.1 | 43.4 | 27.0 | 25.3 |
| µs/CG iter | 2,003 | 1,059 | 596 | 368 | 346 |

These iliac times are lower than the 34.1–35.5 s measured at 10 ranks in
earlier sessions with almost the same code, so treat cross-session CPU
comparisons as ±25%. Compare runs from the same sweep.

Per CG iteration, the pipe costs 11.2× the iliac on one rank, close to its 11.9×
node ratio. On one A40 the iliac costs 368 µs per iteration, mostly fixed launch
and synchronization cost, so the pipe should cost much less than 11× that on the
GPU. The Sherlock job measures this.

## Run on one Sherlock A40

No C++ rebuild is needed; the existing `1227395`+ GPU executable is fine
(this branch's later commits add an optional coarse solver, off by default).
Sync the new scripts, then either generate the case on Sherlock (needs NumPy,
for example `pip install --user numpy` or a NumPy module):

```bash
SVMP_PYTHON=python3 sbatch benchmarks/large_pipe/sherlock_a40.sbatch
```

or copy the generated case from the Mac and point at it:

```bash
rsync -a build/benchmarks/large-pipe/case/ sherlock:/scratch/users/$USER/pipe-case/
SVMP_PIPE_CASE=/scratch/users/$USER/pipe-case sbatch benchmarks/large_pipe/sherlock_a40.sbatch
```

The job reserves one A40 for 90 minutes and runs, sequentially:

| Directory | Purpose |
|---|---|
| `pipe/runtime-details` | 1 step: Kokkos device and AMG hierarchy sizes (not timed) |
| `pipe/gpu-amg-resistance` | 10 steps, 2 repeats: the measurement |
| `iliac/gpu-amg-resistance` | 20-step iliac control in the same allocation (expect ≈39.5 s) |
| `pipe/gpu-classic-resistance` | 10 steps: classical CG reference; expected to fail as on CPU |
| `pipe/profile` | Only with `SVMP_PROFILE=1`: 1-step Nsight Systems trace |

Overrides: `SVMP_STEPS`, `SVMP_REPEAT`, `SVMP_CLASSIC=0`, `SVMP_ILIAC=0`,
`SVMP_PROFILE=1`, `SVMP_EXE`, `SVMP_OUTPUT`. The job writes `report.md`, and a
`summary.json` per case, under `build/benchmarks/large-pipe-JOBID/`. The
classical group is expected to report outer-solve failures, so the job's exit
status can be nonzero even when the AMG runs succeed; check `report.md` and
`summary.json`. Copy the whole directory back and regenerate the summaries with
VTK field checks:

```bash
python3 benchmarks/solver_optimization/summarize.py /path/large-pipe-JOBID/pipe \
  --reference-group gpu-amg-resistance --expect-cuda --allow-diagonal-inner-nonconvergence
python3 benchmarks/schur_amg/compare_fields.py \
  build/benchmarks/large-pipe/cpu-m4/pipe/cpu-10/run-1/10-procs/result_010.vtu \
  /path/large-pipe-JOBID/pipe/gpu-amg-resistance/run-1/1-procs/result_010.vtu
python3 benchmarks/large_pipe/report.py build/benchmarks/large-pipe/cpu-m4/pipe \
  /path/large-pipe-JOBID/pipe
```

### Optional CPU-node reference

`benchmarks/large_pipe/sherlock_cpu.sbatch` runs both cases on 8, 16 and 32
ranks of one CPU node. It needs a **CPU-only** Trilinos build of
svMultiPhysics; a CUDA-built executable runs Tpetra on the GPU.
Use `--constraint=CPU_SKU:7543P` to match the A40 node's host CPU:

```bash
SVMP_CPU_EXE=/path/to/cpu-build/svmultiphysics \
  sbatch --constraint=CPU_SKU:7543P benchmarks/large_pipe/sherlock_cpu.sbatch
```

The same script works on Stampede3 with its partition names and a larger
`SVMP_RANKS`/`--ntasks`.
