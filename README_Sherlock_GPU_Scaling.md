# A40 results and one-node 1/2/4-GPU scaling

## What the completed jobs show

Both new jobs use solver commit `1227395`, one A40, Schur AMG, and unchanged
physics/tolerances. Job `49893347` ran five steps (48 nonlinear iterations);
job `49893846` ran twenty steps (184 nonlinear iterations). Their batch stderr
files are empty, and all recorded simulation runs completed successfully.

| Configuration | 5-step wall time | 20-step wall time |
|---|---:|---:|
| New code, fresh AMG + factored Schur | 22.551 s | 72.053 s |
| New code, reused transfers + factored Schur | 15.277 s | 42.378 s |
| New default, reused transfers + explicit Schur, repeat 1 | 14.717 s | 40.511 s |
| New default, repeat 2 | 14.723 s | 40.448 s |
| New default, legacy CPU element loop in the GPU build | 50.313 s | 184.642 s |

The default median is **40.479 s** for twenty steps. That is **1.780× faster**
than the new-code fresh/factored control and **4.561× faster** than legacy
assembly with the same optimized solver in this GPU build. The latter is not a
comparison against a separately optimized CPU-only Trilinos executable.

The earlier October 27 GPU experiment took 86.800/84.822 s (median 85.811 s).
The current result is approximately **2.120× faster**, or 52.8% less wall time,
against that historical result. Neither new job includes an `old/` directory,
so the old executable was not measured in the new allocations. The historical
comparison therefore includes possible run/environment differences.

The twenty-step phase measurements explain the improvement:

| Phase | Fresh/factored | Reuse/factored | Reuse/explicit, mean of repeats |
|---|---:|---:|---:|
| Pressure AMG setup | 32.104 s | 2.726 s | 2.841 s |
| Pressure CG | 21.476 s | 21.313 s | 19.392 s |
| Momentum GMRES | 3.693 s | 3.665 s | 3.670 s |
| Schur matrix setup | 4.694 s | 4.712 s | 4.706 s |
| Fluid assembly, including initial cache | 1.542 s | 1.550 s | 1.496 s |

Transfer reuse removes approximately 29 seconds of AMG setup. The reused runs
show **one build and 183 numerical refreshes**, so reuse survives time-step
boundaries as intended. Explicit Schur application saves about another two
seconds overall compared with reuse/factored. Pressure iteration counts are
52,530 (fresh/factored), 52,486 (reuse/factored), and 52,908/53,017 (explicit).
The speedup comes from cheaper setup/application despite slightly more CG work.

Native assembly remains CUDA, rebuilds its geometry cache once, and uses the
single-rank `local-copy` export. Legacy element assembly alone takes 143.924
seconds in the twenty-step GPU-build run, versus approximately 1.496 seconds
for the native path including cache setup. The native compute kernel alone
takes approximately 1.181 seconds. Assembly is now only about 3.7% of total
wall time; solver execution and setup remain the main costs.

All runs report zero pressure-CG failures, zero true-residual failures, and
successful outer RI solves. I reran field comparisons locally because VTK was
unavailable in the cluster summary environment. All six output fields pass the
1e-7 relative-L2 criterion against their job's fresh/factored reference. For
the twenty-step explicit repeats, velocity error is approximately 2.5e-14,
pressure at most 1.9e-15, and the largest error among all six fields is about
3.4e-13 (divergence). These are roundoff-level differences.

The build configuration confirms `RelWithDebInfo`, `-O2`, Kokkos 4.7.1 CUDA,
and disabled debug bounds checking. The architecture warning persists:
`AMPERE80` library kernels are running on A40 capability 8.6. Keep the library
unchanged during the scaling sweep to isolate GPU count; benchmark an
`AMPERE86` rebuild separately. GPU unit tests did not run because their
executable was unavailable; the output/solver checks above did run.

## What the profiles imply

Both Nsight captures are separate two-step runs, not timing benchmark samples.
The twenty-step job's two-step capture contains:

- **208,045 GPU kernel launches**, totaling 2.160 seconds of kernel execution,
  with a mean duration of 10.38 microseconds.
- 58,537 `cudaStreamSynchronize` calls and 48,529 `cudaDeviceSynchronize` calls,
  accounting for approximately 1.132 and 0.557 seconds of CUDA API time.
- Approximately 0.815 seconds in `cudaLaunchKernel`, 0.346 seconds in synchronous
  `cudaMemcpy`, and 104,607 `cudaMalloc` API calls. Counts include setup and
  library activity; they do not identify a particular source allocation site.
- The largest kernel groups are Tpetra local residual/Jacobi work, sparse
  multiplies, and batched dot/GEMM operations. The fluid kernel executes 21 times
  at approximately 6.39 ms each and uses 255 registers per thread. Register
  pressure is a tuning lead, not proof of spills or the dominant wall bottleneck.
- Three AMG levels with 15,334, 286, and 21 rows; the coarse solve is still KLU2.

API time overlaps GPU execution: do not add these totals to reconstruct wall
time. CPU sampling and MPI tracing were not enabled, so this is evidence of
many short launches/synchronizations, not a complete attribution of every wait.
The five-step job's separate profile shows the same pattern.

The single-GPU optimization is promising. Multi-GPU scaling remains unmeasured.
At two GPUs the average owned nodes per GPU is only 7,667; at four it is about
3,834, plus ghosts. Kernel launches, host coarse solves, reductions, and MPI
communication may dominate as local work shrinks. Additional GPUs could provide
little gain or increase wall time on this case. Larger meshes will eventually
be needed to assess throughput and scalability beyond this small test.

Extracted measurements, field checks, and profile totals are versioned in
`benchmarks/gpu_scaling/sherlock_results_2026-10-27.json`. The downloaded job
directories now also have locally refreshed `summary.json` field comparisons.

## Submit the matched one-node scaling sweep

Sync the new scripts to Sherlock and use the same loaded modules and GPU
executable as the successful experiment. There are no solver-source changes
in this update, so the existing `1227395` solver binary can be used.
From the repository root:

```bash
sbatch benchmarks/gpu_scaling/sherlock_a40_scaling.sbatch
```

The script requests **one node with four A40 GPUs**, reserves the node
exclusively, and runs **1, 2, and 4 GPU configurations sequentially**. Each uses
one MPI rank per GPU and one host thread per rank, twenty steps, and three
independent repeats. A separate two-step diagnostics run precedes each group;
it is excluded from scaling results. No profiler runs during the timed sweep.
The time limit is thirty minutes; if scaling is reasonable the job should
finish much earlier. GPU resources are reserved for the whole sweep, including
the 1/2-GPU steps, which gives a matched same-node comparison.

The script uses Slurm `srun --exact`, `--gpus-per-task=1`, and
`--gpu-bind=single:1`, plus CPU core binding. See the official
[srun documentation](https://slurm.schedmd.com/srun.html) and
[GPU resource guide](https://slurm.schedmd.com/gres.html).
Do not manually set `CUDA_VISIBLE_DEVICES`; Slurm assigns it per task.

Before each group, `check_binding.py` uses the CUDA runtime to verify exactly
one visible GPU per rank and records its PCI bus address. The job checks that
all ranks are on one host and use distinct physical GPUs. This catches the
case where four MPI ranks accidentally share one device. The probe is outside
the timed simulations and needs no Python CUDA packages.

For a shorter smoke sweep:

```bash
SVMP_STEPS=5 SVMP_REPEAT=2 sbatch benchmarks/gpu_scaling/sherlock_a40_scaling.sbatch
```

To submit two and four GPUs as separate jobs instead:

```bash
SVMP_GPU_COUNTS="2" sbatch --ntasks=2 --gpus-per-node=2 \
  benchmarks/gpu_scaling/sherlock_a40_scaling.sbatch
SVMP_GPU_COUNTS="4" sbatch --ntasks=4 --gpus-per-node=4 \
  benchmarks/gpu_scaling/sherlock_a40_scaling.sbatch
```

The exclusive-node reservation still applies to those jobs. They may land on
different nodes and do not include a one-GPU reference, so the default combined
sweep is preferred for the first scaling comparison. GPU counts, steps, and
repeats are recorded in `sweep.json`. `SVMP_REPO`, `SVMP_BUILD_DIR`, `SVMP_EXE`,
`SVMP_OUTPUT`, and optional `SVMP_TRILINOS_BUILD` override the corresponding paths.
Outputs must use a new directory; existing results are never overwritten.

## Outputs and how to judge scaling

Results go to `build/benchmarks/scaling-JOBID/`:

```
config/                 Slurm, modules, executable hash, caches, GPU topology
binding/gpus-1/         Per-rank CUDA visibility and physical GPU identity
binding/gpus-2/
binding/gpus-4/
warmup/gpus-N/          Separate two-step runtime/hierarchy diagnostics
gpus-1/run-1/ ...        Three timed cases, XML, histories, fields, statistics
gpus-2/run-1/ ...
gpus-4/run-1/ ...
scaling.json            Detailed measurements, phase totals and validation
scaling.csv             Compact table for plotting
scaling.md              Readable comparison table
```

The job captures selected GPU/MPI environment variables and `ompi_info` so we
can inspect the MPI build. It does not force GPU-aware MPI: transport support
must be verified before changing that setting. It also captures GPU topology
and configuration before/after each group.

The summary reports medians and min/max wall times, simulation-history time,
summed maximum-rank solver time, pressure iteration counts, hierarchy counters,
and matrix export transport. Expect `numeric-values` matrix export at multiple
ranks, rather than the one-rank `local-copy` path. Different partitions can
change AMG aggregates and Krylov iteration counts; those changes must be
considered when interpreting the timing.

For GPU count N:

```
wall speedup = median wall time on 1 GPU / median wall time on N GPUs
efficiency  = wall speedup / N
GPU-seconds = N * median wall time
```

Consistently lower 2/4-GPU wall times beyond the repeat-to-repeat spread would
be evidence of useful strong scaling. If solver time improves but wall time
does not, startup and other host work may be limiting the complete simulation.
If GPU-seconds rises substantially, the reduced latency costs more GPU resource
time. No numerical efficiency threshold is imposed; inspect both speed and cost.

The summary requires matching XML, checks all expected repeats, completion,
CUDA assembly, outer convergence, and pressure residual failures. It compares
all six final fields across every repeat/count when Python VTK is available.
Otherwise the field check is explicitly pending. After downloading the whole
sweep directory to this Mac, run:

```bash
python3 benchmarks/gpu_scaling/summarize_scaling.py \
  /absolute/path/to/scaling-JOBID --expect-cuda
```

This refreshes the JSON/CSV/Markdown reports and validates fields against the
same sweep's one-GPU run. If a simulation or binding check fails, the job stops
and attempts a partial summary; incomplete groups are flagged. Keep all logs
and output fields when copying results.

The tooling was checked locally with synthetic scaling/binding fixtures,
negative residual/repeat/input tests, and Python/shell syntax checks. The CUDA
binding probe and Slurm allocation behavior require the cluster run; no 2/4-GPU
performance numbers are inferred from these local checks.
