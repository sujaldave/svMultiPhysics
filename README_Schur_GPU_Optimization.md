# Schur solver optimization after the first Sherlock A40 test

This change addresses the dominant costs identified in the October 27 A40 audit:
repeated AMG construction, factored pressure applications, separate outlet
reductions, and scalar host writes during system preparation. The physics,
resistance/cap convention, tolerances, and FP64 precision are retained.

## What changed

**Schur AMG transfer reuse.** The default Schur policy is now MueLu `RP` reuse.
Aggregation and transfer operators survive Newton iterations and time steps on
the same topology. Each Jacobian refreshes the fine Schur matrix, all coarse
Galerkin matrices, smoother numerical data, and the coarse factorization.
`RAP` reuse was deliberately avoided: in the installed Trilinos implementation
it also retains coarse numerical matrices. A scaling test verifies that doubling
the matrix halves the reused inverse action, catching stale coarse operators.

Reuse checks the topology generation, full local CSR structure, column-map and
domain/range ordering, and a newly calculated spectral bound. Each rank checks
its local graph on its active device; the final rebuild decision is collective.
A graph change with unchanged dimensions/nnz still forces a rebuild. Damping uses
1.8 divided by a bound with 5% headroom; exceeding the retained bound forces a
fresh hierarchy. This avoids retaining an unsafe fine-level Jacobi damping value.
Fresh mode uses the same headroom for comparison. Each cache belongs to one
equation and is released before Kokkos finalization.

**Cached Schur matrix structure.** The B^T B product reuses its sparse graph.
The union graph for S0=L+B^T B and device scatter plans are built once per block
topology. Each Jacobian recomputes the product's numerical values and refreshes
S0 on the active device. Tpetra still performs the distributed sparse product
and transpose communication; this does not replace SpGEMM with a custom kernel.

**Explicit pressure operator.** With Schur AMG, the default inner operator applies
S0 plus exact low-rank outlet corrections. For scaled outlet v, cap c, and
coefficient a=-sign(R)/(1+R||S_face||²), the correction is

```
S x = S0 x + sum_faces a (B^T v) [B^T(v+c)]^T x.
```

The pressure-space update/projection vectors are computed once per Jacobian.
This replaces repeated L, B, resistance, and transpose-B applications with one
Schur sparse multiply plus a batched low-rank update. The factored operator is
available for comparison. Other pressure preconditioners retain their factored
operator.

**Batched outlet work.** Momentum and resistance transformations pack all active
outlets into multivectors. Tpetra computes all projections, including all RHS
columns, in one distributed matrix multiply/reduction, followed by one batched
update. Cap vectors participate only in the projection. Ranks with no local
outlet nodes still participate. Inactive faces are excluded using the replicated
resistance metadata. This retains MPI correctness without implementing a
single-GPU cooperative solver.

**Boundary preparation.** Host face/cap workspaces and ghost Tpetra vectors are
reused. Boundary values and the Dirichlet scaling mask are populated through
bulk host views, avoiding per-entry Tpetra setter calls. Inactive faces require
no boundary upload/export. CPU boundary integration and state handling remain
part of the simulation.

## Controls and diagnostic output

| Environment variable | Values | Default |
|---|---|---|
| `SV_TRILINOS_SCHUR_REUSE` | `transfers`, `fresh` | `transfers` |
| `SV_TRILINOS_SCHUR_OPERATOR` | `explicit`, `factored` | `explicit` with Schur AMG |
| `SV_TRILINOS_BIPN_STATS` | Set to enable | Off |
| `SV_TRILINOS_RUNTIME_DETAILS` | Set to print Kokkos build/device configuration | Off |
| `SV_TRILINOS_AMG_DETAILS` | Set to print each freshly built AMG hierarchy | Off |

`BIPN_STATS` now includes `operator_setup_s`, `scaling_s`, `block_setup_s`, and
cumulative cached `amg_builds`/`amg_refreshes` (fresh mode bypasses this cache
and leaves those counters at zero). `BOUNDARY_SETUP_STATS` measures preparation
outside the BIPN solve timer. Existing assembly, CG, GMRES, convergence, and true
residual fields remain available. Setup subphases overlap the BIPN total: do not
sum them a second time. Rank timings are reduced using the maximum per Jacobian.

The benchmark helper accepts `--schur-reuse`, `--schur-operator`,
`--runtime-details`, and `--profile nsys`. A profile is a separate short run,
because trace collection and verbose configuration output affect timing.

## CPU verification and timing

The reference executable was preserved from commit `3a59fa2` before recompiling.
Both versions use the checked-in iliac artery mesh (15,334 nodes, 75,819 TET4s),
20 time steps, ten MPI ranks, one thread per rank, Schur AMG, and Kokkos assembly.
The local environment is Apple M4 Pro / CPU-only Trilinos 16.2 / Kokkos Serial.
Runs are sequential, with identical final-only output and unchanged tolerances.

| Configuration | Wall seconds | Median |
|---|---|---:|
| Preserved `3a59fa2` executable | 45.088, 46.899 | 45.994 s |
| New default: reuse + explicit | 35.142, 34.253 | 34.697 s |
| New fresh + factored (one ablation run) | 39.377 | 39.377 s |
| New reuse + factored (one ablation run) | 40.755 | 40.755 s |

The default is **1.326× faster**, or **24.6% less wall time**, than the preserved
version. All full runs complete 184 nonlinear iterations with zero reported CG
or true-residual failures. Both new default repeats use one AMG build and 183
numeric refreshes. Mean summed AMG setup falls from 1.073 to 0.352 seconds;
Schur matrix setup falls from 2.335 to 1.040 seconds, and pressure CG from 31.363
to 24.391 seconds. These per-Jacobian maximum-rank phase sums are diagnostic;
they are not independent additive estimates of the total wall-time reduction.

AMG reuse alone does **not** improve total CPU time in the single-run ablations:
its cheaper setup is outweighed by CG timing variation and slightly different
iterations. Fresh/factored uses 50,884 CG iterations; reuse/factored uses 51,018;
the optimized explicit runs use 51,161, versus 50,262 in the old executable.
The final-state errors remain at roundoff. The larger GPU setup bottleneck
identified in the A40 audit must be remeasured on Sherlock; CPU results do not
establish GPU speedup. The two configurations also share the new boundary,
batched-outlet, and Schur-storage changes, so fresh/factored is not an old-code
reference.

All six output fields pass the 1e-7 relative-L2 comparison against the preserved
executable. For the default optimized runs, velocity differs by 2.46e-14 and
pressure by 1.63e-15. The largest relative difference among the six fields is
3.25e-13 (divergence). The two default repeats produce identical final fields.

The complete CPU unit suite passes **363 tests with one expected skip** (the
shared-row export test requires exactly two ranks). The selected two-rank MPI
suite passes **13 tests on each rank**. Tests cover native assembly, shared-row
numeric export, distributed cached Schur products, multi-RHS outlet/cap
corrections, numerical coarse-level refresh, graph changes with unchanged nnz,
and damping-bound invalidation. A build with Trilinos disabled also succeeds.
The separate runtime-details smoke run successfully prints Kokkos 4.7.1 Serial
and the three-level CPU hierarchy.

One ablation launch crashed in ParMETIS mesh partitioning before reaching solver
setup. Its log is preserved at `build/benchmarks/solver-opt-failed-fresh-factored`;
the identical retry completed and is the measured run above. No failed sample
is counted as a timing result.

Versioned timings and field checks are in
`benchmarks/solver_optimization/cpu_results.json`. Local raw logs, measurements,
XML files, and results are under `build/benchmarks/solver-opt-cpu/`; the full
machine-readable summary is `build/benchmarks/solver-opt-cpu/summary.json`.

## Submit the Sherlock comparison

Before rebuilding on Sherlock, preserve the executable used for the first A40
test. Run these commands from the repository root, with the same loaded modules
and Trilinos prefix used for the successful GPU build:

```bash
mkdir -p build/before-schur-optimization
cp -p build/svMultiPhysics-build/bin/svmultiphysics \
  build/before-schur-optimization/svmultiphysics
```

Then update the source and rebuild. The inner build command reliably recompiles
changed solver sources even when the outer superbuild stamps are already current:

```bash
cmake --build build/svMultiPhysics-build -j4
export SVMP_BASELINE_EXE="$PWD/build/before-schur-optimization/svmultiphysics"
sbatch benchmarks/solver_optimization/sherlock_a40.sbatch
```

The default job requests **one A40 and one CPU task for 15 minutes**, runs five
steps per comparison and two repeats for old/default-new timings, plus separate
two-step configuration/profile runs. For the full comparison matching the first
20-step experiment:

```bash
SVMP_STEPS=20 sbatch benchmarks/solver_optimization/sherlock_a40.sbatch
```

The job runs these variants sequentially:

| Directory | Executable / settings | Purpose |
|---|---|---|
| `old` | Preserved executable, if supplied | Genuine before/after reference |
| `fresh-factored` | New, fresh AMG, factored Schur | New-code control |
| `reuse-factored` | New, transfer reuse, factored Schur | Isolate hierarchy reuse |
| `reuse-explicit` | New default, repeated twice | Combined solver changes |
| `legacy-assembly` | New default solver, CPU element loop in GPU build | Isolate native assembly |
| `runtime-details` | New, two steps, verbose runtime/hierarchy | Check architecture and hierarchy sizes |
| `profile` | New, two steps with Nsight Systems, when available | Attribute launches, copies, and synchronization |
| `cpu` | Optional separate CPU executable | CPU/GPU comparison on the same host |

Without `SVMP_BASELINE_EXE`, the script runs the other variants and clearly uses
`fresh-factored` as its reference; that comparison is not the old implementation.
All outputs use a job-specific directory under `build/benchmarks/sherlock-JOBID`.
The helper refuses to overwrite prior run directories.

Optional environment overrides:

```bash
export SVMP_BUILD_DIR="$PWD/build-gpu"       # If your build has a different name
export SVMP_CPU_EXE=/absolute/path/to/cpu/svmultiphysics
export SVMP_TRILINOS_BUILD=/absolute/path/to/trilinos-build
export SVMP_OUTPUT=/absolute/path/to/new/results-directory
```

`SVMP_TRILINOS_BUILD` lets the job copy the library's CMake cache as well as the
application caches. The script records source commit/diff, executable hashes,
loaded modules, GPU information, Slurm allocation, CUDA compiler, linked
libraries, all run logs, measurements, and available unit tests. Nsight Systems
and the unit-test executable are optional; unavailable tools are recorded
explicitly. Load Nsight Systems in the submission environment to get the trace.

The summary checks completion, outer convergence, pressure-CG failures, true
residual failures, and that native GPU assembly reports `Cuda`. If Python VTK is
available it also compares all six final fields. Otherwise timing checks finish
and field validation is marked **pending**. After copying the results here, run:

```bash
python3 benchmarks/solver_optimization/summarize.py /absolute/path/to/copied/results
```

This generates `summary.json`, including phase totals, hierarchy counters,
speedups, and field errors. Check that `amg_refreshes` increases and repeated
Jacobian setup does not keep rebuilding the hierarchy. A new `amg_builds` count
can legitimately indicate graph/topology change or a larger spectral bound.

For multiple GPUs, use one MPI rank per GPU. After the one-GPU run, for example:

```bash
SVMP_RANKS=2 sbatch --ntasks=2 --gpus-per-node=2 \
  benchmarks/solver_optimization/sherlock_a40.sbatch
```

Use larger cases for meaningful scaling conclusions; this artery mesh has only
15k nodes. The optional CPU comparison in this script uses one CPU rank; CPU
scaling at 2/4/8 ranks should be a separate controlled experiment.

## GPU limitations and follow-up

There is no CUDA compiler or NVIDIA GPU on this Mac. CPU/MPI verification does
not establish GPU speedup or successful NVCC compilation; the supplied job is
the GPU validation step. New device lambdas are in ordinary functions rather
than constructors to avoid the earlier NVCC extended-lambda error.

The initial A40 logs showed compute capability 8.0 kernels on an 8.6 device.
Rebuild the **Trilinos/Kokkos library** for `Kokkos_ARCH_AMPERE86=ON` and disable
`Kokkos_ARCH_AMPERE80`, then rebuild the application against that installation.
Changing only the application's flags does not retarget prebuilt library kernels.
For a controlled code-only comparison, first compare old/new against the same
library; measure the architecture rebuild separately.

The coarse solve is still KLU and the inner arithmetic remains FP64. GPU coarse
solver alternatives, mixed precision, and deeper Krylov fusion require the new
trace to establish the remaining costs and must be validated independently.
These changes therefore do not promise Lasso's 11.7-second time or a particular
multi-GPU speedup.
