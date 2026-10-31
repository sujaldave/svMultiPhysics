# Solver options for GPU runs: Schur product, AMG smoothing/reuse, CG reductions

This change attacks the setup and per-iteration costs measured on one A40 for
Lasso's 1M-tetrahedra pipe (job 50440494: 112 s, of which pressure CG 35 s,
Schur `L + BᵀB` setup 24 s). Two changes are on by default; the rest are
environment options so the GPU sweep can choose defaults from measurements.
Physics, tolerances and FP64 are unchanged; every variant below reproduces the
old final fields to roundoff (relative L2 ≤ 3e-13).

## What changed and where it applies

| Change | Applies to | Default | Control |
|---|---|---|---|
| Cached `BᵀB` (transpose and SpGEMM symbolic once; numeric only per Jacobian) | Fluid BIPN, 1 rank | **on for device builds** | `SV_TRILINOS_SCHUR_PRODUCT=auto\|cached\|tpetra` |
| One reduction per CG iteration (Belos "Fold Convergence Detection Into Allreduce") | Every CG: BIPN pressure, solid, mesh motion | **on** | `SV_TRILINOS_CG_FOLD=1\|0` |
| Power-iteration Jacobi damping instead of Gershgorin | Schur AMG | off | `SV_TRILINOS_SCHUR_DAMPING=power` |
| Smoother sweeps / Chebyshev (per-level λmax) / symmetric multicolor GS | Schur AMG | Jacobi, 2 sweeps | `SV_TRILINOS_SCHUR_SMOOTHER`, `SV_TRILINOS_SCHUR_SWEEPS` |
| Levels, coarse size, aggregate size, drop tolerance | Schur AMG | 10, 200, 3–12, 0 | `SV_TRILINOS_SCHUR_LEVELS`, `_COARSE_SIZE`, `_AGG_MIN`, `_AGG_MAX`, `_DROP_TOL` |
| Coarse solver on the device | Schur AMG and generic MueLu | KLU | `SV_TRILINOS_SCHUR_COARSE` / `SV_TRILINOS_MUELU_COARSE` = `klu\|tacho\|cusolver` (Schur also `chebyshev`) |
| AMG hierarchy reuse across Newton iterations and time steps | Generic MueLu (`trilinos-ml`): FSI, solid, scalar, mesh, any GMRES/CG equation | off (rebuilt each solve) | `SV_TRILINOS_MUELU_REUSE=none\|tP\|RP` |
| Refresh AMG once per time step | Schur AMG and generic MueLu | every Jacobian | `SV_TRILINOS_AMG_REFRESH=jacobian\|step` |
| Generic smoother choice incl. device-parallel ones | Generic MueLu | Gauss-Seidel | `SV_TRILINOS_MUELU_SMOOTHER`, `SV_TRILINOS_MUELU_SWEEPS` |

Shared option parsing lives in `Code/Source/solver/TrilinosSolverOptions.{h,cpp}`
and is used by both the BIPN (`TrilinosPreconditionerFactory.cpp`) and generic
(`trilinos_impl.cpp`) paths, so a smoother or coarse solver behaves the same in
every physics. Invalid values throw with the list of accepted choices; a coarse
solver missing from the Trilinos build throws instead of MueLu's silent KLU fallback.

**Cached `BᵀB`.** Tpetra's `Multiply(B, true, B, false)` re-forms the transpose
matrix on every call; on the A40 that is ~285 ms per pipe Jacobian, while the
SpGEMM kernels themselves are a few ms. The cached path builds `Bᵀ` (with the B
value index of each entry) once on the host, runs the KokkosKernels symbolic
phase once, and per Jacobian gathers `Bᵀ` values, runs the numeric phase and
scatters into `S0` through a precomputed plan. It is verified once against the
Tpetra-built `S0` (falls back with a warning on any mismatch). Tpetra's product
graph omits structurally present but numerically zero entries (Dirichlet-scaled
rows: 58,100 of 883,646 on iliac); each refresh checks those stay zero and
rebuilds `S0` (and therefore the AMG) otherwise. On host execution spaces
Tpetra's own Gustavson kernel is faster (16 vs 22 ms per iliac Jacobian on one
M4 core), so `auto` keeps Tpetra there. MPI runs keep Tpetra's distributed product.

**Generic MueLu reuse.** The generic path previously built a complete smoothed-
aggregation hierarchy for every linear solve. `RP` keeps aggregates and transfer
operators, recomputing coarse operators and smoothers from the current matrix;
`tP` also re-smooths the prolongator. With strength-of-connection dropping, `tP`
can change the prolongator's sparsity, which a reused coarse graph cannot hold
(FSI: "insert a new entry into a static graph"). A failed refresh therefore
rebuilds the hierarchy (the old behavior) with a one-time warning. **Use `RP`.**
Multi-rank note: that fallback relies on every rank failing together; `RP` keeps
the coarse pattern fixed and does not need it.

**Correction to the earlier plan (item 6).** The 4.1 s "storage reset" on the A40
pipe is almost entirely the first Jacobian (3.75 s topology build); later
Jacobians take 3.7 ms. There is no recurring 48 ms cost to remove. One-time GPU
setup remains (topology 3.75 s, block extraction 7.5 s, assembly cache 1.9 s).

## Local CPU results (Apple M4 Pro, Trilinos 16.2 Serial)

Fluid, BIPN with Schur AMG + resistance GMRES, 10 ranks:

| Variant | Iliac 20 steps wall | Iliac CG s (µs/iter) | Pipe 10 steps wall | Pipe CG s (µs/iter) |
|---|---:|---:|---:|---:|
| Old code (`SCHUR_PRODUCT=tpetra CG_FOLD=0`) | 26.1 | 18.3 (359) | 141 (145.9, 136.7) | 91.2 / 84.6 (4,314 / 4,002) |
| New defaults (folded CG) | 25.6 | 17.9 (350) | 133.0 | 81.7 (3,886) |
| `DAMPING=power SWEEPS=1` | 21.0 | 12.9 (234) | 109.1 | 57.5 (2,404) |
| `SMOOTHER=chebyshev SWEEPS=1` | 20.9 | 12.7 (240) | 107.4 | 55.3 (2,433) |
| `DAMPING=power SWEEPS=1 AMG_REFRESH=step` | **19.0** | 11.6 (211) | 108.9 | 57.8 (2,425) |
| power + 2 levels + aggregates 2–6 | 20.1 | 11.9 (231) | 147.6 | AMG setup 10.1 s |

Power damping (Rayleigh-quotient λmax, ×1.1, capped by Gershgorin) gives 12%
fewer iterations at equal sweeps; one sweep then halves fine-level smoothing for
~8% more iterations. On the pipe the A40 is bandwidth-bound like these CPU runs,
so the same ~35% CG reduction is expected there. Two-level hierarchies help the
small mesh but cost 10 s of exact coarse factorization on the pipe (2,303 rows):
not recommended for large meshes. Iterations per solve remain ~55–75, above
Lasso's reported ~36; neither damping, sweeps, levels, aggregate sizes nor
dropping closes that gap. Multicolor symmetric GS needs the fewest iterations
but is 3× slower per iteration on the CPU (GPU-only candidate); Tacho was not
faster than KLU on the CPU.

Generic solve, GMRES + MueLu, one rank, 20 steps:

| Physics | Old (rebuild every solve) | `MUELU_REUSE=RP AMG_REFRESH=step` | Preconditioner setup |
|---|---:|---:|---|
| FSI (`fsi/pipe_3d`) | 19.0 s | **14.7 s (1.29×)** | 4.75 → 0.51 s |
| Scalar transport (`precomputed_dye_AD`) | 125.8 s | **107.7 s (1.17×)** | 18.7 → 1.05 s |
| Solid (`linear-elasticity/beam`) | 4.4 s | **3.3 s (1.33×)** | 0.22 → 0.05 s |

Linear iterations are unchanged for solid and FSI (8,740 and 13,889 → 13,892)
and 5% higher for scalar transport (4,040 → 4,244, from the once-per-step
refresh); final fields agree to ≤ 2.5e-13. Two-stage Gauss-Seidel fails on FSI (139 capped GMRES solves) and is
not a general choice; multicolor GS and Chebyshev converge for all three but are
slower on the CPU. Setup is a much larger share of GPU time (solid with MueLu
took 28 s on the A40 versus 3.7 s on one M4 core), so the GPU sweep decides.

Unit tests: 364 pass (one 2-rank-only skip); the Schur suite passes on 2 ranks.
New test `TrilinosSchurAMG.CachedProductMatchesTpetraProductAfterValueChanges`
checks the cached product (rectangular B, value changes) against Tpetra on 1
rank and the Tpetra fallback on 2 ranks.

## Run on one Sherlock A40

Rebuild `build2` (Trilinos 17.2, `AMPERE86`) from this commit. The
`HostMirror` → `host_mirror_type` change in `TrilinosFluidAssembly.cpp` is
included, so no manual edit is needed. Then, from the repository root with the
`build2` modules loaded:

```bash
clean() { env $(compgen -e | grep -E '^(SLURM|SBATCH|SVMP)_' | grep -vx SLURM_CONF | sed 's/^/-u /') "$@"; }
clean SVMP_BUILD_DIR=$PWD/build2 SVMP_PIPE_CASE=/scratch/users/$USER/pipe-case \
  SVMP_PYTHON=/scratch/users/sdave03/svv/bin/python \
  sbatch benchmarks/gpu_options/sherlock_a40.sbatch
```

To split it into two concurrent one-GPU jobs:

```bash
clean SVMP_BUILD_DIR=$PWD/build2 SVMP_PARTS=fluid SVMP_PIPE_CASE=/scratch/users/$USER/pipe-case \
  sbatch --time=02:00:00 benchmarks/gpu_options/sherlock_a40.sbatch
clean SVMP_BUILD_DIR=$PWD/build2 SVMP_PARTS=physics \
  sbatch --time=02:30:00 benchmarks/gpu_options/sherlock_a40.sbatch
```

The job runs every line of `benchmarks/gpu_options/fluid_variants.txt` on iliac
(20 steps) and the pipe (10 steps), then every line of `physics_variants.txt` on
solid, FSI and scalar (GMRES + MueLu, 20 steps, 600 s per launch). `A_old` and
`ml_old` reproduce the previous code in the same allocation; `A1_cached_product`
isolates the cached product. Tacho/cuSOLVER variants fail fast if this Trilinos
lacks them and are listed in `failed.txt`; the available Amesos2 solvers are
recorded in `gpu-options-JOBID-config/amesos2.txt`.

Results: `build2/benchmarks/gpu-options-JOBID/` with `report.md`. Copy it (and
the `-config` sibling and `svmp-options-JOBID.log/.err`) back and rerun the
summary with VTK for field checks:

```bash
/path/to/python-with-vtk benchmarks/gpu_options/summarize.py /local/copy/gpu-options-JOBID
```

The same driver runs locally: `benchmarks/gpu_options/run_options.sh local OUT`
(`SVMP_RANKS`, `SVMP_PARTS`, `SVMP_CASES`, `SVMP_*_VARIANTS` select subsets).

## Diagnostics

- `BIPN_STATS` adds `amg_skips` and `schur_product_cached`.
- `SV_TRILINOS_SOLVER_STATS=1` prints `GENERIC_SOLVE_STATS` per generic solve:
  preconditioner setup and Krylov seconds, iterations, MueLu builds/refreshes/skips.
  The physics runner enables it and stores per-run totals in `measurement.json`.
- `SV_TRILINOS_AMG_DETAILS=1` prints the Schur options, Jacobi damping, the
  hierarchy, and whether/why the cached product is used.
- `benchmarks/multiphysics/run_suite.py --env SV_TRILINOS_NAME=VALUE` passes
  solver options to the physics suite (it otherwise clears all `SV_TRILINOS_*`).
