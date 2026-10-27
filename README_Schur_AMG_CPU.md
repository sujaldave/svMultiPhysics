# CPU validation of Schur AMG for Trilinos BIPN

This change adds an opt-in pressure preconditioner, `trilinos-schur-amg`, to
`gpuAssemblyTest`. Its first purpose is to measure the benefit of Schur AMG on
the CPU against `bipnWithTrilinos` at `a06b122a7166e9e76912f6a441cc12d8c32a2400`.
GPU assembly is a later change.

On this Apple M4 Pro, the clean ten-process, 20-step artery comparison improved
median wall time from **53.33 s to 46.92 s (1.14× speedup, 12.0% less time)**.
Both branches took 184 Newton iterations. Final velocity and pressure agree
within `3.5e-14` and `2.1e-15` relative L2, respectively. The benefit is real but
modest on this CPU; it is not the order-of-magnitude gain of the full GPU experiment.

## Mathematical change

After the existing FSILS-compatible diagonal scaling, BIPN solves the pressure
system with the existing matrix-free operator

```
S p = L p + B^T Q_R B p.
```

`B` maps pressure to velocity. `Q_R` is the existing outlet-resistance transform,
including the existing treatment of caps. The new preconditioner builds

```
S0 = L + B^T B
M^{-1} approximately equals S0^{-1} (one MueLu AMG V-cycle).
```

Tpetra's distributed transpose sparse matrix multiplication includes remote
contributions to `B^T B`. No local-only approximation is substituted. Resistance
corrections remain in the original solve operator; the AMG setup matrix omits
their potentially dense low-rank updates. This is a deliberate approximation,
not an omission from the equations being solved.

Previously, selecting `trilinos-ml` for pressure constructed MueLu on `L` alone.
That option retains its original behavior. The default `trilinos-diagonal`
retains the existing full-system scaling without an additional inner pressure
preconditioner. Momentum preconditioning and the outer reduced-iteration
algorithm are unchanged.

The new V-cycle uses unsmoothed aggregation, symmetric Jacobi pre/post smoothing
(two sweeps), a sparse KLU coarse solve, up to ten levels, and a coarse threshold
of 200 unknowns. Aggregates have requested sizes 3–12. The Jacobi damping is
`1.8 / bound`, where `bound` is the global maximum absolute row sum of
`D^{-1} S0`, clamped below by one. Unlike a fixed damping value, this accounts
for the current matrix scale. The same conservative value is supplied to the
hierarchy smoothers. Full symmetry/positive definiteness still depends on the
underlying problem, as with the existing pressure CG.

Every Jacobian gets a fresh FP64 Schur matrix and numerical hierarchy, built
once and shared by its inner pressure solves. There is no stale coarse-value
reuse, precision reduction, or relaxed solve tolerance. Zero/nonpositive or
nonfinite Schur diagonals are rejected rather than repaired silently. This
first implementation does not cache the Schur product graph or AMG aggregates.
Those setup optimizations can be measured separately after this comparison.

## Build

From this repository root, use the existing build procedure:

```sh
mkdir -p build
cd build
cmake -DSV_USE_TRILINOS:BOOL=ON \
  -DCMAKE_PREFIX_PATH=/Users/sujaldave/Desktop/CBCLStanford/solvers/lib/trilinos/lib/cmake ..
make -j4
cd ..
```

The executable is `build/svMultiPhysics-build/bin/svmultiphysics`.
The default build type here is `RelWithDebInfo` (`-O2 -g -DNDEBUG`).
Use the same build type for both branches.

For unit tests, add `-DENABLE_UNIT_TEST=ON` to the top-level configure command.
Run:

```sh
build/svMultiPhysics-build/bin/run_all_unit_tests --gtest_filter='Trilinos*'
mpirun -np 2 build/svMultiPhysics-build/bin/run_all_unit_tests \
  --gtest_filter='TrilinosSchurAMG.DistributedProductIncludesOffRankContributions'
```

The installed Trilinos is 16.2.0 with Kokkos SERIAL. An MPI-enabled MueLu 16.2
build assumes an MPI communicator in its hierarchy description code, so the
AMG unit test uses `MPI_COMM_SELF`, rather than `Teuchos::SerialComm`.

## Enable Schur AMG

Inside an existing `<LS type="NS">` section:

```xml
<Linear_algebra type="trilinos">
  <Assembly>trilinos</Assembly>
  <Preconditioner>trilinos-diagonal</Preconditioner>
  <GMRES_Preconditioner>trilinos-diagonal</GMRES_Preconditioner>
  <CG_Preconditioner>trilinos-schur-amg</CG_Preconditioner>
</Linear_algebra>
```

For the baseline, change only `CG_Preconditioner` to `trilinos-diagonal`.
`trilinos-schur-amg` is restricted to pressure CG in NS + Trilinos; it is not a
monolithic or momentum preconditioner.

Set `SV_TRILINOS_BIPN_STATS=1` to print one `BIPN_STATS` record per nonzero-RHS
Jacobian solve. Times cover total BIPN entry-to-solution, Schur construction,
pressure preconditioner setup, CG, and GMRES; MPI runs report maximum rank
times. Total BIPN time excludes element computation before the solver call.
The records include pressure call counts, iteration counts, convergence flags,
and independently recomputed true-residual failures at the requested CG
criterion. The overall RI norm is also reported. Individual inner CG failures
can coexist with a converged outer solve, especially in the old baseline.

## Reproduce the artery experiment

`benchmarks/schur_amg/run_benchmark.py` creates independent run directories,
symlinks the existing mesh, and generates input from the checked-in artery case.
It does not overwrite source inputs or previous results. The case has 15,334
nodes and 75,819 linear tetrahedra, matching Lasso's small fluid experiment.
It runs 20 steps with the existing physics, resistances, time integration,
nonlinear limits, and solver tolerances. Both branches use native Trilinos
assembly. VTK and restart output are written at the final step only in both
runs to keep intermediate I/O out of the comparison.

Build the sibling `sujaldave_bipnWithTrilinos_branch` with the same commands,
then run these commands sequentially from this repository root:

```sh
python3 benchmarks/schur_amg/run_benchmark.py \
  --exe ../sujaldave_bipnWithTrilinos_branch/build/svMultiPhysics-build/bin/svmultiphysics \
  --preconditioner diagonal --steps 20 --ranks 10 --repeat 2 \
  --output build/benchmarks/baseline-repeat

python3 benchmarks/schur_amg/run_benchmark.py \
  --exe build/svMultiPhysics-build/bin/svmultiphysics \
  --preconditioner schur-amg --steps 20 --ranks 10 --repeat 2 --stats \
  --output build/benchmarks/schur-repeat
```

`--ranks N` uses `mpirun -np N`. Each run contains its exact `solver.xml`,
`stdout.log`, `measurement.json`, and final result. The script limits OpenMP,
OpenBLAS, and Accelerate to one thread per MPI rank. Measurements include
wall-clock time and the application's final cumulative time; solver phase
measurements are available with the new executable and `--stats`.

The unchanged baseline executable does not implement the new phase counters.
For direct iteration/phase comparisons, also run the new executable with
`--preconditioner diagonal --stats`; the numerical solver path is unchanged.
Avoid overlapping builds or simulation runs when collecting performance data.

Compare final fields with a Python installation providing VTK:

```sh
/opt/homebrew/bin/python3 benchmarks/schur_amg/compare_fields.py \
  build/benchmarks/baseline-repeat/run-1/10-procs/result_020.vtu \
  build/benchmarks/schur-repeat/run-1/10-procs/result_020.vtu
```

The comparison matches coordinates and checks velocity and pressure relative
L2 errors against `1e-7` by default. Its JSON includes maximum absolute errors.
This is a regression threshold for this tightly converged artery case, not a
universal accuracy guarantee for other simulations.

## Results on this machine

Test platform: Apple M4 Pro (10 performance + 4 efficiency cores), 48 GB RAM,
macOS 26.6.2 arm64, AppleClang 21, Trilinos 16.2.0 / Kokkos SERIAL,
`RelWithDebInfo`. Ten MPI ranks, one thread per rank. Runs were sequential;
no build overlapped these four reported timing runs. Processes were not pinned
to specific CPU cores. Two repeats quantify local variation, not statistical
confidence across machines.

| Implementation | Wall time, run 1 | Wall time, run 2 | Median |
|---|---:|---:|---:|
| Unchanged `bipnWithTrilinos`, diagonal / diagonal | 52.972 s | 53.689 s | 53.331 s |
| New diagonal / Schur AMG | 46.635 s | 47.209 s | 46.922 s |

All four runs completed 20 steps and 184 Newton iterations. The new runs have
717 pressure calls and 50,260 total CG iterations each (70.1 per call), with
zero CG convergence or independently checked true-residual failures.

In the first new run, the sums of per-Jacobian maximum-rank phase times were:

| Phase | Time |
|---|---:|
| Total BIPN, excluding element assembly | 36.181 s |
| Schur matrix construction | 2.161 s |
| Pressure AMG setup | 0.928 s |
| CG, including AMG application | 25.394 s |
| Momentum GMRES | 2.905 s |

A separate run of the new executable with the old diagonal pressure policy
confirmed exactly identical final velocity and pressure to the unchanged
branch. It recorded 259,246 CG iterations over 869 calls; 858 calls reached
their convergence limit / failed the true-residual check, although all outer
RI solves converged. With Schur AMG this becomes 50,260 iterations over 717
calls and zero failures: **5.16× fewer total pressure iterations**. CG time in
these diagnostic runs fell from 36.959 s to 25.394 s (about 1.46×). Each AMG
iteration costs more, explaining the smaller overall gain. The diagnostic
baseline wall time was 54.540 s; the four primary runs above define the reported
speedup.

The full wall time includes assembly, distribution, mesh input, and final
output. Phase maxima need not occur on the same rank, so their sum is not an
exact critical-path decomposition. AMG setup/application and the unchanged
CPU assembly limit the total speedup despite the iteration reduction.

Both ten-rank field comparisons passed with relative L2 differences of
`3.4907e-14` in velocity and `2.0801e-15` in pressure. An additional one-rank,
20-step comparison passed at `3.1599e-14` and `6.3048e-15`. One-versus-ten-rank
baseline fields also passed (`2.8057e-14` velocity, `4.3960e-15` pressure).
The first one-rank baseline overlapped compilation, so its 271.872 s time is
excluded from the speedup table; the one-rank AMG run took 230.918 s.

Raw measurements live under `build/benchmarks/baseline-10`, `schur-10`,
`baseline`, and `schur-first20`. The compact, versioned measurements are in
[`benchmarks/schur_amg/cpu_results.json`](benchmarks/schur_amg/cpu_results.json).
Detailed logs, XML inputs, and final VTU files remain in the build directory.


Validation completed:

- All **357 unit tests** passed, including the original Trilinos tests and four
  new Schur tests: matrix-free equivalence with gapped pressure GIDs and multiple
  vectors, V-cycle symmetry/positive energy/residual reduction, invalid setup
  rejection, and distributed transpose-product contributions.
- All four new Schur tests also passed under two MPI ranks.
- Ten-rank full artery runs exercise two resistance outlets with the new
  preconditioner; both repeats passed field and true-residual checks.

One extra diagnostic launch (`build/benchmarks/diagonal-profile-10/run-1`)
crashed in ParMETIS `Match_Global` during mesh distribution, before any BIPN
solve. The repeat in `diagonal-profile-10-retry` completed all 20 steps and
matched the unchanged branch exactly. That failed startup is excluded from
timing results; its log is retained. No partitioner code was changed, and the
cause of that isolated startup failure has not been diagnosed.

## Relationship to Lasso's experiment

Lasso's main fluid work is in commit `91fc046`; `12110da` improves large-mesh
smoother damping and aggregation. His GPU experiment also uses an explicit
pressure operator, FP32 inner work, GPU assembly, a device-resident matrix,
and fused CUDA solver kernels. This CPU change isolates Schur AMG, so it should
not be interpreted as reproducing his full 9× GPU speedup on this artery case.
His `Gt` variable stores negative transpose: `L - Gt*G` corresponds to our
`L + B^T B` before resistance correction.

Reference: https://github.com/lassoan/svMultiPhysics/blob/fc37b11/Code/Source/gpu/README.md
