# Native assembly with classical BIPN preconditioners

This comparison keeps the native Kokkos fluid assembly, reusable Tpetra matrix
storage and export plans, bulk boundary setup, batched outlet operations, and
FP64 physics. It changes only the momentum/pressure preconditioner selection
and uses the corresponding pressure operator. No solver defaults are changed.

## Why test this

The latest A40 scaling test showed assembly falling from 1.48 seconds on one
GPU to 0.37 seconds on four GPUs, while pressure CG grew from approximately
19.42 to 432.49 seconds. The number of pressure iterations stayed near 52,000.
Thus the assembly improvements can be retained independently while testing a
different pressure solver/preconditioner path.

Classical diagonal pressure CG avoids AMG construction and all AMG V-cycle
applications, including its distributed coarse solve. It still performs the
existing matrix-free Schur application L+B^T Q_R B and associated communication.
Resistance preconditioning for GMRES targets the outlet contribution in the
momentum solve. It does not itself precondition the pressure solve or remove
all MPI communication. Cheaper iterations can help, but many more iterations
can erase the saving; final-field validation remains necessary.

All variants retain the existing symmetric diagonal scaling and resistance
terms in the actual equations. "Diagonal CG" means the classical pressure
policy after that scaling, not removal of resistance physics.

| Directory | Momentum GMRES preconditioner | Pressure CG preconditioner | Pressure operator |
|---|---|---|---|
| `amg-diagonal` | diagonal | Schur AMG | explicit with exact outlet/cap corrections |
| `amg-resistance` | resistance | Schur AMG | explicit with exact outlet/cap corrections |
| `classic-diagonal` | diagonal | diagonal | factored L+B^T Q_R B |
| `classic-resistance` | resistance | diagonal | factored L+B^T Q_R B |

The AMG/diagonal variant is the existing optimized default and the comparison
reference. Including AMG/resistance tests whether the momentum choice helps
without discarding the pressure preconditioner.

## CPU measurements

All variants use the same existing executable, ten MPI ranks, one thread per
rank, native Kokkos Serial assembly, and the twenty-step iliac artery case.
Physics, nonlinear tolerance, GMRES/CG tolerances and iteration limits, output
frequency, and mesh are unchanged. Each configuration is measured twice in
sequential runs. Results and six-field checks are recorded in
`benchmarks/solver_optimization/classical_cpu_results.json`; local raw cases
are under `build/benchmarks/classic-bipn-cpu/`.

| Pairing | Repeat 1 | Repeat 2 | Median |
|---|---:|---:|---:|
| AMG / diagonal GMRES | 35.454 s | 34.138 s | **34.796 s** |
| AMG / resistance GMRES | 36.236 s | 34.137 s | 35.187 s |
| Diagonal CG / diagonal GMRES | 49.645 s | 48.360 s | 49.003 s |
| Diagonal CG / resistance GMRES | 51.069 s | 48.603 s | 49.836 s |

The proposed classical resistance/diagonal pairing does **not** win on this CPU
case: it takes approximately 43.2% more wall time than the current optimized
default. Resistance GMRES reduces momentum iterations (12,059 to 9,546 with
AMG pressure), but its extra application work offsets that reduction here.
The approximately 1.1% difference between the two AMG medians is small relative
to the repeat spread and is not evidence of a useful improvement.

Each full run completes 184 nonlinear iterations, with all outer RI solves
successful. AMG requires about 51,000 pressure iterations and reports no inner
failures. The classical variants require about 261,000 iterations: 864 of 877
pressure calls for diagonal GMRES and 859 of 874 for resistance GMRES fail the
inner convergence/true-residual check, generally at the 300-iteration cap.
These remain inexact inner solves. All six final fields still agree with the
AMG/diagonal reference: classical resistance/diagonal velocity error is
3.42e-14 relative L2, pressure 1.45e-15; the largest error over all variants and
fields is 3.64e-13. No tolerance or iteration limit was loosened for the test.

CPU measurements do not determine the GPU winner: removing repeated AMG
applications changes GPU launch and communication costs substantially. The
supplied one-GPU test measures that trade-off without changing assembly.

One launch failed in the pre-existing ParMETIS partitioner before reaching
solver setup. Its log is retained at `build/benchmarks/classic-bipn-failed-partition`;
the identical command was retried. The failed attempt is not a timing sample.

## Run the same comparison on one A40

After syncing the new benchmark scripts, submit from the repository root with
the successful GPU build's environment:

```bash
sbatch benchmarks/solver_optimization/sherlock_preconditioners.sbatch
```

The job reserves one A40 and runs the four variants with native GPU assembly,
twenty steps, two repeats, and detailed phase/convergence statistics. A separate
two-step runtime/hierarchy check is excluded from timings. GPU binding, build
configuration, source version, and executable hash are captured. No C++ rebuild
is required for this experiment; the preconditioner policies already exist.

For a shorter preliminary comparison:

```bash
SVMP_STEPS=5 SVMP_REPEAT=2 sbatch benchmarks/solver_optimization/sherlock_preconditioners.sbatch
```

Results go to `build/benchmarks/preconditioners-JOBID/`. The script writes a
`summary.json` with timings, phase totals, iteration counts, and field checks
when VTK is available. After downloading the complete directory, run locally:

```bash
python3 benchmarks/solver_optimization/summarize.py \
  /absolute/path/to/preconditioners-JOBID \
  --reference-group amg-diagonal --expect-cuda \
  --allow-diagonal-inner-nonconvergence
```

Classical diagonal CG often reaches the 300-iteration limit even when the
outer BIPN/nonlinear solves converge. This also occurred in the original
`bipnWithTrilinos` comparison. The final option reports those classical inner
limits and true-residual failures as visible warnings, without suppressing
their counts. Outer failures and field mismatches remain errors; AMG inner
failures also remain errors. Without that option the summary treats every
inner failure as an error. An unconverged inner solve is not relabeled as
converged merely because the simulation completed.

## Select the configuration manually

For classical resistance/diagonal BIPN with native assembly:

```xml
<Linear_algebra type="trilinos">
  <Assembly>trilinos</Assembly>
  <Preconditioner>trilinos-diagonal</Preconditioner>
  <GMRES_Preconditioner>trilinos-resistance</GMRES_Preconditioner>
  <CG_Preconditioner>trilinos-diagonal</CG_Preconditioner>
</Linear_algebra>
```

Use `SV_TRILINOS_ASSEMBLY=kokkos` to require the native path. The Trilinos/Kokkos
build determines whether that executes on CPU or CUDA. The helper now accepts
`--gmres-preconditioner resistance` in addition to the existing pressure
`--preconditioner` selection.

The comparison should first identify the best one-GPU pairing. If the classical
variant wins, test that exact configuration at two/four GPUs next. A faster
one-GPU result does not establish multi-GPU scaling, and switching off AMG does
not guarantee that the four-GPU communication problem disappears. Larger cases
will still be needed to assess useful GPU throughput.
