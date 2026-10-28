# Scalar, solid, and FSI benchmark of the current Trilinos implementation

This experiment measures the current solver without changing its C++ implementation.
The CPU run uses one MPI rank and one thread (Kokkos Serial). The supplied Sherlock
job uses one rank on one A40. It is a physics/solver comparison, not a strong-scaling
experiment or a comparison against the old `bipnWithTrilinos` executable.

## What these physics actually use

The native Kokkos element integration kernel currently supports rigid-wall 3D
fluid TET4/HEX8 only. It does **not** integrate scalar, solid, FSI, or ALE mesh
motion elements on the GPU. For the cases below, both assembly choices perform
host element integration:

* `Assembly=trilinos`: persistent Trilinos topology/storage, host element
  contributions accumulated through the Trilinos assembly interface and exported
  to the owned matrix.
* `Assembly=fsils`: FSILS host assembly followed by conversion into Trilinos
  objects for the same Trilinos linear solve.

`SV_TRILINOS_ASSEMBLY=auto` is used throughout. Setting it to `kokkos` does not add
missing physics kernels. Comparing `auto` against `legacy` here would not test a
new element kernel. The meaningful comparison is the XML assembly backend above.
CUDA builds execute supported Trilinos operations on the device, but these runs
must not be described as full GPU assembly benchmarks.

Schur AMG, its RP reuse, and the explicit pressure Schur operator are specific to
NS+BIPN. They are not used by the generic GMRES/CG solvers in this experiment.
`trilinos-ml` below means the existing general MueLu implementation, including its
own setup/smoothing choices; it is not `trilinos-schur-amg`.

## Cases and controlled inputs

| Case | Source under `tests/cases` | Size (output nodes) | Equations |
|---|---|---:|---|
| Scalar | `fluid/precomputed_dye_AD` | 30,211 | Scalar transport using prescribed velocity |
| Solid | `linear-elasticity/beam` | 780 | Linear elasticity |
| FSI | `fsi/pipe_3d` | 2,541 | Displacement-based FSI plus ALE mesh motion |

The scalar case is advection-dominated **advection–diffusion**, with its original
conductivity `1e-8`, not exactly zero-diffusion pure advection. No fluid equation
is solved. The beam is a simple linear solid, not a test of all nonlinear solid
constitutive laws. The FSI test is `struct`, not `ustruct`; `ustruct` currently
requires FSILS assembly and is outside this comparison.

All runs use 20 time steps, original time steps, loads, meshes, material parameters,
and tolerances. Final VTU/restart output is written at step 20. Debug output is
disabled uniformly. Auxiliary inputs are symlinked into isolated run directories;
source cases are never edited. All policies use the same current executable.

The scalar source has a nonlinear cap of only five. Pilots reached that cap without
meeting the specified `1e-6` tolerance; a cap of 30 was also insufficient at later
steps. The benchmark raises the cap to **80 for every scalar policy**. This is an
iteration-budget change, not a tolerance relaxation. The FSI mesh equation uses
an explicit common cap of 1,000 for both CG and GMRES. Missing Krylov dimensions
are set to 50; existing main-equation iteration caps remain unchanged.

## Solver choices

* Scalar: GMRES with diagonal, block-Jacobi, RILUK0, or general MueLu.
* Solid: the same four GMRES policies, plus CG with diagonal or block-Jacobi.
* FSI: the four GMRES policies with CG/diagonal for mesh motion; an additional
  RILUK0 main-equation run changes only mesh motion to GMRES/diagonal.
* Every policy runs with both assembly backends and two sequential repeats.

GMRES and CG are Krylov **solvers**, not preconditioners. CG is not applied to the
nonsymmetric scalar-advection or coupled FSI system. General MueLu's current
Gauss–Seidel setup is not assumed to be a symmetric positive-definite CG
preconditioner, so it is tested with GMRES. Resistance preconditioning is omitted:
these cases have no resistance outlets, and the pressure-space Schur
preconditioner is not an option for these generic solves.

The minimal supported Trilinos baseline is `trilinos-diagonal`. The generic solver
always applies symmetric diagonal scaling and attaches no additional Belos
preconditioner in this mode. Although the lower-level implementation has a
`NO_PRECONDITIONER` branch, XML `Preconditioner=none` is rejected by the current
Trilinos allowed-option set. A genuinely unscaled solve is not exposed here.
Also, the current `trilinos-blockjacobi` generic implementation configures
Ifpack2 `RELAXATION` with one Jacobi sweep; it is not a dense nodal-block inversion.

## Completed CPU results (20 steps, two repeats)

All **60 runs completed** on the Apple M4 Pro (48 GiB), with one rank, one
thread, Kokkos Serial, and the existing RelWithDebInfo (`-O2 -g -DNDEBUG`)
Trilinos executable. There were no missing repeats or timeouts. These are
single-core CPU measurements, not timings using all available CPU cores.

| Physics / policy | Direct Trilinos assembly, median wall s | FSILS assembly + conversion, median wall s | Interpretation |
|---|---:|---:|---|
| Scalar, GMRES/diagonal | 116.664 | 103.970 | Best scalar choice; FSILS assembly is 1.122× faster (10.9% less time) |
| Scalar, GMRES/Jacobi | 120.508 | 104.205 | No useful gain over diagonal |
| Scalar, GMRES/RILUK0 | 142.191 | 127.407 | Fewer Krylov iterations, higher total cost |
| Scalar, GMRES/MueLu | 144.097 | 125.288 | Fewer Krylov iterations, higher total cost |
| Solid, CG/diagonal | 1.074 | 1.101 | Best solid choice; assembly difference is small |
| Solid, CG/Jacobi | 1.104 | 1.158 | Similar to diagonal CG |
| Solid, GMRES/MueLu | 3.694 | 3.735 | Converges; diagonal CG is 3.44× faster in wall time |
| FSI, GMRES/MueLu + mesh CG | 19.817 | 21.736 | Fastest FSI policy meeting every linear tolerance |
| FSI, GMRES/RILUK0 + mesh CG | 26.293 | 27.916 | Converges; MueLu is 1.327× faster with direct assembly |
| FSI, GMRES/RILUK0 + mesh GMRES | 26.833 | 28.524 | Changing mesh CG to GMRES gives no gain |
| FSI, GMRES/diagonal + mesh CG | 12.297 | 13.949 | Nonlinear/field checks pass, but 135 linear-limit failures per run |

For FSI/MueLu, direct assembly gives **1.097×** wall speedup, or **8.8% less
time**, over FSILS assembly plus conversion. For the solid CG case, startup is a
large fraction of the roughly one-second wall time: simulation medians are
0.619 s (direct assembly) and 0.645 s (FSILS assembly). The small assembly
advantage should not be generalized from two repeats on this tiny mesh.

All scalar policies, solid CG and solid GMRES/MueLu, and FSI MueLu/RILUK0
policies meet every equation-level linear and nonlinear check. Their final
fields agree with their respective references within **1.22e-12 relative L2**
across all saved point fields. Every scalar policy uses 611 nonlinear iterations
per run. Diagonal/Jacobi require about 15,900 linear iterations; RILUK0 requires
5,046 and MueLu 4,040. The lower iteration count does not translate into lower
end-to-end time for this transport problem.

Solid GMRES/diagonal and GMRES/Jacobi each have 400 linear failures per run and
also miss nonlinear tolerances (20 and 19 steps respectively). Solid
GMRES/RILUK0 has 40 linear failures per run despite nonlinear convergence.
These policies are not recommended as converged alternatives to CG.

FSI diagonal/Jacobi GMRES have 135 linear failures per run but **do meet the
nonlinear tolerances**, and final fields match the reference to about 5e-14.
Diagonal is therefore the fastest observed configuration with matching final
state for this test; MueLu is the fastest with no linear failures. Those are
different acceptance criteria, and both timings are reported explicitly.

The current fluid-specific Kokkos kernel is not invoked in any of these cases.
The scalar result demonstrates that the direct Trilinos host assembly interface
can be slower than FSILS assembly. The FSI result demonstrates a modest benefit
from the direct interface even before adding an FSI device element kernel.
These small cases do not establish performance on larger production meshes.

Raw runs and regenerated summaries:
`build/benchmarks/multiphysics-cpu-converged/`. A compact, versionable record is
[`benchmarks/multiphysics/cpu_results.json`](benchmarks/multiphysics/cpu_results.json).
Each raw run retains its generated XML, stdout, history, final VTU, and measurement.
The abandoned two-step pilots and insufficient-cap scalar experiment are outside
this directory and are excluded from all reported timings.

### Full CPU table

Linear/nonlinear failure counts below are totals across both repeats. Times are
medians. `Validated` requires completion, every linear/nonlinear tolerance, all
expected repeats, and matching fields; FSI diagonal's matching state is separately
recorded by `state_checks_passed` in the JSON.

| Case | Solver/preconditioner | Assembly | Wall s | Simulation s | Linear failures | Nonlinear failures | Validated |
|---|---|---|---:|---:|---:|---:|---|
| fsi | gmres-blockjacobi | fsils | 14.150 | 13.630 | 270 | 0 | False |
| fsi | gmres-blockjacobi | trilinos | 12.464 | 11.925 | 270 | 0 | False |
| fsi | gmres-diagonal | fsils | 13.949 | 13.440 | 270 | 0 | False |
| fsi | gmres-diagonal | trilinos | 12.297 | 11.805 | 270 | 0 | False |
| fsi | gmres-ml | fsils | 21.736 | 21.235 | 0 | 0 | True |
| fsi | gmres-ml | trilinos | 19.817 | 19.295 | 0 | 0 | True |
| fsi | gmres-riluk0 | fsils | 27.916 | 27.415 | 0 | 0 | True |
| fsi | gmres-riluk0 | trilinos | 26.293 | 25.805 | 0 | 0 | True |
| fsi | gmres-riluk0-mesh-gmres | fsils | 28.524 | 28.005 | 0 | 0 | True |
| fsi | gmres-riluk0-mesh-gmres | trilinos | 26.833 | 26.315 | 0 | 0 | True |
| scalar | gmres-blockjacobi | fsils | 104.205 | 103.050 | 0 | 0 | True |
| scalar | gmres-blockjacobi | trilinos | 120.508 | 119.350 | 0 | 0 | True |
| scalar | gmres-diagonal | fsils | 103.970 | 102.800 | 0 | 0 | True |
| scalar | gmres-diagonal | trilinos | 116.664 | 115.550 | 0 | 0 | True |
| scalar | gmres-ml | fsils | 125.288 | 124.100 | 0 | 0 | True |
| scalar | gmres-ml | trilinos | 144.097 | 142.950 | 0 | 0 | True |
| scalar | gmres-riluk0 | fsils | 127.407 | 126.250 | 0 | 0 | True |
| scalar | gmres-riluk0 | trilinos | 142.191 | 140.950 | 0 | 0 | True |
| solid | cg-blockjacobi | fsils | 1.158 | 0.675 | 0 | 0 | True |
| solid | cg-blockjacobi | trilinos | 1.104 | 0.637 | 0 | 0 | True |
| solid | cg-diagonal | fsils | 1.101 | 0.645 | 0 | 0 | True |
| solid | cg-diagonal | trilinos | 1.074 | 0.619 | 0 | 0 | True |
| solid | gmres-blockjacobi | fsils | 17.605 | 17.100 | 800 | 38 | False |
| solid | gmres-blockjacobi | trilinos | 17.259 | 16.750 | 800 | 38 | False |
| solid | gmres-diagonal | fsils | 16.934 | 16.445 | 800 | 40 | False |
| solid | gmres-diagonal | trilinos | 16.606 | 16.115 | 800 | 40 | False |
| solid | gmres-ml | fsils | 3.735 | 3.237 | 0 | 0 | True |
| solid | gmres-ml | trilinos | 3.694 | 3.207 | 0 | 0 | True |
| solid | gmres-riluk0 | fsils | 12.296 | 11.820 | 80 | 0 | False |
| solid | gmres-riluk0 | trilinos | 12.305 | 11.810 | 80 | 0 | False |


## Run locally

From the repository root, using the existing Trilinos-enabled CPU executable:

```bash
python3 benchmarks/multiphysics/run_suite.py \
  --exe build/svMultiPhysics-build/bin/svmultiphysics \
  --output build/benchmarks/multiphysics-cpu-new \
  --steps 20 --repeat 2 --timeout 300
python3 benchmarks/multiphysics/summarize.py \
  build/benchmarks/multiphysics-cpu-new
```

Outputs must not already exist. `--cases solid fsi`, `--variants cg-diagonal`, and
`--assemblies trilinos` can narrow a diagnostic run. Full reference field checks
require the reference policy to be included: scalar GMRES/diagonal, solid
CG/diagonal, FSI GMRES/RILUK0, all with Trilinos assembly.

## Submit on Sherlock

Use the same modules/runtime environment as the successfully completed A40 runs.
No solver rebuild is required if the binary is already the current version.
Sync the new benchmark files and submit from the repository root:

```bash
sbatch benchmarks/multiphysics/sherlock_a40.sbatch
```

A short compatibility check, before the full 20-step comparison:

```bash
SVMP_STEPS=2 SVMP_REPEAT=1 sbatch benchmarks/multiphysics/sherlock_a40.sbatch
```

Optional overrides: `SVMP_REPO`, `SVMP_BUILD_DIR`, `SVMP_EXE`, `SVMP_OUTPUT`,
`SVMP_STEPS`, `SVMP_REPEAT`, `SVMP_TIMEOUT` (seconds per solver launch, default 180).
The four-hour allocation bounds the full 60-run sweep including difficult policies.
Timed-out or failed policies are retained and the sweep continues. Logs and final
summary identify incomplete or unconverged results; those are not winning timings.

The job checks physical GPU binding and captures executable hash, source revision,
modules, build caches, and NVIDIA/Slurm configuration in a sibling `-config`
directory. Runtime configuration is logged by every simulation. Timings include
identical diagnostic logging. There is no profiler attached.

Copy both result/config directories and batch stdout/stderr back to the Mac, then:

```bash
python3 benchmarks/multiphysics/summarize.py \
  /path/to/multiphysics-JOBID --expect-cuda
```

Python VTK is needed for field comparisons; NumPy is not required. Without VTK,
fields remain explicitly pending. Compare CPU/GPU **simulation** times as well as
wall times, since launcher/startup overhead differs. Do not attribute device
speedups to GPU element integration for these cases.

## Validation and interpretation

`summary.json` and `summary.md` report wall-time medians/ranges, simulation time,
iteration counts, linear failures, nonlinear residual failures, and all saved
point-field relative L2 differences. The nonlinear audit checks the final residual
of **every equation at every step** against the integrator's acceptance rule.
This matters because the integrator also advances when the nonlinear cap is
reached, and the existing history warning does not reliably flag reaching the cap.
Exit code zero and a final VTU therefore do not establish convergence.

Field thresholds are `1e-4` for scalar and solid, `1e-7` for FSI. They are comparison
thresholds, not relaxed solver tolerances. All output fields are compared, including
stress/strain for the beam and both FSI and mesh displacements. These checks establish
agreement between solver policies, not independent physical accuracy.

Only fully completed, converged, field-checked groups with all expected repeats
are marked validated. Timing ratios against FSILS assembly are reported only when
both groups validate. A fast failed solve is not a speedup. Two repeats and tiny
meshes support screening, not precise claims about small percentage differences.

The validation label is deliberately strict about equation-level linear solves.
For this FSI test, a capped linear solve can still be followed by successful
nonlinear corrections and a matching final solution. Those timings are shown,
with their failure counts, but kept separate from policies meeting every linear
tolerance. This differs from the earlier BIPN comparison's optional allowance
for an inner pressure-CG cap while the outer BIPN solve succeeds.

The history's solver percentage measures Belos iteration time only and omits
preconditioner construction. It is not a total assembly/setup/solve breakdown.
Use end-to-end timings here; exact attribution of generic setup costs would
require additional instrumentation.
