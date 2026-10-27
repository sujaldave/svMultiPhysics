# Native Tpetra/Kokkos fluid assembly on CPUs and GPUs

The native Trilinos backend now computes supported fluid elements with Kokkos
and accumulates directly into Tpetra's local CRS values and RHS vectors.
There is no intermediate FSILS matrix or per-Jacobian conversion of a host
matrix into Tpetra objects on this path. Kokkos selects the execution space
from the installed Trilinos build: Serial/OpenMP on CPU, or CUDA on NVIDIA GPUs.

On this CPU, element assembly was **3.3× faster** than the retained host loop.
For the 20-step iliac artery case with 10 MPI ranks, median total wall time
improved from **46.41 s to 39.83 s (1.17×)** relative to the previous Schur-AMG
commit `52399c5`. These are CPU measurements; CUDA compilation and GPU speedup
must still be validated on the cluster.

## What is supported

The batched path supports rigid-wall, three-dimensional fluid equations with
constant viscosity, VMS stabilization (`nFs=1`), one fluid domain per equation,
and TET4 or Q1 HEX8 elements. Existing resistance outlets and host boundary
assembly are preserved. Brinkman terms use the existing element arithmetic.

Other physics, moving meshes, non-Newtonian viscosity, mixed function spaces,
and RIS/URIS retain the existing assembly path in automatic mode. This change
does not add FSI/ALE device element kernels. The shared host element functions
still support their existing viscosity models and moving-mesh inputs.

Select native Trilinos assembly in the solver XML:

```xml
<Linear_algebra type="trilinos">
  <Assembly>trilinos</Assembly>
  <Preconditioner>trilinos-diagonal</Preconditioner>
  <GMRES_Preconditioner>trilinos-diagonal</GMRES_Preconditioner>
  <CG_Preconditioner>trilinos-schur-amg</CG_Preconditioner>
</Linear_algebra>
```

The assembly acceleration is independent of the pressure preconditioner choice.
Schur AMG is held fixed in the comparisons here.

`SV_TRILINOS_ASSEMBLY` controls element execution:

| Value | Behavior |
|---|---|
| `auto` (default) | Use Kokkos for supported native fluid assembly; otherwise retain the host loop |
| `kokkos` | Require the supported path when native `construct_fluid` is called; report unsupported fluid configurations |
| `legacy` | Use the existing host element loop for a controlled comparison |

These controls apply when native Trilinos assembly is selected. Selecting
FSILS or PETSc assembly retains that backend's element accumulation.

## Data flow and lifetime

1. **One-time mesh setup:** cache connectivity, equation-local CRS offsets,
   quadrature, and physical geometry. Existing `nn::gnn` / `gn_nxx` routines
   compute the geometry on the CPU once, and Kokkos views hold the cached data
   in the selected memory space. Coordinates, connectivity, quadrature, graph
   identity, and topology generation invalidate the cache when changed.
2. **Each Jacobian:** reset existing Tpetra values and vectors in place. Pack
   acceleration, velocity, and body-force state into a reusable Kokkos buffer.
   GPU builds transfer this state buffer rather than an assembled matrix.
3. **Compute and scatter:** one Kokkos iteration per element integrates the
   residual and tangent, using fixed-size arrays and cached geometry, then adds
   directly into Tpetra CRS values/RHS. Parallel CPU and GPU execution uses
   atomics; Serial avoids unnecessary atomics.
4. **Boundary contributions:** GPU volume assembly enables compact staging of
   subsequent host boundary additions. Only their sparse offsets/values are
   uploaded and scattered; host boundary code does not request a host view of
   the full device matrix. CPU builds add these contributions directly.
5. **Shared rows:** a single-rank run copies local CRS values directly. MPI
   runs reuse a numeric-value export plan, avoiding repeated sparse row packing
   and global-column lookup. The final owned matrix stays in Tpetra and feeds
   the existing scaling, block extraction, and solver code.

The velocity/pressure element arithmetic lives once in `FluidElementKernel.h`.
Both the existing host `fluid_3d_m/c` entrypoints and the Kokkos kernel call it;
there are no separate CUDA copies or generated physics sources to maintain.
The original layout and equations are retained. In particular, the HEX8
continuity pass retains the final momentum-quadrature Hessian, matching the
current host implementation. A change to that discretization would require
separate scientific validation.

Full-system matrix, overlap matrix, RHS, solution vectors, and communication
plans are reused across Jacobians and zeroed before reuse. NS block matrices
and numerical Schur/AMG setup still have their existing per-Jacobian lifetime.
The Schur AMG spectral bound now uses a device reduction instead of reading
its full setup matrix on the host.

### Numeric MPI export

Trilinos 16.2's general sparse-matrix export uses host row handling. The new
plan assigns an ID to each owned CRS entry, imports that ID metadata once,
and uses Tpetra multivectors to export only numerical values with `ADD`.
Device kernels pack/unpack the cached entry order. Tpetra handles GPU-aware
MPI or its configured communication staging; no direct CUDA/MPI dependency
is introduced into the element code.

The current backend uses 32-bit global ordinals. If the flattened global
entry count exceeds `INT_MAX`, this optimization falls back to the general
matrix export. `ASSEMBLY_EXPORT_STATS` identifies `numeric-values`, `local-copy`,
or `matrix-rows` transport, so this fallback is visible when profiling.

Caching trades memory for repeated work. On a 64-bit build, element geometry,
connectivity, RHS offsets, and matrix scatter offsets use about **768 bytes
per tetrahedron** or **7,456 bytes per HEX8** at eight quadrature points,
plus state and shared quadrature storage. Numeric export adds owned and
overlap value buffers and maps. GPU memory capacity should be measured on
large meshes; a coarse-grained benchmark must include peak memory as well as
runtime. Host snapshots of geometry/connectivity also support cache checks.

## CPU build and validation

Use the existing build procedure:

```sh
cmake -S . -B build -DSV_USE_TRILINOS:BOOL=ON \
  -DCMAKE_PREFIX_PATH=/Users/sujaldave/Desktop/CBCLStanford/solvers/lib/trilinos/lib/cmake \
  -DENABLE_UNIT_TEST=ON
cmake --build build -j4
```

An OpenMP-enabled Trilinos installation permits multiple CPU threads through
the same kernel. The installation tested here uses **Kokkos SERIAL**, so the
measured gains do not depend on adding threads within an MPI rank.

```sh
build/svMultiPhysics-build/bin/run_all_unit_tests
mpirun -np 2 build/svMultiPhysics-build/bin/run_all_unit_tests \
  --gtest_filter='TrilinosNumericExport.*:TrilinosFluidAssembly.*:TrilinosSchurAMG.*'
```

The serial suite passes 360 tests and skips the test requiring exactly two
MPI ranks. The two-rank command passes all eight selected tests, including
that shared-row export test. Tests cover tetrahedral and distorted HEX8
matrix/residual equivalence, changed state/material data, changed geometry and
quadrature, invalid geometry, host additions before/after device assembly,
shared-row summation, and reuse of the export plan across Jacobians.
A build with Trilinos disabled also succeeds.

## Reproduce CPU and GPU comparisons

The shared helper `benchmarks/schur_amg/run_benchmark.py` creates isolated
inputs and records wall time, nonlinear iterations, solver statistics, and
assembly statistics. It keeps tolerances fixed and writes final-step output
only in both runs. Choose a new output directory for every experiment.

Compare assembly modes in the same executable:

```sh
python3 benchmarks/schur_amg/run_benchmark.py \
  --exe build/svMultiPhysics-build/bin/svmultiphysics \
  --preconditioner schur-amg --assembly legacy --steps 20 --ranks 10 \
  --repeat 2 --stats --assembly-stats --output build/benchmarks/cpu-host-repeat

python3 benchmarks/schur_amg/run_benchmark.py \
  --exe build/svMultiPhysics-build/bin/svmultiphysics \
  --preconditioner schur-amg --assembly kokkos --steps 20 --ranks 10 \
  --repeat 2 --stats --assembly-stats --output build/benchmarks/cpu-kokkos-repeat
```

`--threads N` sets OpenMP/Kokkos thread counts; it requires a CPU execution
space supporting those threads. The default is one thread per MPI rank, with
BLAS thread counts also limited to one. The legacy toggle retains the new
storage/export improvements, so comparing against a separately built
`52399c5` measures the entire change, while the toggle isolates element work.

For this local comparison, the previous executable is preserved at
`build/assembly-baseline/svmultiphysics`, with its commit recorded next to it.
To rebuild that baseline elsewhere, use a separate checkout/worktree of
`52399c5` with the same compiler, build type, and Trilinos installation.

### Cluster CUDA build

Activate the same MPI/CUDA environment that builds your existing GPU Trilinos
solver. Use the CUDA-enabled Trilinos prefix and its compatible C++ compiler
wrapper; the CPU-only installation on this Mac cannot build CUDA kernels.
For example, with `TRILINOS_CUDA_PREFIX` and `NVCC_WRAPPER` set to cluster paths:

```sh
cmake -S . -B build-gpu -DSV_USE_TRILINOS=ON \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DENABLE_UNIT_TEST=ON \
  -DCMAKE_PREFIX_PATH="${TRILINOS_CUDA_PREFIX}/lib/cmake" \
  -DCMAKE_CXX_COMPILER="${NVCC_WRAPPER}"
cmake --build build-gpu -j4
```

Use one GPU per MPI rank. Inside a suitable Slurm GPU allocation, run the
one-GPU case, then repeat with `--ranks 2`, `4`, and the available GPU count:

```sh
python3 benchmarks/schur_amg/run_benchmark.py \
  --exe build-gpu/svMultiPhysics-build/bin/svmultiphysics \
  --launcher srun --gpus-per-task 1 --ranks 1 \
  --preconditioner schur-amg --assembly kokkos --steps 20 --repeat 2 \
  --stats --assembly-stats --output build-gpu/benchmarks/gpu-kokkos-1
```

The helper uses [Slurm's per-task GPU binding](https://slurm.schedmd.com/srun.html). An MPI launcher with equivalent
per-rank `CUDA_VISIBLE_DEVICES` binding is also suitable. Assembly statistics
must report `execution_space=Cuda`; the environment variable `kokkos` alone
does not select a GPU when the executable was built against CPU Trilinos.

Validate on GPUs before collecting timing runs:

```sh
srun --ntasks=2 --gpus-per-task=1 --gpu-bind=single:1 \
  build-gpu/svMultiPhysics-build/bin/run_all_unit_tests \
  --gtest_filter='TrilinosNumericExport.*:TrilinosFluidAssembly.*:TrilinosSchurAMG.*'
```

Compare (a) CPU legacy versus CPU Kokkos, (b) GPU legacy versus GPU Kokkos,
(c) CPU Kokkos versus GPU Kokkos, and (d) GPU strong scaling. Keep mesh,
tolerances, time steps, output, and preconditioners fixed. Record the GPU model,
Trilinos configuration, rank binding, wall-time spread, phase times, iteration
counts, and peak memory. A speedup is established only after field checks pass.

```sh
python3 benchmarks/schur_amg/compare_fields.py \
  build/benchmarks/cpu-kokkos-repeat/run-1/10-procs/result_020.vtu \
  build-gpu/benchmarks/gpu-kokkos-1/run-1/1-procs/result_020.vtu
```

The comparison requires Python VTK. It matches coordinates and defaults to a
`1e-7` relative L2 threshold for velocity and pressure. This threshold is a
regression criterion for the tightly converged artery case, not a universal
accuracy tolerance for other simulations.

## CPU measurements

Apple M4 Pro, 48 GB RAM, macOS 26.6.2 arm64, AppleClang 21,
Trilinos 16.2.0 / Kokkos SERIAL, `RelWithDebInfo`. Iliac artery: 15,334 nodes,
75,819 tetrahedra, 20 steps, 10 MPI ranks, one thread per rank. Timed simulations
were sequential and did not overlap compilation. Processes were not pinned.

| Implementation | Run 1 | Run 2 | Median |
|---|---:|---:|---:|
| Previous Schur AMG commit `52399c5` | 46.063 s | 46.751 s | 46.407 s |
| Native Kokkos assembly and reusable setup | 39.533 s | 40.136 s | 39.835 s |
| New storage/export with legacy element loop | 44.457 s | — | — |

Overall improvement versus `52399c5`: **1.165×**, or **14.2% less wall time**.
All runs took 184 Newton iterations. Both Kokkos repeats passed final-field
comparison: velocity `2.603e-14`, pressure `1.018e-15` relative L2. The retained
legacy toggle passed the same comparison. All new pressure solves converged
and passed their independent true-residual checks.

Assembly-only phase totals (sum of per-Jacobian maximum-rank times):

| Phase | Legacy loop, new executable | Kokkos, median of two runs |
|---|---:|---:|
| Element assembly, including cache/state work | 7.904 s | 2.387 s |
| Full-system allocation/reset | 0.091 s | 0.088 s |
| Boundary staging plus numeric matrix export | 0.255 s | 0.249 s |

The element-work ratio is **3.31×**. These phase sums are not an exact parallel
critical-path decomposition. Linear solution still dominates total runtime;
accelerating assembly alone does not imply the same factor for an entire run.
Two repeats show local variation but do not establish cross-platform scaling.

The versioned summary is
[`benchmarks/native_assembly/cpu_results.json`](benchmarks/native_assembly/cpu_results.json).
Raw XML, logs, measurements, and VTU files are retained under
`build/benchmarks/assembly-{baseline,kokkos,legacy}-10`.
One-step non-Newtonian Carreau/Casson and FSI regressions also match the prior
executable within floating-point roundoff. The retained FSILS-assembly input
path also matches native assembly in a one-step artery regression. HEX8 is verified by element/global
assembly tests on a distorted cell; the large Beltrami and GPU benchmarks
remain cluster validation tasks.
