# Trilinos Navier-Stokes Bi-Partition Solver

## Purpose

The Trilinos Navier-Stokes bi-partition solver mirrors the legacy FSILS RI
algorithm using Tpetra sparse operators and Belos inner solvers. It is selected
only when an equation uses both `LS type="NS"` and Trilinos linear algebra.
FSILS and non-NS Trilinos solve paths are unchanged.

## XML Configuration

```xml
<LS type="NS">
  <Linear_algebra type="trilinos">
    <GMRES_Preconditioner> trilinos-resistance </GMRES_Preconditioner>
    <CG_Preconditioner> trilinos-diagonal </CG_Preconditioner>
  </Linear_algebra>
  <Max_iterations> 15 </Max_iterations>
  <NS_GM_max_iterations> 10 </NS_GM_max_iterations>
  <NS_CG_max_iterations> 300 </NS_CG_max_iterations>
  <Tolerance> 0.2 </Tolerance>
  <NS_GM_tolerance> 1e-3 </NS_GM_tolerance>
  <NS_CG_tolerance> 0.02 </NS_CG_tolerance>
  <Krylov_space_dimension> 300 </Krylov_space_dimension>
</LS>
```

`GMRES_Preconditioner` controls both momentum GMRES solves.
`CG_Preconditioner` controls the pressure Schur CG solve. Missing tags default
to `trilinos-diagonal`.

Both fields use the existing Trilinos preconditioner registry. The momentum
field accepts `trilinos-diagonal`, `trilinos-blockjacobi`, `trilinos-ilu`,
`trilinos-ilut`, `trilinos-riluk0`, `trilinos-riluk1`, `trilinos-ml`, and
`trilinos-resistance`. The pressure field accepts the algebraic options but
rejects `trilinos-resistance`, which is a velocity-space operator.

## Components

- `trilinos_backend::TopologyCache` owns equation-local maps, static graphs,
  communication plans, and local CRS assembly offsets.
- `trilinos_backend::LocalAssemblyBuffer` accumulates native element tensors
  in overlapping local CRS storage and flushes them once per Jacobian.
- `TrilinosBipartitionNSSolver` owns one Jacobian solve and the RI loop.
- `TrilinosNSBlockTopologyCache` owns equation-local velocity/pressure maps,
  static `A/B/C/L` graphs, value-copy offsets, and block communication plans.
- `TrilinosNSBlockSystem` owns velocity/pressure maps and `A`, `B`, `C`, `L`.
- `MomentumOperator` applies `A` plus coupled-outlet Jacobian terms.
- `TrilinosResistanceOperator` applies the outlet transform `Q_R`.
- `PressureSchurOperator` applies the matrix-free operator
  `L + B^T Q_R B`.
- `TrilinosPreconditionerFactory` constructs reusable Ifpack2, MueLu, or
  resistance handles and attaches them to Belos problems.
- `MueLuReuseCache` retains one role-specific AMG hierarchy across Newton
  iterations in the same time step.

All solver-facing types except the shared resistance operator live in the
`trilinos_bipartition` namespace. The resistance operator is also used by the
existing monolithic Trilinos solver.

## Matrix And Vector Flow

The full scalar Jacobian retains the original global ID
`node*dof + component`. Block extraction creates:

- A velocity map for components `0 <= component < nsd`.
- A pressure map for component `nsd`.
- `A`: velocity row and velocity column.
- `B`: velocity row and pressure column.
- `C`: pressure row and velocity column.
- `L`: pressure row and pressure column.

The full residual is imported into velocity and pressure subvectors. The final
corrections are exported back into the disjoint full-system entries, right
scaled by the existing Jacobi factor, imported to ghost nodes, and copied into
the legacy node/DOF result ordering.

Native Tpetra assembly and the existing FSILS assembly fallback both feed this
same block solver. The fallback preserves the established nodal CSR traversal
and lifts its assembled values and residual into the full Tpetra system before
block extraction.

For native assembly, element matrices are produced in host memory. The
equation-local assembly buffer uses cached scalar offsets to add each dense
element block directly to a host view of an overlapping `Tpetra::CrsMatrix`.
It does not allocate column/value arrays or call global-index insertion inside
the element loop. At solve entry, `getLocalMatrixDevice()` performs one bulk
host-to-device synchronization, and a cached `Tpetra::Export` sums overlapping
ghost rows into the uniquely owned matrix. The RHS follows the same overlap
layout and is exported separately.

The overlap is required for finite element assembly: an element on one rank
can contribute to a node owned by another rank. Directly writing only the
owned matrix would lose that contribution. The existing FSILS assembly
fallback intentionally retains its global-index insertion path.

The solver's `rowPtr` connectivity uses unsorted application-local node
indices, while Tpetra local rows follow map order. Topology construction maps
each row GID back to its application-local index before assigning row capacity
or caching scalar offsets. This distinction is essential when MPI partitions
contain ghost nodes and the two orders differ.

## RI Solver Flow

Each RI iteration performs:

1. Momentum predictor: solve `A_m U = R_m` with Belos Block GMRES, where
   `A_m` includes coupled-boundary Jacobian contributions.
2. Pressure right-hand side: form `R_c - C U`.
3. Pressure correction: solve `(L + B^T Q_R B) P = R_c - C U` with Belos
   Pseudoblock CG.
4. Momentum correction: solve `A_m U = R_m - B P` with the same GMRES setup.
5. Apply the block images, assemble the RI Gram system with distributed
   Tpetra dot products, and solve that small dense system on the host.
6. Update the residual basis, convergence state, and final correction.

One momentum and one pressure preconditioner handle are prepared after
extracting the blocks. The same two handles are attached to fresh Belos linear
problems throughout all RI iterations for that Jacobian. If a block uses
MueLu, the handle references an equation-local hierarchy cache described
below; Ifpack2 and resistance handles remain scoped to one Jacobian.

## MueLu Reuse

Momentum and pressure use separate `MueLuReuseCache` objects because their
maps, sparsity patterns, smoothers, and solver roles differ. Reuse follows a
fixed time-step policy:

1. The first Newton solve at time step `cTS` builds a complete hierarchy.
2. Later Newton solves at the same `cTS` call
   `MueLu::ReuseTpetraPreconditioner()` with `reuse: type = RAP`.
3. The first solve after `cTS` changes discards the retained hierarchy and
   performs another complete build.
4. A topology-generation, map, dimension, or nonzero-count change also forces
   a complete build.

RAP reuse retains the expensive multigrid transfer structure and refreshes
the hierarchy for the new Jacobian values. It is more conservative than full
reuse, which could keep stale coarse operators, and less expensive than
repeating aggregation and hierarchy construction at every Newton iteration.

This policy requires no XML option. It applies only when either BIPN inner
preconditioner is `trilinos-ml`; all other policies retain their existing
lifetime. Build and refresh counts are stored independently for the two block
caches and are available for unit-level verification.

## Resistance Semantics

For every active resistance or RCR outlet, the pressure Schur action includes
`Q_R` regardless of the selected momentum preconditioner. Selecting
`trilinos-resistance` for GMRES additionally attaches that velocity-space
operator as the momentum left preconditioner. Other momentum policies attach
only their selected Ifpack2 or MueLu operator.

Cap vectors participate in the outlet scalar projection but not in its update
vector, matching the FSILS convention.

## GMRES Budgeting

FSILS interprets `NS_GM_max_iterations` as a count of complete restarted-GMRES
cycles. Belos uses a total iteration limit, so the solver translates:

```text
Maximum Iterations = NS_GM_max_iterations * Krylov_space_dimension
Num Blocks         = Krylov_space_dimension
Maximum Restarts   = NS_GM_max_iterations - 1
```

Thus `10` cycles with a Krylov dimension of `300` permit at most `3000` Belos
iterations and `9` restarts. Invalid and overflowing budgets are rejected.
`NS_CG_max_iterations` remains the total pressure-CG iteration limit.

The FSILS absolute and relative stopping rules are preserved by passing Belos
an effective relative tolerance equal to
`max(relative_tolerance, absolute_tolerance / ||b||_2)` for a nonzero RHS.

## Portability And Ownership

Sparse products, block operator composition, vector updates, norms, and dot
products use Tpetra and its configured Kokkos execution space. Only the small
RI coefficient solve and final application-facing copy execute explicitly on
the host.

Solver, block, and preconditioner state is owned by the equation-level
Trilinos object. The implementation adds no mutable file-scope state. The
FSILS assembly fallback reads the same equation-local topology metadata as the
native Trilinos path and does not change the block-system interface.

Kokkos itself is process-wide and may be shared by multiple equation-level
backends in FSI and mesh-motion simulations. Finalization first releases each
backend's Tpetra state, then uses a synchronized process-local user count so
only the last Trilinos equation shuts down Kokkos. No Tpetra object is allowed
to survive that shutdown.

### Static Topology Reuse

Each equation-level Trilinos object owns a `trilinos_backend::TopologyCache`.
The cache retains the owned and ghost maps, solution importer, overlap-to-owned
exporter, owned and overlapping fill-complete `Tpetra::CrsGraph` objects, and
derived assembly offsets across Newton iterations and time steps. `alloc()`
creates fresh matrices and vectors from those static graphs and maps, so no
Jacobian values or residual data survive between solves.

The cache signature includes communicator size and rank, global/local/ghost
node counts, DOF count, index base, local-to-global maps, CSR row pointers, and
CSR column indices. Any change rebuilds the maps, importer, derived metadata,
and graph together. An exact match reuses the same Tpetra objects. This keeps
cache ownership equation-local and prevents FSI, mesh, or other equations from
sharing incompatible distributions.

The scalar topology cache and `TrilinosNSBlockTopologyCache` have separate
responsibilities. The scalar cache describes the complete equation, while the
block cache derives and retains the velocity/pressure split. On its first use
for a scalar topology generation, the block cache:

1. Creates the velocity and pressure maps using the original scalar global
   IDs.
2. Builds and fill-completes the static `A`, `B`, `C`, and `L` graphs.
3. Creates full-to-block importers and block-to-full exporters.
4. Computes local CRS offset pairs from each full-matrix entry to its block
   entry and stores those pairs in Kokkos device views.

For every Jacobian, the solver constructs fresh matrices from the four cached
graphs and copies only values from the full matrix into their local device CRS
arrays. It does not rebuild submaps, globally insert block entries, or repeat
graph `fillComplete()`. Boundary vectors and residuals use the cached import
plans, and the final correction uses cached export plans. A changed scalar
topology generation or incompatible `nsd`/`dof` split rebuilds the complete
block cache.

This lifetime deliberately differs from MueLu reuse. Block graph topology may
persist across time steps, while numerical block values are refreshed for
every Jacobian. MueLu then applies its per-time-step RAP policy to those fresh
matrices after checking topology, map, and matrix-shape signatures.
