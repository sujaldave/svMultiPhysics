# 2D vs. 3D audit of the fluid (RBVMS) and FSI (ALE) formulations

Branch: `fix2d_CFD_FSI`

This document records a line-by-line comparison of the two-dimensional and
three-dimensional code paths for the residual-based variational multiscale
(RBVMS) incompressible Navier–Stokes formulation and for the arbitrary
Lagrangian–Eulerian (ALE) FSI formulation, together with the changes made as a
result. Nothing here is staged or committed.

---

## 1. Scope and method

Every function in the solver that has a `_2d` / `_3d` pair, or an
`if (nsd == 3) … else …` split, on the fluid/FSI path was read side by side and
compared term by term. Concretely:

| Area | Functions compared |
|---|---|
| Fluid RBVMS interior | `fluid_2d_m` ↔ `fluid_3d_m`, `fluid_2d_c` ↔ `fluid_3d_c` |
| Fluid weak Dirichlet | `bw_fluid_2d` ↔ `bw_fluid_3d` |
| Fluid Neumann / backflow | `b_fluid` (both branches) |
| FSI assembly driver | `construct_fsi` (both branches) |
| Solid, displacement-based | `struct_2d` ↔ `struct_3d`, `b_struct_2d` ↔ `b_struct_3d` |
| Solid, velocity–pressure | `ustruct_2d_m` ↔ `ustruct_3d_m`, `ustruct_2d_c` ↔ `ustruct_3d_c`, `b_ustruct_2d` ↔ `b_ustruct_3d` |
| ustruct assembly | `ustruct_do_assem`, `ustruct_r` (both branches) |
| ALE mesh motion | `construct_mesh`, `l_elas_2d` ↔ `l_elas_3d` |
| Geometry / basis | `nn::gnn`, `nn::gnnb`, `nn::gn_nxx`, `utils::cross`, Hessian packing in `nn.cpp`, `fs::alloc_fs` / `get_thood_fs` / `set_thood_fs`, 2D quadrature and element property tables |
| Boundary conditions | `set_bc_rbnl_l` (Robin), coupled-Dirichlet diagonalisation, `eq_assem` follower-pressure dispatch |
| Post-processing | `bpost` (WSS/traction), `tpost` (stress/strain), `div_post`, strain invariants |
| Material models | `compute_pk2cc` (templated on `nsd`), `compute_visc_stress_*`, Voigt ↔ tensor maps |

The audit produced two classes of result: (a) code where 2D genuinely mirrors
3D and needed no change, and (b) five real 2D-only defects plus one 3D
post-processing typo, all fixed below.

---

## 2. Headline conclusion

**The fluid RBVMS kernels are correct in 2D.** `fluid_2d_m` / `fluid_2d_c`
reproduce `fluid_3d_m` / `fluid_3d_c` term for term, with the correct index
reductions everywhere. If your 2D CFD benchmark is off, the cause is not in
these kernels — see §5 for where to look instead.

**The 2D FSI path was genuinely broken**, but on the *solid* side, not the
fluid side:

* the velocity–pressure solid (`ustruct`) was never wired into the FSI
  assembly in 2D — it threw `"USTRUCT2D_M not implemented"` — even though the
  2D kernels have existed all along and are used by the standalone
  `construct_usolid`;
* those 2D kernels themselves contain four defects, three of which are
  out-of-bounds accesses that abort at runtime (array bounds checking is
  compiled in via `Array_check_enabled` / `Vector_check_enabled` /
  `Array3_check_enabled`), and one of which silently writes tangent
  contributions into the wrong matrix block;
* `ustruct_r`'s 2D branch overruns each CSR row by one entry.

Taken together, 2D `ustruct` and 2D `ustruct`-based FSI have never executed
successfully. The displacement-based solid (`struct`) in 2D FSI is sound, so
2D FSI with a `struct` solid plus a `mesh` equation was and remains the working
configuration.

---

## 3. What matches (verified, no change needed)

### 3.1 `fluid_2d_m` vs `fluid_3d_m`

Checked and confirmed identical in structure:

* **Velocity gradient and Hessian gathering.** In 2D, `uxx(i,j,k)` = ∂²u_j/∂x_i∂x_k
  is filled from `Nwxx` rows `{0: ∂²N/∂x², 1: ∂²N/∂y², 2: ∂²N/∂x∂y}`, and the
  missing symmetric entries `uxx(0,0,1)`, `uxx(0,1,1)` are back-filled the same
  way 3D back-fills its nine. `d2u2` is the correct Laplacian.
* **Strain rate and its gradient.** `es_x(1,0,k) = uxx(1,0,k) + uxx(0,1,k)`
  matches the 3D `es_x[1][0][k]`. The shear-rate gradient contraction
  `mu_x(k) = ½(es_x(0,0,k)·es(0,0) + es_x(1,1,k)·es(1,1)) + es_x(1,0,k)·es(1,0)`
  is the correct 2D reduction of the 3D expression (the off-diagonal pair is
  counted once because it is symmetric, exactly as in 3D).
* **Stabilisation.** `kT`, `kU`, `kS`, `tauM`, `tauC = 1/(tauM·tr(Kxi))`, and
  `tauB` all reduce correctly; `ctM = 1`, `ctC = 36` in both.
* **Fine-scale velocity `up`, `ua`, `pa`, `rM`, `rV`** — term-for-term identical.
* **Tangent block indices.** With `dof = nsd+1 = 3`, the momentum blocks land at
  `lK(0)`, `lK(1)`, `lK(3)`, `lK(4)`, pressure columns at `lK(2)`, `lK(5)`;
  `fluid_2d_c` writes continuity at `lK(6)`, `lK(7)`, `lK(8)`. These are the
  correct `row*dof + col` positions and mirror the 3D `dof = 4` layout.

The only intentional 2D/3D asymmetry is that the unfitted-RIS (`uris`) valve
resistance terms exist solely in the 3D kernels. That is a feature restriction,
not a formulation error, and it is inert whenever `com_mod.urisFlag` is false.

### 3.2 Everything else that checked out

* `bw_fluid_2d` (weak Dirichlet, Nitsche-type) matches `bw_fluid_3d`, including
  the backflow term `un = (|u·n| − u·n)/2` and all tangent indices.
* `b_fluid`'s 2D branch (`lK(0)`, `lK(4)`) matches its 3D branch
  (`lK(0)`, `lK(5)`, `lK(10)`).
* `construct_fsi`'s ALE bookkeeping is dimension-correct: the fluid element
  coordinates are pushed to the current configuration with
  `xl(i,j) += dl(nsd+i+1,j)`, and `fluid_2d_m/c` subtract the mesh velocity from
  `yl(3,·)`, `yl(4,·)` — the right rows for `tDof = 2·nsd+1 = 5` in 2D.
* `l_elas_2d` (the ALE mesh-motion operator) matches `l_elas_3d`, including the
  `lDm = λ/µ` tangent factorisation.
* `struct_2d` / `b_struct_2d` match their 3D counterparts, including the `Bm`
  strain–displacement matrix and the geometric-stiffness contraction.
* `b_ustruct_2d`'s follower-pressure tangent indices (`lKd(1)`/`lKd(2)`,
  `lK(1)`/`lK(3)`) are the correct 2D reduction of the 3D block.
* Geometry: `nn::gnn`'s 2D branch builds `ksix` correctly; `utils::cross` on a
  2×1 matrix returns the correct in-plane normal; `nn::gnnb`'s interior-node
  pick (`ptr(lFa.eNoN)`) is dimension-general.
* `nn::gn_nxx`'s 2D system — unknowns `[N_xx, N_yy, N_xy]`, rows for ξξ, ηη, ξη —
  is set up correctly, and the reference Hessians are packed as
  `{dxx, dyy, dxy}`, matching what `gn_nxx` and `fluid_2d_*` expect.
* 2D element tables and quadrature (`TRI3` nG=3 w=1/6, `TRI6` nG=7,
  `QUD4` nG=4 w=1, `QUD9` nG=9) integrate the reference element exactly;
  `set_thood_fs` gives `TRI6→TRI3` and `QUD8/9→QUD4`, mirroring
  `TET10→TET4` and `HEX20/27→HEX8`.
* `compute_pk2cc` is templated on `nsd`, so the isochoric/volumetric split uses
  `J^(-2/nsd)` consistently. See §6 for the modelling caveat this implies.
* Robin BC, coupled-Dirichlet diagonalisation, `thood_val_rc`, and the
  WSS/traction post-processing are all dimension-general or correctly branched.

---

## 4. Defects found and fixed

All edits carry an inline `// [2D fix]` (or `// [3D fix]`) comment naming the
3D counterpart they were checked against.

### 4.1 `ustruct_2d_c` — out-of-bounds body-force write
`Code/Source/solver/ustruct.cpp` ~line 451

```cpp
Vector<double> fb(2);
fb[0] = …f_x;  fb[1] = …f_y;
fb[2] = …f_z;   // writing index 2 into a length-2 Vector
```

With `Vector_check_enabled` compiled in, this throws
`Index 2 is out of bounds` on the first Gauss point. Removed; 2D has no z body
force.

### 4.2 `ustruct_2d_c` — wrong tangent block for `dC/dV_2`
`Code/Source/solver/ustruct.cpp` ~line 614

`dof = nsd+1 = 3`, so the continuity row is row 2 and its columns are
`lK(6)`, `lK(7)`, `lK(8)`. The code wrote `dC/dV_2` into `lK(8)` — the `dC/dP`
slot. Corrected to `lK(7)`. Cross-checked two ways: `ustruct_3d_c` writes
`dC/dV_1..3` to `lK(12)`, `lK(13)`, `lK(14)` (= `3*dof + col`), and
`ustruct_do_assem`'s 2D branch reads exactly `lK(6)` and `lK(7)` for `dC/dV`
and `lK(8)` for `dC/dP`.

### 4.3 `ustruct_2d_c` — `dC/dP` written out of bounds
`Code/Source/solver/ustruct.cpp` ~line 629

Written to `lK(9)`, but `lK` has `dof*dof = 9` rows (valid indices 0–8).
Corrected to `lK(8)`, matching `ustruct_3d_c`'s `lK(15) = 3*4 + 3`.

Also tightened two array shapes in the same function that were harmlessly
oversized: `NqxFi` is now `(2, eNoNq)` rather than `(2, eNoNw)`, and `VxNwx`
is `(2, eNoNw)` rather than `(3, eNoNw)`.

### 4.4 `ustruct_2d_m` — `rC` used the wrong trace
`Code/Source/solver/ustruct.cpp` ~line 1034

```cpp
double rC = beta*pd + VxFi(1,1) + VxFi(2,2);   // VxFi is 2x2
```

This both dropped the `VxFi(0,0)` term and read out of bounds. `rC` is
`β·ṗ + tr(∇v·F⁻¹)`, so in 2D it is `VxFi(0,0) + VxFi(1,1)`. Note that `rC` also
feeds `rCl = -p + tauC·rC`, which enters the momentum residual directly — so
even setting the out-of-bounds abort aside, this would have poisoned the whole
element residual.

### 4.5 `ustruct_2d_m` — `Bm` shear row copied from 3D with 3D indices
`Code/Source/solver/ustruct.cpp` ~line 1065

```cpp
Bm(2,0,a) = Nwx(2,a)*F(0,2) + F(0,0)*Nwx(1,a);
Bm(2,1,a) = Nwx(2,a)*F(1,2) + F(1,0)*Nwx(1,a);
```

`Nwx` is `2 × eNoN` and `F` is `2 × 2` here, so `Nwx(2,·)` and `F(·,2)` are both
out of bounds. The correct 2D shear row — confirmed against both `struct_2d`'s
`Bm(2,i,a)` and `ustruct_3d_m`'s `Bm(3,i,a)` — is

```cpp
Bm(2,i,a) = Nwx(0,a)*F(i,1) + F(i,0)*Nwx(1,a);
```

### 4.6 `ustruct_2d_m` — `dM_2/dP` written into the continuity row
`Code/Source/solver/ustruct.cpp` ~line 1174

`dM_2/dP` belongs at `1*dof + 2 = 5`; the code wrote `lK(6)`, which is
`dC/dV_1`. Corrected to `lK(5)`. `ustruct_3d_m` writes `lK(3)`, `lK(7)`,
`lK(11)` (= `row*4 + 3`), confirming the pattern.

### 4.7 `ustruct_r` — 2D CSR loop overruns each row
`Code/Source/solver/ustruct.cpp` ~line 1851

```cpp
for (int i = rowPtr(a); i <= rowPtr(a+1); i++)   // one too far
```

A CSR row spans `[rowPtr(a), rowPtr(a+1)-1]`. As written, each row absorbed the
first entry of the next row into its `K·ΔU` product, and the last row read past
the end of `colPtr`/`Kd`. The `nsd == 3` branch immediately above — and every
other CSR sweep in the code base (`set_bc.cpp:2094`, `set_bc.cpp:2119`,
`fs.cpp:455`, `ustruct.cpp:1805`) — uses `rowPtr(a+1)-1`. Corrected.

### 4.8 `construct_fsi` — 2D `ustruct` never wired in
`Code/Source/solver/fsi.cpp` ~lines 263 and 342

Both Gauss loops threw `"USTRUCT2D_M not implemented"` /
`"USTRUCT2D_C not implemented"` for `Equation_ustruct` in 2D. The kernels exist
with signatures identical to their 3D counterparts (modulo dimension) and are
already called by `construct_usolid`, so the two branches now mirror the
`nsd == 3` code exactly. `ustruct.h` was already included.

### 4.9 `post.cpp` — 2D linear-elastic shear strain typo
`Code/Source/solver/post.cpp` ~line 1885

```cpp
ed(2) = ed(2) + Nx(1,a)*dl(i,a) + Nx(1,a)*dl(j,a);
//                                  ^^^ should be Nx(0,a)
```

Engineering shear strain is `∂u_x/∂y + ∂u_y/∂x`. Both `l_elas_2d`'s `ed(2)` and
the 3D branch's `ed(3)` have it right; only the post-processing copy was wrong.
Affects reported strain, Cauchy stress and von Mises for 2D `lElas` — output
only, not the solution.

### 4.10 `post.cpp` — 3D strain-invariant typo (found in passing)
`Code/Source/solver/post.cpp` ~line 1067

```cpp
ksix(2,2) = ksix(2,2) + Nx(1,a)*yl(1,a);   // that is e_yy, not e_zz
```

Corrected to `Nx(2,a)*yl(2,a)`. This one is a **3D** bug, outside the stated
scope; flagged explicitly here so you can decide separately whether to include
it in the same commit. It affects the `Strain_invariants` output only.

---

## 5. If your 2D CFD results are still wrong

Since the 2D RBVMS kernels are clean, a discrepancy in a 2D CFD benchmark is
most likely to come from one of the following. These are hypotheses, not
findings — I did not reproduce your benchmark.

1. **Plane-strain / unit-depth convention.** svMultiPhysics treats a 2D mesh as
   a unit-depth slab. Forces integrated over a boundary therefore come out per
   unit depth. Benchmarks such as DFG 2D-2 (cylinder drag/lift) quote
   coefficients normalised with a specific reference length and mean velocity;
   the raw integral is *not* directly comparable.
2. **Stabilisation constants.** `ctC = 36` is hard-coded in both 2D and 3D. It
   is a 3D-tuned value (`ctC = 36` corresponds to `C_I` for linear tets); the
   analogous constant for linear triangles is not the same. This does not make
   2D "wrong" relative to 3D — the code is self-consistent — but it will shift
   the τ_C used in a 2D benchmark relative to a reference implementation that
   tunes per element type. Worth a sensitivity study before concluding the
   formulation is at fault.
3. **P1/P1 vs Taylor–Hood.** With `TRI3` (`nFs == 1`) the solver runs stabilised
   equal-order; with `TRI6` it runs P2/P1. For a benchmark with a sharp pressure
   feature the two will differ noticeably at coarse resolution.
4. **Hessian term on linear triangles.** The `∇²u` contribution to the
   fine-scale residual is identically zero on `TRI3` (and on `TET4`), so the
   viscous part of the fine-scale residual vanishes. Again self-consistent
   between 2D and 3D, but it changes what "the same formulation" means across
   element orders.
5. **Boundary-layer resolution.** 2D benchmarks are usually run at much higher
   effective resolution than typical 3D vascular meshes; a mesh that "looks
   fine" in 3D practice can be far too coarse for a 2D benchmark tolerance.

A concrete next step would be a manufactured-solution convergence study on a
2D Taylor–Green or Kovasznay flow: that isolates the discretisation from the
benchmark's normalisation conventions and would settle items 1–3 quickly.

---

## 6. Known 2D limitations (not bugs, worth recording)

* **No 2D remesher.** `remesh.cpp:1593` raises
  `"Remesher not yet developed for 2D objects."` Large-deformation 2D FSI
  (Turek–Hron FSI2/FSI3, for instance) cannot remesh and will be limited by
  mesh distortion.
* **`lElas` is not available inside FSI** in either 2D or 3D
  (`construct_fsi` throws for `Equation_lElas` in both branches). Symmetric, so
  not a 2D-specific gap, but it removes one obvious 2D FSI solid option.
* **Isochoric split convention.** `compute_pk2cc` uses `J^(-2/nsd)`, so in 2D the
  volumetric/isochoric split is a genuinely two-dimensional split, *not* the
  plane-strain restriction of the 3D `J^(-2/3)` split. A Neo-Hookean or
  Mooney–Rivlin solid in 2D therefore will not match the plane-strain section of
  the same 3D material. This is a deliberate modelling choice, but it must be
  accounted for when comparing a 2D FSI benchmark against a 3D or analytical
  reference.
* **`construct_mesh` omits the Jacobian** from the Gauss weight
  (`double w = lM.w(g);`, not `lM.w(g)*Jac`), unlike `construct_l_elas`. This
  affects 2D and 3D identically, so it is out of scope for this audit, but it
  scales the ALE mesh-motion operator element-by-element and is worth a separate
  look.
* **Test coverage.** There is exactly one 2D case in the whole suite
  (`tests/cases/fluid/driven_cavity_2d`, plus its porous variant) and **no 2D FSI
  case at all**. Every `fsi`, `fsi_ustruct`, `struct` and `ustruct` case is 3D.
  That is why the defects in §4 survived: none of the affected code is exercised
  by CI. Adding a small 2D FSI regression case is the single highest-value
  follow-up.

---

## 7. Verification performed

* Full rebuild of `svmultiphysics` in `build-noTril` — clean, no new warnings.
* `tests/cases/fluid/driven_cavity_2d` re-run and compared against the committed
  reference `1-procs/result_002.vtu`:

  | field | max abs. difference |
  |---|---|
  | Velocity | 1.1e-13 |
  | Pressure | 9.6e-10 |
  | WSS | 8.7e-11 |
  | Traction | 8.6e-10 |
  | Vorticity | 8.6e-12 |
  | Divergence | 2.8e-12 |

  i.e. bitwise-equivalent to round-off — the fluid path is untouched, as intended.

* The 2D `ustruct` and 2D `ustruct`-FSI paths are **not** covered by any test and
  could not be exercised here. The fixes in §4.1–§4.8 are justified by direct
  correspondence with the 3D kernels and with `ustruct_do_assem`'s own index
  expectations, but they have not been validated against a running 2D
  simulation. Before trusting them, please build a small 2D `ustruct` case
  (block compression is the natural analogue of the existing 3D one) and check
  the Newton convergence rate — a mis-indexed tangent shows up immediately as
  loss of quadratic convergence even when the residual is right.

---

## 8. Files changed

```
Code/Source/solver/fsi.cpp     | 28 +++++++++++++++++++--------
Code/Source/solver/post.cpp    |  9 +++++++--
Code/Source/solver/ustruct.cpp | 44 ++++++++++++++++++++++++++++++++----------
```

Nothing is staged or committed.
