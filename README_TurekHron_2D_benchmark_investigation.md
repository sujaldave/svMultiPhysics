# Turek–Hron CFD2/CFD3 and FSI2/FSI3 in svMultiPhysics — diagnosis

Branch: `fix2d_CFD_FSI`. Companion to
[`README_2D_CFD_FSI_investigation.md`](README_2D_CFD_FSI_investigation.md), which
covers the 2D-vs-3D source audit. This document covers the benchmark runs in
`~/Documents/ChatGPT/svMultiPhysics Benchmarl/hron-Turek-FSI3/generated/`.

Nothing is staged or committed.

---

## 1. Short answer

Your 2D fluid formulation is fine. CFD2 reproduces the benchmark drag to 0.36%.
The problems are, in order of impact:

**Both physics are verified against their benchmarks.** CFD2 (steady) 0.36% on
drag, CFD3 (unsteady, on a sufficient mesh) 0.9% on mean drag with a proper
vortex street, and CSM3 (structure, refined) exact on frequency and −1.3% on
amplitude. None of your problems were formulation bugs. In order of impact:

1. **The flap is far too stiff** — two independent causes, both measured below:
   linear triangles lock in bending, and svMultiPhysics silently adds a
   volumetric penalty on top of St. Venant–Kirchhoff. Together they cost ~36% of
   the flap's amplitude and raise its natural frequency by 25%. FSI2 is a
   lock-in phenomenon, so a flap with the wrong natural frequency cannot
   reproduce it.
2. **The fluid mesh is too coarse to sustain the wake instability at Re = 200.**
   CFD3 converges to a *steady* solution. The same mesh sheds violently at
   Re = 400, so this is numerical dissipation raising the effective critical
   Reynolds number, not an inability to represent unsteadiness.
3. **The ALE mesh ratchets.** By t = 35 s the tria FSI2 mesh has elements at
   0.9% of their original area; the quad FSI2 run inverted elements at
   t = 10.86 s and that is what killed it.
4. **You are using the wrong linear solver.** `<LS type="GMRES">` on a
   saddle-point system costs 5,000–20,000 Krylov iterations per Newton step.
   `<LS type="NS">` runs the same CFD3 case **127× faster** (15.3 s/step →
   0.12 s/step). It was aborting with "FSILS: Singular matrix detected"; that is
   a solver bug, fixed here.

**Remeshing is not the fix**, and it cannot be ported from 3D as-is. Details in
§7.

---

## 2. What the CFD runs already told us

Your own results, re-read:

| | svMP | reference | error |
|---|---|---|---|
| CFD2 drag | 136.207 | 136.7 | **0.36%** |
| CFD2 lift | 9.605 | 10.53 | 8.8% |
| CFD3 drag mean | 436.775 | 439.45 | 0.61% |
| CFD3 drag amplitude | 0.0009 | 5.6183 | **~100%** |
| CFD3 lift amplitude | 0.14 | 437.81 | **~100%** |

CFD2 is a *steady* benchmark and you nail it. That single number clears the
fluid kernels, the force extraction in `extract_boundary_forces.py`, the
geometry, the inflow table, and the boundary conditions all at once.

CFD3 is the *unsteady* benchmark and the amplitude is zero to five digits. The
force history is a smooth monotone approach to a fixed point:

```
 t= 2.82  drag=435.688  lift= -5.133
 t= 4.22  drag=436.688  lift=-23.669
 t= 6.32  drag=436.762  lift=-29.862
 t= 8.42  drag=436.773  lift=-31.212
 t=10.00  drag=436.775  lift=-31.582
```

So the diagnosis is not "the forces are wrong" — it is **"the vortex street
never starts."** Everything downstream follows from that, because the FSI2/FSI3
flap oscillation is driven by the shedding.

### Verified: your inflow file is correct

`inflow/inlet_FSI2.dat` and `inlet_FSI3.dat` parse cleanly as
`nsd nTime nNodes` + time list + per-node blocks. The ramp matches
`0.5(1 − cos(πt/2))` to 4 decimals, the plateau is exactly `1.5·Ubar`
(1.499108 for Ubar=1, 2.998215 for Ubar=2), `vy ≡ 0`, and the trailing
`t = 41` / `t = 21` entry correctly holds the profile constant past t = 2 for
the whole run. No issue here.

### Verified: your geometry is correct

Channel 2.5 × 0.41, flap y ∈ [0.19, 0.21]. The deliberate 0.005 m asymmetry
(cylinder centre at y = 0.2, channel centreline at y = 0.205) is present — which
is why CFD2 gives a non-zero lift at all.

---

## 3. Why CFD3 does not shed — measured, not guessed

### 3.1 It is not the numerical damping

My first hypothesis was `<Spectral_radius_of_infinite_time_step>0.5</…>`
over-damping the instability. **I tested it and it is wrong.** Rerunning CFD3 at
dt = 0.005 with ρ∞ = 0.5 and ρ∞ = 1.0:

| | drag mean | drag amp | lift mean | lift amp |
|---|---|---|---|---|
| ρ∞ = 0.5 | 435.370 | 0.0028 | −25.778 | 0.374 |
| ρ∞ = 1.0 | 435.380 | 0.0028 | −25.784 | 0.371 |

Identical to four digits. Removing the numerical damping entirely changes
nothing. Raise ρ∞ towards 1.0 for benchmark work anyway, but it is not your bug.

### 3.2 It is the spatial resolution

Same extra-coarse triangle mesh, same solver, ρ∞ = 1.0, but with the inflow
doubled to Ubar = 4 (Re = 400):

```
 t= 1.85  drag= 1602.290  lift=  -665.233
 t= 2.05  drag= 1666.713  lift= -1456.255
 t= 2.45  drag= 1692.245  lift=   733.675
 t= 2.85  drag= 1599.405  lift= -1530.222
 t= 3.05  drag= 1566.263  lift=  2432.149
   second half: lift mean -229.2, amplitude 3309.2
```

The mesh sheds violently at Re = 400 and not at all at Re = 200. So the
discretisation *can* represent an unsteady wake; its effective critical Reynolds
number simply sits somewhere between 200 and 400, whereas the true value for
this geometry is below 200.

Why: the mesh has 5,130 points / 9,792 triangles, with a median element edge of
**0.0116 m near the flap**. The flap is 0.02 m thick, so there are fewer than
two elements across it, and the Re = 200 boundary layer (δ ~ D/√Re ≈ 0.007 m) is
covered by one element. With P1/P1 equal-order elements the RBVMS τ terms then
have to supply a lot of stabilisation, and that stabilisation is what is damping
the instability.

### 3.3 Confirmed: refining the fluid mesh triggers the shedding

`medium_example` (34k quads, median near-body edge **0.0022 m** — 5× finer) run
at the same conditions (Ubar = 2, ρ∞ = 1.0, dt = 0.005) does **not** settle to a
fixed point. Successive changes in the total lift:

```
 t= 3.30  lift= -63.948   dLift=  -2.987
 t= 3.50  lift= -64.792   dLift=  +4.720
 t= 3.60  lift= -78.816   dLift= -14.024
 t= 3.70  lift= -66.216   dLift= +12.600
 t= 3.80  lift= -87.803   dLift= -21.587
 t= 3.90  lift= -67.857   dLift= +19.946
 t= 4.00  lift= -89.757   dLift= -21.900
```

From t ≈ 3.5 s the sign alternates every sample with growing magnitude: an
oscillation has appeared and is amplifying, roughly e-folding every 0.4 s. The
extra-coarse mesh under identical settings approached its fixed point
monotonically to six digits and never did this.

So the conclusion of §3.2 is confirmed from both directions — the coarse mesh
sheds if you raise Re, and the same Re sheds if you refine the mesh.
**`medium_example` is the mesh to run CFD3 on.**

Run to t = 9 s, it settles into a limit cycle that reproduces the benchmark:

| window | lift mean | lift amp | drag mean | drag amp |
|---|---|---|---|---|
| t = [4,5) | −84.45 | 59.87 | 437.770 | 0.598 |
| t = [5,6) | −101.50 | 279.58 | 438.257 | 3.448 |
| t = [6,7) | −10.25 | 468.62 | 436.767 | 5.863 |
| t = [7,8) | 9.30 | 423.76 | 435.756 | 5.297 |
| t = [8,9) | 5.62 | 405.38 | 435.370 | 4.775 |
| **Turek–Hron reference** | **−11.893** | **437.81** | **439.450** | **5.6183** |

Taking t ≥ 7 as settled: **drag mean 435.40 (0.9% error), drag amplitude 5.30
(5.7%), lift amplitude 424 (3.2%)**. On a mesh that was not tailored for this
benchmark, that is a reproduction.

Two honest caveats. First, VTK output every 20 steps gives a 0.1 s sample
interval against the 0.2275 s reference period — 2.3 samples per cycle. The
frequency therefore cannot be read off this series at all, and the amplitudes
are *lower bounds* because the sampling misses the true peaks. Use
`<Increment_in_saving_VTK_files>` of 4 or less (≥ 11 samples/cycle) if you want
to measure the 4.3956 Hz frequency and tight amplitudes. Second, the lift
amplitude drifts down over the last three windows (469 → 424 → 405) around the
437.81 reference; some of that is sampling noise at 2.3 samples/cycle, but a
longer run with denser output would be needed to confirm the limit cycle is
truly stationary.

---

## 4. Why the flap is too stiff — measured with Turek–Hron CSM3

The cleanest way to separate the structure from the flow is the CSM3 benchmark:
the flap alone, clamped, gravity g = 2 m/s² switched on at t = 0, no fluid.
Reference: tip `ux = −14.305e-3 ± 14.305e-3`, `uy = −63.607e-3 ± 65.160e-3`,
f = 1.0995 Hz.

I ran CSM3 with **your own solid meshes** (both have exactly 230 nodes, so this
isolates element type and material settings, not resolution):

| case | ux mean (e-3) | uy mean (e-3) | uy amp (e-3) | freq (Hz) |
|---|---|---|---|---|
| **Turek–Hron reference** | **−14.305** | **−63.607** | **65.160** | **1.0995** |
| QUAD flap, default penalty | −9.242 | −51.837 | 52.354 | 1.2270 |
| QUAD flap, `Penalty_parameter` ≈ 0 | −12.322 | −59.675 | 59.978 | 1.1407 |
| TRIA flap, default penalty | −5.685 | −40.752 | 41.913 | 1.3746 |

Two separate, additive problems.

### 4.1 svMultiPhysics adds a volumetric penalty on top of StVK

`read_files.cpp:2337` sets `stM.Kpen = kap = E/(3(1−2ν))` whenever
`<Penalty_parameter>` is absent, and `set_material_props.h:48` sets it again for
stVK. `compute_pk2cc` then adds an ST91 volumetric term **before** the StVK
term:

```cpp
if (!ustruct) {
  if (!utils::is_zero(Kp)) {
    compute_svol_p(com_mod, cep_mod, stM, J, p, pl);
    S += p * J * Ci;                      // <-- extra volumetric penalty
  }
}
...
case ConstitutiveModelType::stIso_StVK:
  S += g1*trE*Idm + g2*E;                 // <-- full StVK, lambda included
```

StVK's λ already carries the volumetric response, so this double-counts
compressibility. Measured cost: uy amplitude 52.354e-3 → 59.978e-3 when the
penalty is made negligible, i.e. **the default penalty alone throws away 13% of
the flap's amplitude** and adds 7% to its frequency. Your material is not the
Turek–Hron material.

Note that `struct` refuses `Kpen` of exactly zero
(`read_files.cpp:1618`, *"An incompressible material model is not allowed for
'struct' physics"*), so use a small non-zero value rather than 0:

```xml
<Constitutive_model type="stVK"/>
<Elasticity_modulus>1.4e6</Elasticity_modulus>
<Poisson_ratio>0.4</Poisson_ratio>
<Penalty_parameter>1.0</Penalty_parameter>   <!-- ~0 vs kappa = 2.33e6 -->
```

### 4.2 Linear triangles lock in bending

On the *identical* 230 nodes, splitting the quads into triangles costs another
20% of the amplitude (52.354e-3 → 41.913e-3) and pushes the frequency from
1.2270 to 1.3746 Hz. This is textbook constant-strain-triangle shear locking:
with 4 elements through a 0.02 m thickness, CST elements cannot represent the
linear-through-thickness bending strain.

### 4.3 How much mesh you actually need — measured

To turn the above into a number, I generated structured quad flaps directly from
the Turek–Hron geometry (curved root on the cylinder arc, `nlen × nthk`
elements) and ran CSM3 on each with `Penalty_parameter = 1.0`. The 45×4 case
reproduces your own quad mesh to within 1%, which validates the generator.

| case | ux mean (e-3) | uy mean (e-3) | uy amp (e-3) | freq (Hz) | amp err | freq err |
|---|---|---|---|---|---|---|
| **Turek–Hron reference** | **−14.305** | **−63.607** | **65.160** | **1.0995** | — | — |
| your TRIA 45×4 split, default penalty | −5.793 | −40.972 | 42.133 | 1.3736 | **−35.3%** | **+24.9%** |
| your QUAD 45×4, default penalty | −9.263 | −51.890 | 52.406 | 1.2252 | −19.6% | +11.4% |
| your QUAD 45×4, `Kpen`≈0 | −12.338 | −59.681 | 59.983 | 1.1412 | −7.9% | +3.8% |
| QUAD 45×4 (regenerated), `Kpen`≈0 | −12.164 | −59.274 | 59.381 | 1.1466 | −8.9% | +4.3% |
| QUAD 45×8, `Kpen`≈0 | −12.981 | −61.189 | 61.173 | 1.1285 | −6.1% | +2.6% |
| QUAD 90×8, `Kpen`≈0 | −13.944 | −63.329 | 63.343 | 1.1083 | −2.8% | +0.8% |
| QUAD 90×16, `Kpen`≈0 | −14.179 | −63.843 | 63.846 | 1.1035 | −2.0% | +0.4% |
| QUAD 120×24, `Kpen`≈0 | −14.380 | −64.279 | 64.286 | 1.0995 | −1.3% | **0.0%** |

The sequence converges monotonically onto the reference, and the finest mesh
reproduces the benchmark frequency exactly (1.0995 Hz) with a 1.3% amplitude
error. **That is an independent confirmation that svMultiPhysics' 2D `struct`
solid is correct** — the errors in your runs are entirely discretisation plus
the penalty setting, not formulation. Combined with CFD2 at 0.36%, both physics
are verified; what is left is meshing and input choices.

Reading down the table, in order of leverage:

1. **Triangles → quads** at fixed mesh: −35.3% → −19.6%.
2. **Kill the volumetric penalty** at fixed mesh: −19.6% → −7.9%. This is the
   single biggest lever and it is a one-line XML change.
3. **Refine uniformly.** Note that thickness refinement alone is *not* the
   answer — 45×4 → 45×8 only buys 1.8 points, while the uniform 2× refinement
   45×4 → 90×8 buys 5.1 points. Keep the elements roughly square; 90×8 is the
   sweet spot at 2.8% error and only 720 elements.

So a quad flap at 90×8 with `<Penalty_parameter>1.0</Penalty_parameter>` puts
you within 3% on amplitude and 1% on frequency — good enough that FSI2 lock-in
can actually happen. Your current tria 45×4 with the default penalty is 35% and
25% off.

The generator for these meshes is checked in at
[`utilities/turek-hron/generate_flap_mesh.py`](utilities/turek-hron/generate_flap_mesh.py):

```
python3 utilities/turek-hron/generate_flap_mesh.py ./meshes 90x8
```

It writes `mesh-complete.mesh.vtu` plus `fixed.vtp` / `interface.vtp` with the
`GlobalNodeID` / `GlobalElementID` arrays svMultiPhysics expects, and puts the
root edge on the cylinder arc rather than a straight line. Note it only makes
the *solid*; you still need a matching fluid mesh whose interface nodes line up
with the flap's.

**Combined effect on your tria FSI2 flap: 35% too little deflection and a
natural frequency 25% too high.** FSI2 is a lock-in between vortex shedding and
the structural mode; detune the structure by 25% and the limit cycle cannot
establish. That is why your tria FSI2 wanders at ±0.01 m instead of oscillating
at ±0.08 m, and why FSI3 (a stiffer flap, E = 5.6e6, less bending-dominated)
comes out closer to right.

This also explains the difference you noticed between your two meshes: the
**quad** FSI2 run was actually behaving correctly. Its tip amplitude was growing
steadily (0.0016 → 0.008 → 0.024 → 0.039 → 0.045 m) on its way to the right
answer when it died — of mesh failure, not of physics.

---

## 5. Why the quad FSI2 run crashed, and what the tria run is hiding

### Quad FSI2: element inversion at t = 10.86 s

Tracking the minimum deformed/reference element area ratio:

```
 t= 8.02  tip uy=-0.0237  min(area/area0)= 0.7733  inverted=0
 t= 8.82  tip uy= 0.0391  min(area/area0)= 0.5986  inverted=0
 t=10.42  tip uy=-0.0450  min(area/area0)= 0.4762  inverted=0
 t=10.84  tip uy= 0.0267  min(area/area0)= 0.2930  inverted=0
 t=10.86  tip uy=-0.0251  min(area/area0)=-2.5004  inverted=5   maxVel=49.4
```

Five elements invert, the velocity explodes to 49 m/s, the linear solver hits
its 51,000-iteration cap and the run diverges. This is a genuine ALE mesh
failure — the one place where remeshing would have helped.

### Tria FSI2: slow strangulation instead of a crash

The tria run survives 35 s but its mesh is destroyed:

```
 t=11.62  maxMeshDisp=0.0339  min(area/area0)=0.1888
 t=17.42  maxMeshDisp=0.0851  min(area/area0)=0.0244
 t=29.02  maxMeshDisp=0.1002  min(area/area0)=0.0092
 t=35.00  maxMeshDisp=0.1047  min(area/area0)=0.0107
```

Elements at **0.9% of their original area**. The fluid solution around the flap
after t ≈ 15 s is not meaningful.

### The mesh displacement ratchets

Probing a fluid node 10 mm downstream of the flap tip:

```
   t     flapTip_uy    meshDisp_x @probe
  2.02    0.000586        0.000011
 10.02   -0.015628       -0.002829
 14.02   -0.029461        0.013304
 20.02    0.008898        0.017826
 26.02    0.001756        0.020053
 35.00   -0.004184        0.020215
```

`meshDisp_y` tracks the flap correctly. But `meshDisp_x` grows monotonically to
+0.0202 m and stays there, while the flap tip that drives it moves about
−1e-4 m in x. **A node has been dragged 20 mm downstream by a boundary that
moved 0.1 mm.** That is accumulated drift in the incremental mesh update, not
physical motion.

`construct_mesh` re-references to the configuration at the start of each step
(`mesh.cpp:97–103`), which is the right approach for large motion but is
path-dependent by construction. One thing worth examining while you are in
there: [mesh.cpp:122](Code/Source/solver/mesh.cpp#L122) omits the Jacobian from
the Gauss weight —

```cpp
double w = lM.w(g);          // construct_mesh
double w = lM.w(g) * Jac;    // construct_l_elas, l_elas.cpp:120
```

This may be deliberate Jacobian-based stiffening (it makes small elements
relatively stiffer, which is a standard ALE heuristic), but it means the mesh
operator is not a consistent discretisation of any elasticity problem. I have
**not** changed it — I could not establish that it causes the ratcheting, and
changing it would alter every existing 3D FSI result. Flagging it for you.

---

## 6. Solver bug found and fixed: `<LS type="NS">` aborts

### The symptom

`<LS type="NS">` — the bi-partitioned (BIPN) solver, which is the *default* for
`fluid` equations and the right solver for saddle-point systems — dies
immediately on 2D problems:

```
FSILS: Singular matrix detected
```

Reproduced on the repository's own `tests/cases/fluid/driven_cavity_2d` just by
switching `GMRES` → `NS`.

### It is not actually 2D-specific

The 3D `tests/cases/fluid/pipe_RCR_3d` aborts the same way once you ask BIPN for
a tight tolerance (`<Tolerance>1e-9</Tolerance>`, `<Max_iterations>100</…>`). 2D
just hits it sooner.

### The cause

BIPN builds a 2-vectors-per-iteration search space and solves a small projection
system by Gauss elimination. When the newest pair becomes a linear combination
of the previous ones — the normal "search space exhausted, no further progress
possible" termination — `ge::ge()` returns false. `ns_solver.cpp` then did this:

```cpp
} else {
  if (lhs.commu.masF) {
    throw std::runtime_error("FSILS: Singular matrix detected");
  }
  xB = oldxB;              // <-- every other rank recovers gracefully
  if (i > 0) { iB -= 2; iBB -= 2; }
  break;
}
```

The non-master path is clearly the intended behaviour: fall back on the last
good coefficients and stop. Only rank 0 threw. In parallel that means rank 0
aborts while the others carry on.

### The fix

[`Code/Source/linear_solver/ns_solver.cpp`](Code/Source/linear_solver/ns_solver.cpp) —
replaced the master-rank throw with the same graceful stop plus a warning, so
all ranks take the same path.

### Verification

| check | result |
|---|---|
| `driven_cavity_2d`, GMRES (untouched path) | max diff vs shipped reference **9.6e-10** |
| `pipe_RCR_3d`, NS solver | max diff vs shipped reference **1.8e-12** |
| `driven_cavity_2d`, NS solver | now runs; matches GMRES reference to **1.1e-8** |
| `pipe_RCR_3d`, NS, tight tolerance | now runs to completion (previously aborted) |

### Why you should care

On your CFD3 case, 20 steps:

| configuration | wall time | per step |
|---|---|---|
| your `<LS type="GMRES">`, `Max_iterations 1000` | 306 s | 15.3 s |
| `<LS type="NS">` + BIPN iteration counts | 2.4 s | **0.12 s** |

**127× faster, same answer.**

It works for the FSI equation too. Krylov iterations for the `FS` equation on
your own extra-coarse tria FSI2 case (this is from your `histor.dat` versus a
rerun of the identical case with `<LS type="NS">`, so it is machine- and
load-independent):

| Newton step | your GMRES config | NS/BIPN |
|---|---|---|
| 1-1 | 15,248 | 6 |
| 1-2 | 14,229 | 6 |
| 2-1 | 9,897 | 6 |
| 2-2 | 19,172 | 6 |

Each BIPN iteration is more expensive than a GMRES iteration, so the wall-clock
gain is smaller than 2,500×, but it is still large — and it is the reason your
quad FSI2 run died: at t = 10.86 s the GMRES solve hit its 51,000-iteration cap
and never recovered.

One trap: `<Max_iterations>` means something completely different for the two
solvers. For GMRES it is the Krylov restart count (1000 is reasonable). For NS
it is the number of **BIPN outer iterations**, each of which is a full GMRES
solve on the momentum block plus a CG solve on the Schur complement. Use the
values from `pipe_RCR_3d`:

```xml
<LS type="NS">
  <Linear_algebra type="fsils"><Preconditioner>fsils</Preconditioner></Linear_algebra>
  <Max_iterations> 15 </Max_iterations>
  <NS_GM_max_iterations> 10 </NS_GM_max_iterations>
  <NS_CG_max_iterations> 300 </NS_CG_max_iterations>
  <Tolerance> 1e-3 </Tolerance>
</LS>
```

---

## 7. Remeshing: no, and not "exactly how it is done for 3D"

### It would not fix your problem

- CFD3 has **no mesh motion at all** and still fails. Remeshing is irrelevant to
  the main symptom.
- The tria FSI2 amplitude is 8× too small because the flap is too stiff (§4).
  Remeshing does not change the structure.
- It *would* have saved the quad FSI2 run at t = 10.86 s. That is real, but it
  is third on the list.

### It cannot be copied from the 3D path

`remesher_3d()` (`remesh.cpp:1287`) is a thin wrapper around
`remesh3d_tetgen()`, which calls the **Tetgen** library vendored at
`Code/ThirdParty/tetgen`. There is no 2D mesh generator anywhere in the
dependency set, so there is nothing to call. The 2D analogue needs a
constrained Delaunay triangulator — Triangle (non-commercial licence, a problem
for BSD-3 svMultiPhysics) or Gmsh (GPL) or a hand-rolled one. That is a
licensing and build-system decision, not a code-porting exercise, which is why I
have not done it unasked.

The `<Remesher type="Tetgen">` parameters do not carry over either: they drive a
tetrahedralisation of a closed surface triangulation, which has no 2D meaning.

### The good news

The hard part is already dimension-generic. `interp()` in `remesh.cpp:557`
explicitly declares support for both:

```cpp
if (nsd+1 != msh[iM].eNoN) {
  throw std::runtime_error("[interp] ... Can support 2D Tri or 3D Tet elements only.");
}
```

So the solution-transfer, node-distribution and element-search machinery
(`distrn`, `find_n`, `interp`) should work in 2D as-is. The remaining work is
bounded: a triangulator dependency, a `remesher_2d()` entry point, and removing
the guard at `remesh.cpp:1593`. Say the word and I will scope it properly —
but do §8 first, because I do not think you will need it.

---

## 8. What I would do, in order

**1. Switch the linear solver.** One XML change, 127× speedup, makes everything
below cheap to iterate on. Needs the `ns_solver.cpp` fix in this branch.

**2. Fix the flap material.** Add `<Penalty_parameter>1.0</Penalty_parameter>`
to the solid domain. Recovers ~13% of the amplitude for free.

**3. Use quadrilaterals for the solid and refine it uniformly to ~90×8.**
Per the table in §4.3 that lands you at 2.8% on amplitude and 0.8% on frequency.
Refining only through the thickness does *not* work — refine both directions and
keep the elements roughly square. Then re-run CSM3 and confirm you are within a
few percent of `uy = −63.607e-3 ± 65.160e-3` at 1.0995 Hz **before** touching
FSI2 again. This costs about 20 s of compute and it is a precondition for FSI2
meaning anything.

**4. Use `medium_example` (or finer) for the fluid.** Confirmed in §3.3: it
sheds, and reproduces CFD3 to 0.9% on mean drag and ~3% on lift amplitude. The
extra-coarse mesh cannot, at any time step or ρ∞. Until CFD3 sheds, FSI2 cannot
work, so this is a precondition. Drop `<Increment_in_saving_VTK_files>` to 4 so
the shedding frequency is resolvable.

**5. Only then run FSI2**, with ρ∞ ≈ 0.9–1.0 and dt ≤ 0.005 (you need ~50 steps
per period at the 3.8 Hz ux mode, and dt = 0.01 gives 26).

**6. If and only if the mesh then fails**, deal with mesh motion — first by
raising the mesh equation's `<Poisson_ratio>` towards 0.45–0.49 and improving
the grading around the flap-tip sweep region, and only after that by adding a
remesher.

Steps 3 and 4 are the ones that actually decide whether this benchmark
reproduces.

---

## 9. Files changed in this document's scope

```
Code/Source/linear_solver/ns_solver.cpp   (BIPN graceful termination)
```

plus the 2D kernel fixes described in
[`README_2D_CFD_FSI_investigation.md`](README_2D_CFD_FSI_investigation.md).
Nothing staged or committed.
