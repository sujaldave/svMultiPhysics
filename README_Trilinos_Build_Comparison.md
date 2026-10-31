# Rerun all A40 suites with a new Trilinos/Kokkos build

`build/` links Trilinos 15.1 built with `Kokkos_ARCH_AMPERE80`. Every earlier
A40 log warns that these compute-capability 8.0 kernels run on the A40's 8.6
device. `build2/` links Trilinos 17.2 built with `Kokkos_ARCH_AMPERE86`. One
submitter reruns every earlier suite with `build2`, and one script compares
the results with both the earlier jobs and same-job runs of the old build.

## Submit (Sherlock login node, repository root)

Load the modules `build2` needs (sbatch copies this environment into each job),
then:

```bash
SVMP_PYTHON=/scratch/users/sdave03/svv/bin/python \
  benchmarks/trilinos_comparison/submit_all.sh
```

`SVMP_DRY_RUN=1` prints the five `sbatch` commands without submitting.
Before submitting, the script checks that both executables resolve all their
libraries in the current environment. It records each build's Trilinos version,
Kokkos architecture macros, linked libraries and hash in `build-old.txt` and
`build-new.txt`.

| Suite | Script | Settings (as before) | Old-build control in the same job |
|---|---|---|---|
| `schur` | `solver_optimization/sherlock_a40.sbatch` | iliac, 20 steps, 2 repeats | `old/` (old build, 2 repeats) |
| `pc` | `solver_optimization/sherlock_preconditioners.sbatch` | iliac, 4 preconditioner pairs, 20 steps, 2 repeats | — |
| `pipe` | `large_pipe/sherlock_a40.sbatch` | pipe 10 steps ×2, iliac, classical CG | `pipe/old-amg-resistance` ×2, `iliac/old-amg-resistance` |
| `scale` | `gpu_scaling/sherlock_a40_scaling.sbatch` | 1/2/4 A40s, 20 steps, 3 repeats; limit raised to 2 h | — |
| `physics` | `multiphysics/sherlock_a40.sbatch` | scalar/solid/FSI, 20 steps, 2 repeats, 180 s per launch | — |

All results go under one root, by default `build2/benchmarks/rerun-DATE/`
(`physics-config/` sits next to `physics/`). The jobs run concurrently;
`SVMP_CHAIN=1` makes each start after the previous one finishes, and
`SVMP_SUITES="schur pipe"` selects a subset. Job IDs are written to
`jobs.txt`. Each job starts with only the `SVMP_*` variables the submitter sets.

## Compare (on the Mac, after copying the root back)

```bash
/path/to/python-with-vtk benchmarks/trilinos_comparison/compare_builds.py \
  /local/copy/rerun-DATE \
  --old-root /Users/sujaldave/Documents/ChatGPT/svmpWithGPU/oct27-tests
```

The script reruns each suite's summarizer, adding VTK field checks, and writes
`comparison.md` and `comparison.json` in the root. For every suite it reports:

- **Same allocation** (schur, pipe, iliac): old vs new executable on the same
  node and GPU. This is the clean build comparison.
- **Against the earlier job**: wall time, Newton and CG iterations, µs per CG
  iteration, CG, Schur setup, AMG setup, GMRES and assembly time, for every
  group. These rows also include node/time variation. For schur, pc, scale
  and physics the October jobs ran binary `d4abecb6…` while `build/` now holds
  `79c4f195…` (the pipe job's binary). Those rows therefore also include source
  changes since October; the report states this for each suite.
- **Final fields** of each new run against the matching old run, over all saved
  point fields. Tolerances are 1e-7 for fluid/FSI and 1e-4 for scalar/solid.
  Groups that capped linear solves or did not validate (classical CG, some
  physics policies) are labeled informational: another Trilinos version can
  legitimately take a different iteration path.
- **Architecture**: how many logs still contain the SM 8.0-on-8.6 Kokkos warning
  (all earlier logs do; with `AMPERE86` the count should be zero).

Overrides: `--old SUITE=DIR` points a suite at a different earlier job,
`--no-refresh` skips the summarizer reruns, and `--no-fields` skips field
comparisons.

The comparison tool was checked locally by comparing the October results
against themselves. All 39 comparable groups give speedup 1.00 and zero field
differences. The six groups that timed out or crashed in October report no
final state, and FSI outputs with coincident interface points are compared by
point index.
