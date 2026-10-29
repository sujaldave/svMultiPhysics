#!/bin/bash
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
output=${SVMP_OUTPUT:-$repo/build/benchmarks/amg-application/cpu}
exe=${SVMP_EXE:-$repo/build/svMultiPhysics-build/bin/svmultiphysics}
for coarse in klu chebyshev; do
  export SV_TRILINOS_SCHUR_COARSE=$coarse
  python3 "$repo/benchmarks/schur_amg/run_benchmark.py" --exe "$exe" \
    --output "$output/$coarse" --preconditioner schur-amg \
    --gmres-preconditioner resistance --assembly kokkos --stats --assembly-stats \
    --steps "${SVMP_STEPS:-20}" --ranks "${SVMP_RANKS:-2}" --repeat 2
 done
python3 "$repo/benchmarks/solver_optimization/summarize.py" "$output" --reference-group klu
