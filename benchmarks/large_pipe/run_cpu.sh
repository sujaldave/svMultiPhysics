#!/bin/bash
# Local CPU rank sweep: Lasso's 1M-tet pipe (10 steps) and the iliac artery
# (20 steps) with the current best BIPN policy (Schur AMG + resistance GMRES,
# native Kokkos assembly). Runs are sequential; do not overlap with builds.
#
# Environment overrides:
#   SVMP_EXE        solver executable (default: build/svMultiPhysics-build/bin/svmultiphysics)
#   SVMP_OUTPUT     new output directory (default: build/benchmarks/large-pipe/cpu-JOBTIME)
#   SVMP_PIPE_CASE  generated pipe case (default: build/benchmarks/large-pipe/case)
#   SVMP_PYTHON     Python with NumPy, used only if the pipe case must be generated
#   SVMP_RANKS      rank counts (default: "1 2 4 8 10")
#   SVMP_REPEAT_AT  rank count measured twice (default: 10)
#   SVMP_ILIAC=0    skip the iliac sweep;  SVMP_CLASSIC=0 skip the classical-BIPN pipe run
set -euo pipefail

repo=$(cd "$(dirname "$0")/../.." && pwd)
exe=${SVMP_EXE:-$repo/build/svMultiPhysics-build/bin/svmultiphysics}
out=${SVMP_OUTPUT:-$repo/build/benchmarks/large-pipe/cpu-$(date +%Y%m%d-%H%M%S)}
case_dir=${SVMP_PIPE_CASE:-$repo/build/benchmarks/large-pipe/case}
ranks=${SVMP_RANKS:-"1 2 4 8 10"}
repeat_at=${SVMP_REPEAT_AT:-10}
runner=$repo/benchmarks/schur_amg/run_benchmark.py

[[ -e "$out" ]] && { echo "ERROR: $out exists; choose a new SVMP_OUTPUT" >&2; exit 1; }
"$repo/benchmarks/large_pipe/prepare_case.sh" "$case_dir" "${SVMP_PYTHON:-python3}"
mkdir -p "$out/pipe" "$out/iliac"
{ git -C "$repo" rev-parse HEAD; git -C "$repo" status --short; } > "$out/source.txt"
shasum -a 256 "$exe" > "$out/executable.sha256"
sysctl -n machdep.cpu.brand_string hw.ncpu hw.memsize > "$out/machine.txt" 2>/dev/null || uname -a > "$out/machine.txt"

common=(--exe "$exe" --preconditioner schur-amg --gmres-preconditioner resistance
        --assembly kokkos --stats --assembly-stats)
for r in $ranks; do
  n=1; [[ $r == "$repeat_at" ]] && n=2
  python3 "$runner" "${common[@]}" --case "$case_dir" --steps 10 --ranks "$r" \
    --repeat "$n" --output "$out/pipe/cpu-$r"
done
if [[ ${SVMP_CLASSIC:-1} == 1 ]]; then
  # Classical BIPN (diagonal pressure CG) with the same assembly: the weak baseline.
  python3 "$runner" --exe "$exe" --preconditioner diagonal --gmres-preconditioner resistance \
    --assembly kokkos --stats --assembly-stats --schur-operator factored \
    --case "$case_dir" --steps 10 --ranks "$repeat_at" --output "$out/pipe/classic-cpu-$repeat_at"
fi
if [[ ${SVMP_ILIAC:-1} == 1 ]]; then
  for r in $ranks; do
    python3 "$runner" "${common[@]}" --steps 20 --ranks "$r" --output "$out/iliac/cpu-$r"
  done
fi
echo "Results: $out"
