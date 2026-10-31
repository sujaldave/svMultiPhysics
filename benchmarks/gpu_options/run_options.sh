#!/bin/bash
# Run every solver-option variant in fluid_variants.txt and physics_variants.txt.
# Usage: run_options.sh local|slurm OUTPUT_DIR
#
# Environment:
#   SVMP_EXE        solver (default: build/svMultiPhysics-build/bin/svmultiphysics)
#   SVMP_PARTS      "fluid physics" (default) or a subset
#   SVMP_CASES      fluid cases (default: "iliac pipe")
#   SVMP_PIPE_CASE  generated pipe case (default: build/benchmarks/large-pipe/case)
#   SVMP_RANKS      MPI ranks in local mode (default 1; slurm mode uses one GPU)
#   SVMP_FLUID_VARIANTS / SVMP_PHYSICS_VARIANTS  label subsets (default: all)
#   SVMP_TIMEOUT    seconds per physics launch (default 600)
# A failed variant (e.g. Tacho absent from this Trilinos build) is recorded and
# the sweep continues. Runs are sequential.
set -uo pipefail

mode=${1:?usage: run_options.sh local|slurm OUTPUT_DIR}
out=${2:?usage: run_options.sh local|slurm OUTPUT_DIR}
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
exe=${SVMP_EXE:-$repo/build/svMultiPhysics-build/bin/svmultiphysics}
parts=${SVMP_PARTS:-"fluid physics"}
cases=${SVMP_CASES:-"iliac pipe"}
pipe_case=${SVMP_PIPE_CASE:-$repo/build/benchmarks/large-pipe/case}
[[ $mode == local || $mode == slurm ]] || { echo "mode must be local or slurm" >&2; exit 1; }
[[ -x $exe ]] || { echo "missing executable $exe" >&2; exit 1; }
[[ ! -e $out ]] || { echo "$out exists; choose a new output directory" >&2; exit 1; }
mkdir -p "$out"
cp "$here"/*_variants.txt "$out/"
echo "$exe" > "$out/executable.txt"

selected() {  # selected LABEL "SUBSET"
  [[ -z $2 ]] && return 0
  [[ " $2 " == *" $1 "* ]]
}

launch=(--ranks "${SVMP_RANKS:-1}")
[[ $mode == slurm ]] && launch=(--launcher srun --srun-exact --cpu-bind cores
    --gpus-per-node 1 --gpus-per-task 1 --ranks 1)

if [[ " $parts " == *" fluid "* ]]; then
  for case in $cases; do
    steps=20; case_args=()
    [[ $case == pipe ]] && { steps=10; case_args=(--case "$pipe_case"); }
    while read -r label settings; do
      [[ -z $label || $label == \#* ]] && continue
      selected "$label" "${SVMP_FLUID_VARIANTS:-}" || continue
      echo "[fluid/$case] $label $settings"
      env $settings python3 "$repo/benchmarks/schur_amg/run_benchmark.py" --exe "$exe" \
        "${launch[@]}" ${case_args[@]+"${case_args[@]}"} --preconditioner schur-amg \
        --gmres-preconditioner resistance --assembly kokkos --steps "$steps" \
        --stats --assembly-stats --output "$out/fluid/$case/$label" < /dev/null \
        || echo "$case/$label" >> "$out/failed.txt"
    done < "$here/fluid_variants.txt"
  done
fi

if [[ " $parts " == *" physics "* ]]; then
  suite_launch=()
  [[ $mode == slurm ]] && suite_launch=(--launcher srun)
  while read -r label settings; do
    [[ -z $label || $label == \#* ]] && continue
    selected "$label" "${SVMP_PHYSICS_VARIANTS:-}" || continue
    envs=()
    for item in $settings; do envs+=(--env "$item"); done
    echo "[physics] $label $settings"
    python3 "$repo/benchmarks/multiphysics/run_suite.py" --exe "$exe" ${suite_launch[@]+"${suite_launch[@]}"} \
      --cases solid fsi scalar --variants gmres-ml --assemblies trilinos --repeat 1 \
      --steps 20 --timeout "${SVMP_TIMEOUT:-600}" ${envs[@]+"${envs[@]}"} \
      --output "$out/physics/$label" < /dev/null || echo "physics/$label" >> "$out/failed.txt"
  done < "$here/physics_variants.txt"
fi

python3 "$here/summarize.py" "$out" || true
echo "Results: $out"
