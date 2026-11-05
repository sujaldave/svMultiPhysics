#!/bin/bash
# Submit the Marlowe H100 scaling campaign: for every case, one 1-node job
# (1, 2, 4, 8 GPUs) and one 2-node job (16 GPUs), all under one campaign root.
#
# Run from anywhere on a Marlowe login node:
#   SVMP_BUILD_DIR=/path/to/svMultiPhysics/build benchmarks/gpu_scaling/submit_marlowe.sh
#
# Environment (all optional except where the defaults do not fit):
#   SVMP_BUILD_DIR   build directory with svMultiPhysics-build/bin/svmultiphysics (default: ./build)
#   SVMP_ENV         svmp_marlowe_env.sh from buildLibrariesAndSVMP.sh (default: none;
#                    the job then loads gcc/13.1.0 cmake/3.30.3 nvhpc/24.7 + HPC-X 2.19)
#   SVMP_CAMPAIGN    campaign root (default: $SVMP_BUILD_DIR/benchmarks/marlowe-scale-DATE)
#   SVMP_CASES       cases, one job each per node count (default: "iliac pipe")
#   SVMP_NODE_COUNTS node counts (default: "1 2")
#   SVMP_PIPE_CASE   generated pipe case (default: ./build/benchmarks/large-pipe/case);
#                    generate it once beforehand or rsync it from Sherlock (/scratch/users/$USER/pipe-case)
#   SVMP_ACCOUNT, SVMP_PARTITION, SVMP_TIME   sbatch overrides of the script header
#   SVMP_DRY_RUN=1   print the sbatch commands only
# Every other SVMP_* setting of marlowe_h100_scaling.sbatch (SVMP_BIND_MODES,
# SVMP_COMM_MODES, SVMP_REPEAT, ...) is passed through unchanged.
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
cd "$repo"   # the job uses SLURM_SUBMIT_DIR as the repository root
build=$(cd "${SVMP_BUILD_DIR:-$repo/build}" && pwd)
campaign=${SVMP_CAMPAIGN:-$build/benchmarks/marlowe-scale-$(date +%Y%m%d-%H%M%S)}
read -r -a cases <<< "${SVMP_CASES:-iliac pipe}"
read -r -a node_counts <<< "${SVMP_NODE_COUNTS:-1 2}"
pipe_case=${SVMP_PIPE_CASE:-$repo/build/benchmarks/large-pipe/case}
script=$repo/benchmarks/gpu_scaling/marlowe_h100_scaling.sbatch

[[ -x $build/svMultiPhysics-build/bin/svmultiphysics ]] ||
  { echo "ERROR: no executable in $build/svMultiPhysics-build/bin" >&2; exit 1; }
if [[ " ${cases[*]} " == *" pipe "* && ! -f $pipe_case/mesh/mesh-complete.mesh.vtu ]]; then
  echo "NOTE: $pipe_case has no pipe mesh yet; every pipe job would generate it (needs NumPy)." >&2
  echo "      Generate it once first: benchmarks/large_pipe/prepare_case.sh $pipe_case <python-with-numpy>" >&2
  [[ ${SVMP_DRY_RUN:-0} == 1 ]] || exit 1
fi
mkdir -p "$campaign"
(git rev-parse HEAD && git status --short) > "$campaign/source.txt" 2>&1 || true

extra=()
[[ -n ${SVMP_ACCOUNT:-} ]] && extra+=(--account="$SVMP_ACCOUNT")
[[ -n ${SVMP_PARTITION:-} ]] && extra+=(--partition="$SVMP_PARTITION")
[[ -n ${SVMP_TIME:-} ]] && extra+=(--time="$SVMP_TIME")
for case_name in "${cases[@]}"; do
  for nodes in "${node_counts[@]}"; do
    out=$campaign/$case_name-nodes$nodes
    cmd=(sbatch --parsable --nodes="$nodes" --job-name="svmp-h100-$case_name-n$nodes" ${extra[@]+"${extra[@]}"} "$script")
    vars=(SVMP_BUILD_DIR="$build" SVMP_OUTPUT="$out" SVMP_CASES="$case_name" SVMP_PIPE_CASE="$pipe_case")
    echo "[$case_name, $nodes node(s)] ${vars[*]} ${cmd[*]}" | tee -a "$campaign/submissions.txt"
    [[ ${SVMP_DRY_RUN:-0} == 1 ]] && continue
    id=$(env "${vars[@]}" "${cmd[@]}")
    echo "$case_name-nodes$nodes ${id%%;*}" >> "$campaign/jobs.txt"
  done
done
cat <<EOF

Campaign root: $campaign   (job IDs: $campaign/jobs.txt)
Watch:   squeue -u \$USER
Summary on Marlowe (timings, binding audit):
  python3 benchmarks/gpu_scaling/summarize_marlowe.py $campaign --expect-cuda
Then copy the campaign root back and rerun the same command locally with Python VTK
for the field comparisons.
EOF
