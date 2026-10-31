#!/bin/bash
#SBATCH --job-name=svmp-pipe
#SBATCH --output=svmp-pipe-%j.log
#SBATCH --error=svmp-pipe-%j.err
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=GPU_SKU:A40&CPU_SKU:7543P
#SBATCH --mem=48G
#SBATCH --time=01:30:00

# One-A40 benchmark of Lasso's 1M-tetrahedra pipe (181,875 nodes) with the
# current branch, plus a same-allocation iliac control. No C++ rebuild needed.
#
# Groups under $output/pipe (10 steps) and $output/iliac (20 steps):
#   runtime-details      pipe, 1 step, Kokkos/AMG hierarchy details (not timed)
#   gpu-amg-resistance   pipe, Schur AMG + resistance GMRES (best policy), repeated
#   gpu-classic-resistance  pipe, classical diagonal pressure CG (weak baseline)
#   iliac/gpu-amg-resistance  iliac control, compare with the earlier ~39.5 s
#   profile              pipe, 1 step under Nsight Systems (SVMP_PROFILE=1 only)
#
# Overrides: SVMP_REPO, SVMP_BUILD_DIR, SVMP_EXE, SVMP_OUTPUT, SVMP_PIPE_CASE,
# SVMP_PYTHON (needs NumPy, only to generate the case), SVMP_STEPS (10),
# SVMP_REPEAT (2), SVMP_CLASSIC (1), SVMP_ILIAC (1), SVMP_PROFILE (0).

set -euo pipefail
repo=${SVMP_REPO:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}
build=${SVMP_BUILD_DIR:-$repo/build}
exe=${SVMP_EXE:-$build/svMultiPhysics-build/bin/svmultiphysics}
output=${SVMP_OUTPUT:-$repo/build/benchmarks/large-pipe-${SLURM_JOB_ID}}
case_dir=${SVMP_PIPE_CASE:-$repo/build/benchmarks/large-pipe/case}
steps=${SVMP_STEPS:-10}
repeat=${SVMP_REPEAT:-2}
[[ -x "$exe" ]] || { echo "ERROR: missing executable $exe" >&2; exit 1; }
[[ ! -e "$output" ]] || { echo "ERROR: $output exists" >&2; exit 1; }

# The case is generated once and reused; or rsync build/benchmarks/large-pipe/case
# from the Mac and point SVMP_PIPE_CASE at it.
"$repo/benchmarks/large_pipe/prepare_case.sh" "$case_dir" "${SVMP_PYTHON:-python3}"

mkdir -p "$output/config" "$output/pipe" "$output/iliac"
git -C "$repo" rev-parse HEAD > "$output/config/commit.txt"
git -C "$repo" status --short > "$output/config/git-status.txt"
git -C "$repo" diff > "$output/config/source.diff"
sha256sum "$exe" > "$output/config/executable.sha256"
sha256sum "$case_dir/mesh/mesh-complete.mesh.vtu" > "$output/config/pipe-mesh.sha256"
nvidia-smi -q > "$output/config/nvidia-smi.txt"
nvidia-smi topo -m > "$output/config/gpu-topology.txt" 2>&1 || true
scontrol show job "$SLURM_JOB_ID" > "$output/config/slurm-job.txt"
module list > "$output/config/modules.txt" 2>&1 || true
ldd "$exe" > "$output/config/ldd.txt" 2>&1 || true
for cache in CMakeCache.txt svMultiPhysics-build/CMakeCache.txt; do
  if [[ -f "$build/$cache" ]]; then cp "$build/$cache" "$output/config/${cache//\//_}"; fi
done

finish() {
  local run_status=$?
  local summary_status=0
  trap - EXIT
  for group in pipe iliac; do
    [[ -n "$(ls "$output/$group" 2>/dev/null)" ]] || continue
    python3 "$repo/benchmarks/solver_optimization/summarize.py" "$output/$group" \
      --reference-group gpu-amg-resistance --expect-cuda \
      --allow-diagonal-inner-nonconvergence || summary_status=$?
  done
  python3 "$repo/benchmarks/large_pipe/report.py" "$output"/pipe "$output"/iliac \
    > "$output/report.md" || true
  if (( run_status != 0 )); then exit "$run_status"; fi
  exit "$summary_status"
}
trap finish EXIT

srun --nodes=1 --exact --ntasks=1 --cpus-per-task=1 --cpu-bind=cores \
  --gpus-per-node=1 --gpus-per-task=1 --gpu-bind=single:1 \
  python3 "$repo/benchmarks/gpu_scaling/check_binding.py" "$output/binding" \
  > "$output/config/binding.log" 2>&1
python3 "$repo/benchmarks/gpu_scaling/check_binding.py" "$output/binding" --validate 1

benchmark=(python3 "$repo/benchmarks/schur_amg/run_benchmark.py"
  --exe "$exe" --launcher srun --srun-exact --cpu-bind cores
  --gpus-per-node 1 --gpus-per-task 1 --ranks 1 --threads 1
  --assembly kokkos --stats --assembly-stats --schur-reuse transfers)

# Diagnostics first, outside the timing samples.
"${benchmark[@]}" --case "$case_dir" --preconditioner schur-amg \
  --gmres-preconditioner resistance --steps 1 --runtime-details \
  --output "$output/pipe/runtime-details"

"${benchmark[@]}" --case "$case_dir" --preconditioner schur-amg \
  --gmres-preconditioner resistance --steps "$steps" --repeat "$repeat" \
  --output "$output/pipe/gpu-amg-resistance"

if [[ ${SVMP_ILIAC:-1} == 1 ]]; then
  "${benchmark[@]}" --preconditioner schur-amg --gmres-preconditioner resistance \
    --steps 20 --output "$output/iliac/gpu-amg-resistance"
fi

# The classical pressure CG usually stops at its 300-iteration cap; it is the
# reference for how much Schur AMG contributes at this size, not a candidate.
if [[ ${SVMP_CLASSIC:-1} == 1 ]]; then
  "${benchmark[@]}" --case "$case_dir" --preconditioner diagonal \
    --gmres-preconditioner resistance --schur-operator factored --steps "$steps" \
    --output "$output/pipe/gpu-classic-resistance"
fi

if [[ ${SVMP_PROFILE:-0} == 1 ]] && command -v nsys > /dev/null; then
  "${benchmark[@]}" --case "$case_dir" --preconditioner schur-amg \
    --gmres-preconditioner resistance --steps 1 --profile nsys \
    --output "$output/pipe/profile"
fi
echo "Large-pipe results: $output"
