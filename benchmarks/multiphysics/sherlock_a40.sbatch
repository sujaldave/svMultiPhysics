#!/bin/bash
#SBATCH --job-name=svmp-physics
#SBATCH --output=svmp-physics-%j.log
#SBATCH --error=svmp-physics-%j.err
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=GPU_SKU:A40&CPU_SKU:7543P
#SBATCH --mem=24G
#SBATCH --time=04:00:00
set -euo pipefail
repo=${SVMP_REPO:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}
build=${SVMP_BUILD_DIR:-$repo/build}
exe=${SVMP_EXE:-$build/svMultiPhysics-build/bin/svmultiphysics}
output=${SVMP_OUTPUT:-$build/benchmarks/multiphysics-${SLURM_JOB_ID}}
[[ -x "$exe" && ! -e "$output" ]]
# The runner creates the output directory exclusively. Configuration and binding
# live in a sibling directory so interrupted suites remain identifiable.
config=${output}-config
mkdir -p "$(dirname "$config")"
mkdir "$config"
cp "$repo/benchmarks/multiphysics/run_suite.py" "$repo/benchmarks/multiphysics/summarize.py" "$config/"
git -C "$repo" rev-parse HEAD > "$config/commit.txt"
git -C "$repo" status --short > "$config/git-status.txt"
git -C "$repo" diff > "$config/source.diff"
sha256sum "$exe" > "$config/executable.sha256"
nvidia-smi -q > "$config/nvidia-smi.txt"
scontrol show job "$SLURM_JOB_ID" > "$config/slurm-job.txt"
module list > "$config/modules.txt" 2>&1 || true
ldd "$exe" > "$config/ldd.txt" 2>&1 || true
for cache in CMakeCache.txt svMultiPhysics-build/CMakeCache.txt; do
  if [[ -f "$build/$cache" ]]; then cp "$build/$cache" "$config/${cache//\//_}"; fi
done
srun --nodes=1 --exact --ntasks=1 --cpus-per-task=1 --cpu-bind=cores \
  --gpus-per-node=1 --gpus-per-task=1 --gpu-bind=single:1 \
  python3 "$repo/benchmarks/gpu_scaling/check_binding.py" "$config/binding"
python3 "$repo/benchmarks/gpu_scaling/check_binding.py" "$config/binding" --validate 1
finish() {
  local status=$?
  trap - EXIT
  if [[ -f "$output/manifest.json" ]]; then
    python3 "$repo/benchmarks/multiphysics/summarize.py" "$output" --expect-cuda || status=1
  fi
  exit "$status"
}
trap finish EXIT
# Individual solver failures/timeouts are retained and do not cancel the other
# policies. These physics use host element integration even in a CUDA build.
python3 "$repo/benchmarks/multiphysics/run_suite.py" \
  --exe "$exe" --output "$output" --launcher srun \
  --steps "${SVMP_STEPS:-20}" --repeat "${SVMP_REPEAT:-2}" \
  --timeout "${SVMP_TIMEOUT:-180}"
echo "Results: $output; configuration: $config"
