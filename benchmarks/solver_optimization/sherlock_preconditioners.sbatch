#!/bin/bash
#SBATCH --job-name=svmp-pc
#SBATCH --output=svmp-pc-%j.log
#SBATCH --error=svmp-pc-%j.err
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=GPU_SKU:A40&CPU_SKU:7543P
#SBATCH --mem=24G
#SBATCH --time=00:20:00

set -euo pipefail
repo=${SVMP_REPO:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}
build=${SVMP_BUILD_DIR:-$repo/build}
exe=${SVMP_EXE:-$build/svMultiPhysics-build/bin/svmultiphysics}
output=${SVMP_OUTPUT:-$repo/build/benchmarks/preconditioners-${SLURM_JOB_ID}}
steps=${SVMP_STEPS:-20}
repeat=${SVMP_REPEAT:-2}
[[ -x "$exe" && ! -e "$output" ]]
mkdir -p "$output/config"
git -C "$repo" rev-parse HEAD > "$output/config/commit.txt"
git -C "$repo" status --short > "$output/config/git-status.txt"
git -C "$repo" diff > "$output/config/source.diff"
sha256sum "$exe" > "$output/config/executable.sha256"
nvidia-smi -q > "$output/config/nvidia-smi.txt"
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
  python3 "$repo/benchmarks/solver_optimization/summarize.py" "$output" \
    --reference-group amg-diagonal --expect-cuda \
    --allow-diagonal-inner-nonconvergence || summary_status=$?
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

# Keep diagnostics outside the timing samples. The same native assembly,
# storage reuse, boundary kernels and FP64 physics are used by all variants.
"${benchmark[@]}" --preconditioner schur-amg --gmres-preconditioner diagonal \
  --steps 2 --runtime-details --output "$output/runtime-details"

for label in amg-diagonal classic-resistance classic-diagonal amg-resistance; do
  case "$label" in
    amg-diagonal)       pressure=schur-amg; momentum=diagonal; operator=explicit ;;
    amg-resistance)     pressure=schur-amg; momentum=resistance; operator=explicit ;;
    classic-diagonal)   pressure=diagonal; momentum=diagonal; operator=factored ;;
    classic-resistance) pressure=diagonal; momentum=resistance; operator=factored ;;
  esac
  "${benchmark[@]}" --preconditioner "$pressure" --gmres-preconditioner "$momentum" \
    --schur-operator "$operator" --steps "$steps" --repeat "$repeat" \
    --output "$output/$label"
done
echo "Preconditioner comparison: $output"
