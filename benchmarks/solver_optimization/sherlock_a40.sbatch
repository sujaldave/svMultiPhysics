#!/bin/bash
#SBATCH --job-name=svmp-schur
#SBATCH --output=svmp-schur-%j.log
#SBATCH --error=svmp-schur-%j.err
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=GPU_SKU:A40&CPU_SKU:7543P
#SBATCH --mem=24G
#SBATCH --time=00:15:00

set -euo pipefail
# Submit from the repository root, using the same modules/environment as the
# successful GPU build. Optional overrides are described in the README.
repo=${SVMP_REPO:-${SLURM_SUBMIT_DIR:?Submit with sbatch from the repository root}}
build=${SVMP_BUILD_DIR:-$repo/build}
exe=${SVMP_EXE:-$build/svMultiPhysics-build/bin/svmultiphysics}
output=${SVMP_OUTPUT:-$repo/build/benchmarks/sherlock-${SLURM_JOB_ID}}
steps=${SVMP_STEPS:-5}
repeat=${SVMP_REPEAT:-2}
ranks=${SVMP_RANKS:-1}
mkdir -p "$output/config"
[[ -x "$exe" && -f "$repo/benchmarks/schur_amg/run_benchmark.py" ]]

git -C "$repo" rev-parse HEAD > "$output/config/commit.txt"
git -C "$repo" status --short > "$output/config/git-status.txt"
git -C "$repo" diff > "$output/config/source.diff"
uname -a > "$output/config/uname.txt"
nvidia-smi -q > "$output/config/nvidia-smi.txt"
scontrol show job "$SLURM_JOB_ID" > "$output/config/slurm-job.txt"
command -v nvcc > "$output/config/nvcc-path.txt" || true
nvcc --version > "$output/config/nvcc-version.txt" 2>&1 || true
module list > "$output/config/modules.txt" 2>&1 || true
ldd "$exe" > "$output/config/ldd.txt" 2>&1 || true
sha256sum "$exe" > "$output/config/executable.sha256"
for cache in CMakeCache.txt svMultiPhysics-build/CMakeCache.txt; do
  if [[ -f "$build/$cache" ]]; then
    cp "$build/$cache" "$output/config/${cache//\//_}"
  fi
done
if [[ -n ${SVMP_TRILINOS_BUILD:-} ]]; then
  cp "$SVMP_TRILINOS_BUILD/CMakeCache.txt" "$output/config/trilinos-CMakeCache.txt"
fi

benchmark=(python3 "$repo/benchmarks/schur_amg/run_benchmark.py"
  --launcher srun --gpus-per-task 1 --ranks "$ranks" --threads 1
  --preconditioner schur-amg --stats --assembly-stats)

# Produce a partial summary even if a solver or profiler command fails.
finish() {
  local run_status=$?
  local summary_status=0
  trap - EXIT
  python3 "$repo/benchmarks/solver_optimization/summarize.py" "$output" --expect-cuda || summary_status=$?
  if (( run_status != 0 )); then exit "$run_status"; fi
  exit "$summary_status"
}
trap finish EXIT

# An old executable is the genuine before/after comparison. The fresh/factored
# run below is an ablation of the new executable, not an old-version substitute.
if [[ -n ${SVMP_BASELINE_EXE:-} ]]; then
  sha256sum "$SVMP_BASELINE_EXE" > "$output/config/baseline.sha256"
  "${benchmark[@]}" --exe "$SVMP_BASELINE_EXE" --assembly kokkos \
    --steps "$steps" --repeat "$repeat" --output "$output/old"
fi

"${benchmark[@]}" --exe "$exe" --assembly kokkos --steps "$steps" \
  --schur-reuse fresh --schur-operator factored --output "$output/fresh-factored"
"${benchmark[@]}" --exe "$exe" --assembly kokkos --steps "$steps" \
  --schur-reuse transfers --schur-operator factored --output "$output/reuse-factored"
"${benchmark[@]}" --exe "$exe" --assembly kokkos --steps "$steps" \
  --schur-reuse transfers --schur-operator explicit --repeat "$repeat" \
  --output "$output/reuse-explicit"
"${benchmark[@]}" --exe "$exe" --assembly legacy --steps "$steps" \
  --output "$output/legacy-assembly"

# A separate short run prints Kokkos configuration and AMG hierarchy details.
# Keep configuration printing and profiling outside the timed comparisons.
"${benchmark[@]}" --exe "$exe" --assembly kokkos --steps 2 --runtime-details \
  --output "$output/runtime-details"
if command -v nsys >/dev/null 2>&1; then
  "${benchmark[@]}" --exe "$exe" --assembly kokkos --steps 2 --profile nsys \
    --output "$output/profile"
else
  echo 'nsys unavailable in this job environment; load Nsight Systems and rerun the profile command.' \
    > "$output/profile-unavailable.txt"
fi

test_exe=$build/svMultiPhysics-build/bin/run_all_unit_tests
if [[ -x "$test_exe" ]]; then
  srun --ntasks "$ranks" --cpus-per-task 1 --gpus-per-task 1 --gpu-bind=single:1 \
    "$test_exe" --gtest_filter='TrilinosSchurAMG.*:TrilinosResistanceOperator.*:TrilinosNumericExport.*' \
    > "$output/unit-tests.log" 2>&1
else
  echo 'Unit test executable unavailable; configure ENABLE_UNIT_TEST=ON to include it.' \
    > "$output/unit-tests-unavailable.txt"
fi

# Optional matched CPU build on the same host (still within this allocation).
if [[ -n ${SVMP_CPU_EXE:-} ]]; then
  python3 "$repo/benchmarks/schur_amg/run_benchmark.py" \
    --exe "$SVMP_CPU_EXE" --launcher srun --ranks 1 --threads 1 \
    --preconditioner schur-amg --assembly kokkos --steps "$steps" --repeat "$repeat" \
    --stats --assembly-stats --output "$output/cpu"
fi

echo "Results, configuration, and profiling artifacts: $output"
