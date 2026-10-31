#!/bin/bash
#SBATCH --job-name=svmp-options
#SBATCH --output=svmp-options-%j.log
#SBATCH --error=svmp-options-%j.err
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=GPU_SKU:A40&CPU_SKU:7543P
#SBATCH --mem=48G
#SBATCH --time=04:00:00

# One-A40 sweep of the new solver options (benchmarks/gpu_options/*_variants.txt):
#   fluid:   iliac (20 steps) and Lasso's 1M-tet pipe (10 steps), BIPN + Schur AMG
#   physics: solid, FSI, scalar transport with GMRES + MueLu (generic solve)
# Requires a build of this branch. Overrides: SVMP_BUILD_DIR, SVMP_EXE,
# SVMP_OUTPUT, SVMP_PIPE_CASE, SVMP_PYTHON (NumPy, only to generate the pipe),
# SVMP_PARTS ("fluid physics"), SVMP_CASES, SVMP_FLUID_VARIANTS,
# SVMP_PHYSICS_VARIANTS, SVMP_TIMEOUT.
set -euo pipefail
repo=${SVMP_REPO:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}
build=${SVMP_BUILD_DIR:-$repo/build}
export SVMP_EXE=${SVMP_EXE:-$build/svMultiPhysics-build/bin/svmultiphysics}
output=${SVMP_OUTPUT:-$build/benchmarks/gpu-options-${SLURM_JOB_ID}}
export SVMP_PIPE_CASE=${SVMP_PIPE_CASE:-$repo/build/benchmarks/large-pipe/case}
[[ -x $SVMP_EXE ]] || { echo "ERROR: missing $SVMP_EXE" >&2; exit 1; }
[[ ! -e $output ]] || { echo "ERROR: $output exists" >&2; exit 1; }
if [[ " ${SVMP_PARTS:-fluid physics} " == *" fluid "* ]]; then
  "$repo/benchmarks/large_pipe/prepare_case.sh" "$SVMP_PIPE_CASE" "${SVMP_PYTHON:-python3}"
fi

config=${output}-config
mkdir -p "$config"
(cd "$repo" && git rev-parse HEAD && git status --short) > "$config/source.txt" 2>&1 || true
(cd "$repo" && git diff) > "$config/source.diff" 2>&1 || true
sha256sum "$SVMP_EXE" > "$config/executable.sha256"
ldd "$SVMP_EXE" > "$config/ldd.txt" 2>&1 || true
module list > "$config/modules.txt" 2>&1 || true
nvidia-smi -q > "$config/nvidia-smi.txt" 2>&1 || true
scontrol show job "$SLURM_JOB_ID" > "$config/slurm-job.txt"
for cache in CMakeCache.txt svMultiPhysics-build/CMakeCache.txt; do
  if [[ -f "$build/$cache" ]]; then cp "$build/$cache" "$config/${cache//\//_}"; fi
done
srun --nodes=1 --exact --ntasks=1 --cpus-per-task=1 --cpu-bind=cores \
  --gpus-per-node=1 --gpus-per-task=1 --gpu-bind=single:1 \
  python3 "$repo/benchmarks/gpu_scaling/check_binding.py" "$config/binding" > "$config/binding.log" 2>&1
python3 "$repo/benchmarks/gpu_scaling/check_binding.py" "$config/binding" --validate 1

# Record which Amesos2 coarse solvers this Trilinos provides (Tacho/cuSOLVER
# variants fail fast without them and are listed in failed.txt).
grep -hE 'define HAVE_AMESOS2_(KLU2|SHYLU_NODETACHO|CUSOLVER)' \
  $(grep -E '^Trilinos_DIR' "$build/svMultiPhysics-build/CMakeCache.txt" 2>/dev/null | sed 's/^[^=]*=//; s#/lib6\{0,1\}4\{0,1\}/cmake/Trilinos$##')/include/Amesos2_config.h \
  > "$config/amesos2.txt" 2>&1 || true

"$repo/benchmarks/gpu_options/run_options.sh" slurm "$output"
