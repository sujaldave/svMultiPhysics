#!/bin/bash
#SBATCH --job-name=svmp-scale
#SBATCH --output=svmp-scale-%j.log
#SBATCH --error=svmp-scale-%j.err
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=4
#SBATCH --constraint=GPU_SKU:A40&CPU_SKU:7543P
#SBATCH --exclusive
#SBATCH --mem=48G
#SBATCH --time=00:30:00

set -euo pipefail
# Submit from the repository root with the successful GPU build's modules.
repo=${SVMP_REPO:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}
build=${SVMP_BUILD_DIR:-$repo/build}
exe=${SVMP_EXE:-$build/svMultiPhysics-build/bin/svmultiphysics}
output=${SVMP_OUTPUT:-$repo/build/benchmarks/scaling-${SLURM_JOB_ID}}
steps=${SVMP_STEPS:-20}
repeat=${SVMP_REPEAT:-3}
# Default: a matched 1/2/4-GPU sweep within this single allocation.
read -r -a counts <<< "${SVMP_GPU_COUNTS:-1 2 4}"
[[ -x "$exe" && -f "$repo/benchmarks/gpu_scaling/summarize_scaling.py" ]]
[[ ! -e "$output" ]] || { echo "Output already exists: $output" >&2; exit 1; }
for count in "${counts[@]}"; do
  [[ "$count" == 1 || "$count" == 2 || "$count" == 4 ]] || exit 1
done
mkdir -p "$output/config"
python3 - "$output" "$steps" "$repeat" "${counts[@]}" <<'PY'
import json, sys
from pathlib import Path
steps, repeat = int(sys.argv[2]), int(sys.argv[3])
counts = [int(value) for value in sys.argv[4:]]
assert steps > 0 and repeat > 0 and counts and len(set(counts)) == len(counts)
Path(sys.argv[1], 'sweep.json').write_text(json.dumps(dict(
    steps=steps, repeat=repeat, gpu_counts=counts), indent=2)+'\n')
PY
python3 - "$output" <<'PY'
import json, os, sys
from pathlib import Path
keys = ['CUDA_VISIBLE_DEVICES','CUDA_LAUNCH_BLOCKING','TPETRA_ASSUME_GPU_AWARE_MPI',
        'OMPI_MCA_opal_cuda_support','UCX_TLS','UCX_MEMTYPE_CACHE','OMP_NUM_THREADS',
        'KOKKOS_NUM_THREADS']
Path(sys.argv[1],'config','selected-environment.json').write_text(
    json.dumps({key:os.environ.get(key) for key in keys}, indent=2)+'\n')
PY
git -C "$repo" rev-parse HEAD > "$output/config/commit.txt"
git -C "$repo" status --short > "$output/config/git-status.txt"
git -C "$repo" diff > "$output/config/source.diff"
sha256sum "$exe" > "$output/config/executable.sha256"
uname -a > "$output/config/uname.txt"
scontrol show job "$SLURM_JOB_ID" > "$output/config/slurm-job.txt"
nvidia-smi -q > "$output/config/nvidia-smi-before.txt"
nvidia-smi topo -m > "$output/config/gpu-topology.txt"
module list > "$output/config/modules.txt" 2>&1 || true
ldd "$exe" > "$output/config/ldd.txt" 2>&1 || true
nvcc --version > "$output/config/nvcc-version.txt" 2>&1 || true
ompi_info --parsable > "$output/config/ompi-info.txt" 2>&1 || true
for cache in CMakeCache.txt svMultiPhysics-build/CMakeCache.txt; do
  if [[ -f "$build/$cache" ]]; then cp "$build/$cache" "$output/config/${cache//\//_}"; fi
done
if [[ -n ${SVMP_TRILINOS_BUILD:-} ]]; then
  cp "$SVMP_TRILINOS_BUILD/CMakeCache.txt" "$output/config/trilinos-CMakeCache.txt"
fi

finish() {
  local run_status=$?
  local summary_status=0
  trap - EXIT
  python3 "$repo/benchmarks/gpu_scaling/summarize_scaling.py" "$output" --expect-cuda || summary_status=$?
  if (( run_status != 0 )); then exit "$run_status"; fi
  exit "$summary_status"
}
trap finish EXIT

for count in "${counts[@]}"; do
  # CUDA_VISIBLE_DEVICES is assigned by Slurm. Do not override it manually.
  binding=$output/binding/gpus-$count
  srun --exact --nodes=1 --ntasks="$count" --cpus-per-task=1 --cpu-bind=cores \
    --gpus-per-node="$count" --gpus-per-task=1 --gpu-bind=single:1 \
    python3 "$repo/benchmarks/gpu_scaling/check_binding.py" "$binding" \
    > "$output/config/binding-${count}.log" 2>&1
  python3 "$repo/benchmarks/gpu_scaling/check_binding.py" "$binding" --validate "$count"

  benchmark=(python3 "$repo/benchmarks/schur_amg/run_benchmark.py"
    --exe "$exe" --launcher srun --srun-exact --cpu-bind cores
    --gpus-per-node "$count" --gpus-per-task 1 --ranks "$count" --threads 1
    --preconditioner schur-amg --assembly kokkos
    --schur-reuse transfers --schur-operator explicit --stats --assembly-stats)
  # Separate two-step diagnostics warm the filesystem and verify this rank count
  # before measuring; all timed runs still use independent solver processes.
  "${benchmark[@]}" --steps 2 --runtime-details \
    --output "$output/warmup/gpus-$count"
  "${benchmark[@]}" --steps "$steps" --repeat "$repeat" \
    --output "$output/gpus-$count"
  nvidia-smi -q > "$output/config/nvidia-smi-after-${count}.txt"
done
echo "Scaling results and binding/configuration records: $output"
