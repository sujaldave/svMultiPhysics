#!/bin/bash
#SBATCH --job-name=svmp-amg
#SBATCH --output=svmp-amg-%j.log
#SBATCH --error=svmp-amg-%j.err
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=4
#SBATCH --constraint=GPU_SKU:A40&CPU_SKU:7543P
#SBATCH --exclusive
#SBATCH --mem=48G
#SBATCH --time=02:00:00

set -euo pipefail
repo=${SVMP_REPO:-${SLURM_SUBMIT_DIR:?Submit from repository root}}
exe=${SVMP_EXE:-$repo/build/svMultiPhysics-build/bin/svmultiphysics}
output=${SVMP_OUTPUT:-$repo/build/benchmarks/amg-application-${SLURM_JOB_ID}}
[[ -x "$exe" && ! -e "$output" ]]
mkdir -p "$output/config"
cp "$repo/benchmarks/amg_application/sherlock_a40.sbatch" "$output/config/"
for cache in "$repo/build/CMakeCache.txt" "$repo/build/svMultiPhysics-build/CMakeCache.txt"; do
  if [[ -f "$cache" ]]; then cp "$cache" "$output/config/$(basename "$(dirname "$cache")")-CMakeCache.txt"; fi
done
git -C "$repo" diff > "$output/config/source.diff"
git -C "$repo" rev-parse HEAD > "$output/config/commit.txt"
sha256sum "$exe" > "$output/config/executable.sha256"
nvidia-smi topo -m > "$output/config/topology.txt"
nvidia-smi -q > "$output/config/nvidia-smi.txt"
ompi_info --all > "$output/config/mpi.txt" 2>&1 || true
module list > "$output/config/modules.txt" 2>&1 || true
read -r -a counts <<< "${SVMP_GPU_COUNTS:-1 2 4}"
for count in "${counts[@]}"; do
  [[ "$count" == 1 || "$count" == 2 || "$count" == 4 ]]
  binding=$output/binding-$count
  srun --exact --nodes=1 --ntasks="$count" --cpus-per-task=1 --cpu-bind=verbose,cores \
    --gpus-per-node="$count" --gpus-per-task=1 --gpu-bind=single:1 \
    python3 "$repo/benchmarks/gpu_scaling/check_binding.py" "$binding" \
    > "$output/config/binding-$count.log" 2>&1
  python3 "$repo/benchmarks/gpu_scaling/check_binding.py" "$binding" --validate "$count"
  common=(python3 "$repo/benchmarks/schur_amg/run_benchmark.py" --exe "$exe"
    --launcher srun --srun-exact --cpu-bind cores --gpus-per-node "$count"
    --gpus-per-task 1 --ranks "$count" --threads 1 --assembly kokkos
    --preconditioner schur-amg --gmres-preconditioner resistance --stats --assembly-stats)
  for mode in klu-auto klu-staged klu-device polynomial-auto; do
    unset TPETRA_ASSUME_GPU_AWARE_MPI
    export SV_TRILINOS_SCHUR_COARSE=klu
    case "$mode" in
      klu-staged) export TPETRA_ASSUME_GPU_AWARE_MPI=0 ;;
      klu-device) export TPETRA_ASSUME_GPU_AWARE_MPI=1 ;;
      polynomial-auto) export SV_TRILINOS_SCHUR_COARSE=chebyshev ;;
    esac
    "${common[@]}" --steps "${SVMP_STEPS:-5}" --repeat "${SVMP_REPEAT:-2}" \
      --output "$output/gpus-$count/$mode"
  done
  python3 "$repo/benchmarks/solver_optimization/summarize.py" \
    "$output/gpus-$count" --reference-group klu-auto --expect-cuda
  if [[ ${SVMP_PROFILE:-0} == 1 && "$count" != 1 ]]; then
    unset TPETRA_ASSUME_GPU_AWARE_MPI
    export SV_TRILINOS_SCHUR_COARSE=klu
    "${common[@]}" --steps 2 --runtime-details --profile nsys --profile-mpi \
      --output "$output/profile-$count"
  fi
done
