#!/bin/bash
#SBATCH --job-name=svmp-pipe-cpu
#SBATCH --output=svmp-pipe-cpu-%j.log
#SBATCH --error=svmp-pipe-cpu-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# CPU-node reference for the large-pipe and iliac GPU runs. Requires a
# CPU-only Trilinos build (Kokkos Serial/OpenMP): a CUDA-built executable
# runs Tpetra on the GPU and is not a CPU measurement.
#   SVMP_CPU_EXE=/path/to/cpu/svmultiphysics sbatch [--partition=...] \
#     [--constraint=CPU_SKU:7543P] benchmarks/large_pipe/sherlock_cpu.sbatch
# Use --constraint=CPU_SKU:7543P to match the A40 nodes' host CPU.
# Overrides: SVMP_RANKS ("8 16 32"), SVMP_OUTPUT, SVMP_PIPE_CASE, SVMP_PYTHON,
# SVMP_ILIAC (1). The same script works on Stampede3 with its partition names
# and a larger SVMP_RANKS / --ntasks.

set -euo pipefail
repo=${SVMP_REPO:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}
exe=${SVMP_CPU_EXE:?Set SVMP_CPU_EXE to a CPU-only svmultiphysics build}
output=${SVMP_OUTPUT:-$repo/build/benchmarks/large-pipe-cpu-${SLURM_JOB_ID}}
case_dir=${SVMP_PIPE_CASE:-$repo/build/benchmarks/large-pipe/case}
ranks=${SVMP_RANKS:-"8 16 32"}
[[ -x "$exe" && ! -e "$output" ]] || { echo "ERROR: bad exe or existing output" >&2; exit 1; }
"$repo/benchmarks/large_pipe/prepare_case.sh" "$case_dir" "${SVMP_PYTHON:-python3}"

mkdir -p "$output/config" "$output/pipe" "$output/iliac"
git -C "$repo" rev-parse HEAD > "$output/config/commit.txt"
sha256sum "$exe" "$case_dir/mesh/mesh-complete.mesh.vtu" > "$output/config/sha256.txt"
lscpu > "$output/config/lscpu.txt"
scontrol show job "$SLURM_JOB_ID" > "$output/config/slurm-job.txt"
module list > "$output/config/modules.txt" 2>&1 || true
ldd "$exe" > "$output/config/ldd.txt" 2>&1 || true

finish() {
  local status=$?
  trap - EXIT
  for group in pipe iliac; do
    ref=$(ls "$output/$group" 2>/dev/null | sort -t- -k2 -n | tail -1)
    [[ -n "$ref" ]] && python3 "$repo/benchmarks/solver_optimization/summarize.py" \
      "$output/$group" --reference-group "$ref" || status=$?
  done
  python3 "$repo/benchmarks/large_pipe/report.py" "$output"/pipe "$output"/iliac \
    > "$output/report.md" || true
  exit "$status"
}
trap finish EXIT

common=(python3 "$repo/benchmarks/schur_amg/run_benchmark.py" --exe "$exe"
  --launcher srun --srun-exact --cpu-bind cores --threads 1
  --preconditioner schur-amg --gmres-preconditioner resistance
  --assembly kokkos --stats --assembly-stats)
for r in $ranks; do
  "${common[@]}" --case "$case_dir" --steps 10 --ranks "$r" --output "$output/pipe/cpu-$r"
  if [[ ${SVMP_ILIAC:-1} == 1 ]]; then
    "${common[@]}" --steps 20 --ranks "$r" --output "$output/iliac/cpu-$r"
  fi
done
echo "CPU results: $output"
