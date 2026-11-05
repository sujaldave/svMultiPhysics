#!/bin/bash
#SBATCH --job-name=svmp-h100-scale
#SBATCH --output=svmp-h100-scale-%j.log
#SBATCH --error=svmp-h100-scale-%j.err
#SBATCH --partition=preempt
#SBATCH --account=marlowe-m000218
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append

# Strong scaling of the GPU BIPN solver (Kokkos assembly + Schur AMG) on
# Marlowe DGX H100 nodes: the Marlowe counterpart of sherlock_a40_scaling.sbatch.
# One MPI rank per GPU, launched with HPC-X mpirun (as in all earlier Marlowe runs).
#
#   1 node  (default):  1, 2, 4, 8 GPUs   sbatch benchmarks/gpu_scaling/marlowe_h100_scaling.sbatch
#   2 nodes:            16 GPUs          sbatch --nodes=2 benchmarks/gpu_scaling/marlowe_h100_scaling.sbatch
#   (benchmarks/gpu_scaling/submit_marlowe.sh submits both, one job per case)
#
# Each GPU count is run for every combination of
#   binding (SVMP_BIND_MODES, default "core numa none"):
#     core  --map-by ppr:N:node:PE=1 --bind-to core   (the earlier Marlowe recipe)
#     numa  --bind-to none, each rank pinned to its GPU's NUMA-local CPUs
#     none  --bind-to none, no pinning at all
#   communication (SVMP_COMM_MODES, default "ucx-best gpuaware-ipc"; GPU counts > 1):
#     ucx-best      fastest settings of the August Marlowe UCX sweeps:
#                   1 node  UCX_TLS=self,sm,cuda_copy     cache off, GDR off
#                   2 nodes UCX_TLS=rc,sm,self,cuda_copy  cache off, GDR off
#                   TPETRA_ASSUME_GPU_AWARE_MPI left unset, as in those sweeps
#     gpuaware-ipc  TPETRA_ASSUME_GPU_AWARE_MPI=1 and cuda_ipc added to UCX_TLS,
#                   so halo exchanges between GPUs of one node can use NVLink
#     also available: ucx-cache-on (2-node candidate), host-staged (Tpetra=0),
#                   gpuaware (Tpetra=1 without cuda_ipc), ucx-default (reference)
# Cases (SVMP_CASES, default "iliac pipe"): the iliac artery (20 steps, as the
# A40 scale suite) and Lasso's 1M-tet pipe (10 steps, as the A40 pipe suite).
#
# Overrides: SVMP_REPO, SVMP_BUILD_DIR, SVMP_EXE, SVMP_ENV (svmp_marlowe_env.sh
# from buildLibrariesAndSVMP.sh), SVMP_OUTPUT, SVMP_GPU_COUNTS, SVMP_CASES,
# SVMP_BIND_MODES, SVMP_COMM_MODES, SVMP_STEPS (iliac, 20), SVMP_PIPE_STEPS (10),
# SVMP_PIPE_CASE, SVMP_PYTHON (NumPy, only to generate the pipe), SVMP_REPEAT (3),
# SVMP_RUN_TIMEOUT (900 s per run), SVMP_SCHUR_COARSE (e.g. chebyshev),
# SVMP_MPIRUN_EXTRA (e.g. "--mca coll ^ucc,hcoll"), SVMP_RETRY_FAILED=1.
#
# Preemption: the job requeues with the same job ID and output directory and
# skips every finished configuration. Failed or timed-out configurations are
# listed in failed.txt and skipped on requeue unless SVMP_RETRY_FAILED=1.
set -euo pipefail

repo=${SVMP_REPO:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}
build=${SVMP_BUILD_DIR:-$repo/build}
exe=${SVMP_EXE:-$build/svMultiPhysics-build/bin/svmultiphysics}
nodes=${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}
output=${SVMP_OUTPUT:-$build/benchmarks/marlowe-scale-${SLURM_JOB_ID}}
tools=$repo/benchmarks/gpu_scaling
wrapper=$tools/marlowe_rank.sh

# ---- Environment: HPC-X Open MPI (CUDA-aware) and the libraries of the build ----
set +u   # module and environment scripts may read unset variables
if [[ -n ${SVMP_ENV:-} ]]; then
  source "$SVMP_ENV"
else
  module purge
  module load slurm gcc/13.1.0 cmake/3.30.3 nvhpc/24.7
  hpcx=${SVMP_HPCX_HOME:-/cm/shared/apps/nvhpc/24.7/Linux_x86_64/24.7/comm_libs/12.5/hpcx/hpcx-2.19}
  export MPI_HOME=$hpcx/ompi
  export PATH=$MPI_HOME/bin:$PATH
  export LD_LIBRARY_PATH=$MPI_HOME/lib:${LD_LIBRARY_PATH:-}
fi
set -u
[[ -x $exe ]] || { echo "ERROR: missing executable $exe" >&2; exit 1; }
[[ -x $wrapper ]] || { echo "ERROR: $wrapper is not executable (chmod +x)" >&2; exit 1; }
if ldd "$exe" | grep 'not found' > /dev/null; then   # no -q: pipefail + SIGPIPE
  echo "ERROR: unresolved libraries for $exe; set SVMP_ENV to the build's svmp_marlowe_env.sh:" >&2
  ldd "$exe" | grep 'not found' >&2
  exit 1
fi
mpirun --version 2>&1 | grep 'Open MPI' > /dev/null || { echo "ERROR: mpirun is not Open MPI/HPC-X" >&2; exit 1; }

# The binary must target Hopper; an Ampere (A40) build would not run here.
trilinos_prefix=$(grep -E '^Trilinos_DIR[:=]' "$build/svMultiPhysics-build/CMakeCache.txt" 2>/dev/null |
  sed 's/^[^=]*=//; s#/lib6\{0,1\}4\{0,1\}/cmake/Trilinos$##' || true)
kokkos_arch=$(cat "$trilinos_prefix"/include/KokkosCore_config.h "$trilinos_prefix"/include/kokkos/KokkosCore_config.h 2>/dev/null |
  grep -oE 'KOKKOS_ARCH_(HOPPER|AMPERE|VOLTA|ADA)[0-9]+' | sort -u | tr '\n' ' ' || true)
if [[ -n $kokkos_arch && $kokkos_arch != *HOPPER90* && ${SVMP_ALLOW_ARCH_MISMATCH:-0} != 1 ]]; then
  echo "ERROR: Trilinos/Kokkos at $trilinos_prefix targets '$kokkos_arch', not KOKKOS_ARCH_HOPPER90" >&2
  exit 1
fi

# ---- Sweep definition ----
IFS=, read -r -a node_gpus <<< "${CUDA_VISIBLE_DEVICES:?Slurm did not assign GPUs}"
gpus_per_node=${#node_gpus[@]}
export SVMP_NODE_GPUS=$CUDA_VISIBLE_DEVICES
if (( nodes == 1 )); then default_counts="1 2 4 8"; else default_counts=$((nodes * gpus_per_node)); fi
read -r -a counts <<< "${SVMP_GPU_COUNTS:-$default_counts}"
read -r -a cases <<< "${SVMP_CASES:-iliac pipe}"
read -r -a binds <<< "${SVMP_BIND_MODES:-core numa none}"
read -r -a comms <<< "${SVMP_COMM_MODES:-ucx-best gpuaware-ipc}"
iliac_steps=${SVMP_STEPS:-20}
pipe_steps=${SVMP_PIPE_STEPS:-10}
repeat=${SVMP_REPEAT:-3}
run_timeout=${SVMP_RUN_TIMEOUT:-900}
pipe_case=${SVMP_PIPE_CASE:-$repo/build/benchmarks/large-pipe/case}
for count in "${counts[@]}"; do
  if (( count % nodes || count / nodes > gpus_per_node || count < nodes )); then
    echo "ERROR: $count GPUs cannot be spread evenly over $nodes node(s) with $gpus_per_node GPUs each" >&2
    exit 1
  fi
done
for bind in "${binds[@]}"; do
  [[ $bind =~ ^(core|numa|none)$ ]] || { echo "ERROR: unknown binding mode $bind" >&2; exit 1; }
done
for comm in "${comms[@]}"; do
  [[ $comm =~ ^(ucx-best|ucx-cache-on|host-staged|gpuaware|gpuaware-ipc|ucx-default)$ ]] ||
    { echo "ERROR: unknown communication mode $comm" >&2; exit 1; }
done
for case_name in "${cases[@]}"; do
  [[ $case_name =~ ^(iliac|pipe)$ ]] || { echo "ERROR: unknown case $case_name" >&2; exit 1; }
done
if [[ " ${cases[*]} " == *" pipe "* ]]; then
  "$repo/benchmarks/large_pipe/prepare_case.sh" "$pipe_case" "${SVMP_PYTHON:-python3}"
fi

mkdir -p "$output/config"
exe_sha=$(sha256sum "$exe" | cut -d' ' -f1)
manifest=$(python3 - "$exe_sha" "$nodes" "$repeat" "$iliac_steps" "$pipe_steps" "$run_timeout" \
  "${counts[*]}" "${cases[*]}" "${binds[*]}" "${comms[*]}" "${SVMP_SCHUR_COARSE:-klu}" "${SVMP_MPIRUN_EXTRA:-}" <<'PY'
import json, sys
a = sys.argv[1:]
print(json.dumps(dict(platform='marlowe-h100', executable_sha256=a[0], nodes=int(a[1]),
    repeat=int(a[2]), steps=dict(iliac=int(a[3]), pipe=int(a[4])), run_timeout_s=float(a[5]),
    gpu_counts=[int(x) for x in a[6].split()], cases=a[7].split(), bind_modes=a[8].split(),
    comm_modes=a[9].split(), schur_coarse=a[10], mpirun_extra=a[11]), indent=2))
PY
)
if [[ -f $output/sweep.json ]]; then
  [[ $manifest == "$(cat "$output/sweep.json")" ]] ||
    { echo "ERROR: $output was started with different settings; choose a new SVMP_OUTPUT" >&2; exit 1; }
  echo "Resuming $output (restart ${SLURM_RESTART_COUNT:-0})"
else
  echo "$manifest" > "$output/sweep.json"
fi

# ---- Records: source, build, node, GPU/NIC topology, MPI/UCX ----
attempt=$output/config/attempt-${SLURM_JOB_ID}-${SLURM_RESTART_COUNT:-0}
mkdir -p "$attempt"
(cd "$repo" && git rev-parse HEAD) > "$attempt/commit.txt" 2>&1 || true
(cd "$repo" && git status --short) > "$attempt/git-status.txt" 2>&1 || true
(cd "$repo" && git diff) > "$attempt/source.diff" 2>&1 || true
sha256sum "$exe" > "$attempt/executable.sha256"
uname -a > "$attempt/uname.txt"
scontrol show job "$SLURM_JOB_ID" > "$attempt/slurm-job.txt" 2>&1 || true
scontrol show hostnames "${SLURM_JOB_NODELIST:-}" > "$attempt/hosts.txt" 2>&1 || true
lscpu > "$attempt/lscpu.txt" 2>&1 || true
numactl -H > "$attempt/numa.txt" 2>&1 || true
nvidia-smi -q > "$attempt/nvidia-smi-before.txt" 2>&1 || true
nvidia-smi topo -m > "$attempt/gpu-topology.txt" 2>&1 || true
nvidia-smi nvlink -s > "$attempt/nvlink-status.txt" 2>&1 || true
(ibdev2netdev || ibv_devinfo -l) > "$attempt/ib-devices.txt" 2>&1 || true
module list > "$attempt/modules.txt" 2>&1 || true
ldd "$exe" > "$attempt/ldd.txt" 2>&1 || true
mpirun --version > "$attempt/mpirun-version.txt" 2>&1 || true
ompi_info --parsable --all > "$attempt/ompi-info.txt" 2>&1 || true
ucx_info -v > "$attempt/ucx-version.txt" 2>&1 || true
ucx_info -d > "$attempt/ucx-devices.txt" 2>&1 || true
env | grep -E '^(SLURM_|CUDA_|UCX_|OMPI_|TPETRA_|SVMP_|HPCX)' | sort > "$attempt/environment.txt" || true
{
  echo "trilinos_prefix=$trilinos_prefix"
  echo "kokkos_arch=$kokkos_arch"
  grep -hE 'set\(Trilinos_VERSION ' "$trilinos_prefix"/lib*/cmake/Trilinos/TrilinosConfig.cmake 2>/dev/null || true
  grep -hE 'define KOKKOS_VERSION|define KOKKOS_ENABLE_CUDA\b' "$trilinos_prefix"/include/KokkosCore_config.h \
    "$trilinos_prefix"/include/kokkos/KokkosCore_config.h 2>/dev/null || true
  grep -hE 'define KOKKOSKERNELS_ENABLE_TPL_(CUSPARSE|CUBLAS|CUSOLVER)' "$trilinos_prefix"/include/KokkosKernels_config.h 2>/dev/null ||
    echo "KokkosKernels cuSPARSE/cuBLAS/cuSOLVER TPLs: not enabled"
} > "$attempt/trilinos.txt" 2>&1
for cache in CMakeCache.txt svMultiPhysics-build/CMakeCache.txt; do
  if [[ -f $build/$cache ]]; then cp "$build/$cache" "$attempt/${cache//\//_}"; fi
done

# One GPU -> PCI bus -> NUMA node -> local CPUs table per node, read by marlowe_rank.sh.
export SVMP_GPU_MAP_DIR=$output/config/gpu-map
mpirun -np "$nodes" --map-by ppr:1:node --bind-to none "$wrapper" --write-map "$SVMP_GPU_MAP_DIR"

# ---- Common runtime settings ----
unset OMP_PROC_BIND OMP_PLACES CUDA_LAUNCH_BLOCKING
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export UCX_WARN_UNUSED_ENV_VARS=n
export OMPI_MCA_pml=ucx OMPI_MCA_osc=ucx OMPI_MCA_btl=^openib
if [[ -n ${SVMP_SCHUR_COARSE:-} ]]; then export SV_TRILINOS_SCHUR_COARSE=$SVMP_SCHUR_COARSE; fi

set_comm() {   # $1: communication mode
  unset UCX_TLS UCX_MEMTYPE_CACHE UCX_IB_GPU_DIRECT_RDMA TPETRA_ASSUME_GPU_AWARE_MPI
  local tls=self,sm,cuda_copy
  (( nodes > 1 )) && tls=rc,sm,self,cuda_copy
  export SVMP_COMM=$1
  case $1 in
    ucx-best|single) export UCX_TLS=$tls UCX_MEMTYPE_CACHE=n ;;
    ucx-cache-on)    export UCX_TLS=$tls UCX_MEMTYPE_CACHE=y ;;
    host-staged)     export UCX_TLS=$tls UCX_MEMTYPE_CACHE=n TPETRA_ASSUME_GPU_AWARE_MPI=0 ;;
    gpuaware)        export UCX_TLS=$tls UCX_MEMTYPE_CACHE=n TPETRA_ASSUME_GPU_AWARE_MPI=1 ;;
    gpuaware-ipc)    export UCX_TLS=$tls,cuda_ipc UCX_MEMTYPE_CACHE=n TPETRA_ASSUME_GPU_AWARE_MPI=1 ;;
    ucx-default)     return ;;   # UCX chooses everything (4-7x slower in the August sweeps)
  esac
  export UCX_IB_GPU_DIRECT_RDMA=n   # forced GDR was pathological (295 s vs 21 s)
}

bind_args() {   # $1: binding mode, $2: ranks per node
  case $1 in
    core) echo "--map-by ppr:$2:node:PE=1 --bind-to core" ;;
    numa|none) echo "--map-by ppr:$2:node --bind-to none" ;;
  esac
}

# Skip finished configurations; set aside partial ones from a preempted attempt.
should_run() {   # $1: result directory
  if [[ -f $1/.complete ]]; then echo "Skipping finished: $1"; return 1; fi
  if [[ -f $1/.failed && ${SVMP_RETRY_FAILED:-0} != 1 ]]; then echo "Skipping failed: $1"; return 1; fi
  if [[ -e $1 ]]; then mv "$1" "$1.incomplete-${SLURM_JOB_ID}-${SLURM_RESTART_COUNT:-0}-$(date +%s)"; fi
  return 0
}

run_config() {   # $1: result directory; remaining: command
  local dir=$1; shift
  if "$@"; then
    touch "$dir/.complete"
  else
    mkdir -p "$dir"; touch "$dir/.failed"
    echo "$dir" >> "$output/failed.txt"
    echo "WARNING: failed: $dir (continuing with the next configuration)" >&2
  fi
}

finish() {
  local status=$?
  trap - EXIT
  python3 "$tools/summarize_marlowe.py" "$output" --expect-cuda || true
  [[ -s $output/failed.txt ]] && echo "Failed configurations: $output/failed.txt" >&2
  exit "$status"
}
trap finish EXIT

# ---- Binding probes: one per binding mode and GPU count, outside the timings ----
for bind in "${binds[@]}"; do
  for count in "${counts[@]}"; do
    probe=$output/binding/$bind/gpus-$count
    [[ -f $probe/.complete ]] && continue
    rm -rf "$probe"
    export SVMP_BIND=$bind
    set_comm ucx-best
    # shellcheck disable=SC2046
    mpirun -np "$count" $(bind_args "$bind" $((count / nodes))) --report-bindings ${SVMP_MPIRUN_EXTRA:-} \
      "$wrapper" python3 "$tools/check_binding.py" "$probe" > "$output/config/binding-$bind-$count.log" 2>&1
    numa_flag=; [[ $bind == numa ]] && numa_flag=--require-numa-local
    python3 "$tools/check_binding.py" "$probe" --validate "$count" --multi-node $numa_flag
    touch "$probe/.complete"
  done
done

# ---- Timed runs ----
for case_name in "${cases[@]}"; do
  if [[ $case_name == pipe ]]; then
    case_args=(--case "$pipe_case" --gmres-preconditioner resistance); steps=$pipe_steps
  else
    case_args=(--gmres-preconditioner diagonal); steps=$iliac_steps
  fi
  for count in "${counts[@]}"; do
    rpn=$((count / nodes))
    if (( count == 1 )); then case_comms=(single); else case_comms=("${comms[@]}"); fi
    common=(python3 "$repo/benchmarks/schur_amg/run_benchmark.py" --exe "$exe"
      --launcher mpirun --rank-wrapper "$wrapper" --ranks "$count" --threads 1
      --preconditioner schur-amg --assembly kokkos --schur-reuse transfers
      --schur-operator explicit --stats --assembly-stats --timeout "$run_timeout" "${case_args[@]}")
    # Two-step diagnostic run (Kokkos/AMG details): warms the filesystem and
    # checks this rank count before the timed repeats. Not a timing sample.
    warm=$output/$case_name/warmup/gpus-$count
    if should_run "$warm"; then
      export SVMP_BIND=${binds[0]}; set_comm "${case_comms[0]}"
      run_config "$warm" "${common[@]}" --mpirun-args "$(bind_args "${binds[0]}" "$rpn") ${SVMP_MPIRUN_EXTRA:-}" \
        --steps 2 --runtime-details --output "$warm"
    fi
    for comm in "${case_comms[@]}"; do
      for bind in "${binds[@]}"; do
        dir=$output/$case_name/${bind}__${comm}/gpus-$count
        should_run "$dir" || continue
        export SVMP_BIND=$bind; set_comm "$comm"
        echo "== $case_name: $count GPU(s) on $nodes node(s), bind=$bind, comm=$comm"
        run_config "$dir" "${common[@]}" --mpirun-args "$(bind_args "$bind" "$rpn") ${SVMP_MPIRUN_EXTRA:-}" \
          --steps "$steps" --repeat "$repeat" --output "$dir"
      done
    done
    nvidia-smi -q > "$attempt/nvidia-smi-after-$case_name-$count.txt" 2>&1 || true
  done
done
echo "Scaling results and binding/configuration records: $output"
