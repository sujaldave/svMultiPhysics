#!/bin/bash
# Start one MPI rank on a Marlowe DGX H100 node, then exec the given command.
#
#   mpirun ... marlowe_rank.sh <command> [args...]
#   marlowe_rank.sh --write-map DIR      (once per node: GPU -> NUMA/CPU table)
#
# Each rank sees exactly one GPU: entry OMPI_COMM_WORLD_LOCAL_RANK of
# SVMP_NODE_GPUS (default: the job's CUDA_VISIBLE_DEVICES). The solver
# initializes Kokkos without choosing a device, so this selection is required.
#
# SVMP_BIND selects how the rank's CPUs are chosen:
#   core  mpirun binds the rank to one core (--bind-to core); nothing to do here
#   none  no binding (--bind-to none); the rank may run on any core of the node
#   numa  --bind-to none, then pin the rank to the CPUs local to its GPU's
#         NUMA domain (from sysfs, via the table written by --write-map)
set -euo pipefail

gpu_table() {
  # index, PCI bus ID (sysfs form), NUMA node, NUMA-local CPU list
  nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader |
    while IFS=', ' read -r index bus; do
      bus=$(tr '[:upper:]' '[:lower:]' <<< "${bus: -12}")
      dev=/sys/bus/pci/devices/$bus
      printf '%s\t%s\t%s\t%s\n' "$index" "$bus" \
        "$(cat "$dev/numa_node" 2>/dev/null || echo -1)" \
        "$(cat "$dev/local_cpulist" 2>/dev/null || echo unknown)"
    done
}

if [[ ${1:-} == --write-map ]]; then
  mkdir -p "${2:?usage: marlowe_rank.sh --write-map DIR}"
  gpu_table > "$2/$(hostname -s).tsv"
  exit 0
fi

local_rank=${OMPI_COMM_WORLD_LOCAL_RANK:?marlowe_rank.sh must be started by Open MPI mpirun}
IFS=, read -r -a gpus <<< "${SVMP_NODE_GPUS:-${CUDA_VISIBLE_DEVICES:-}}"
gpu=${gpus[$local_rank]:-}
if [[ -z $gpu ]]; then
  echo "marlowe_rank.sh: no GPU for local rank $local_rank (GPUs: ${SVMP_NODE_GPUS:-${CUDA_VISIBLE_DEVICES:-none}})" >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES=$gpu

if [[ ${SVMP_BIND:-core} == numa ]]; then
  table=${SVMP_GPU_MAP_DIR:+$SVMP_GPU_MAP_DIR/$(hostname -s).tsv}
  if [[ -n $table && -f $table ]]; then
    cpus=$(awk -F'\t' -v g="$gpu" '$1 == g {print $4}' "$table")
  else
    cpus=$(gpu_table | awk -F'\t' -v g="$gpu" '$1 == g {print $4}')
  fi
  if [[ -z $cpus || $cpus == unknown ]]; then
    echo "marlowe_rank.sh: NUMA-local CPUs of GPU $gpu are unknown on $(hostname -s)" >&2
    exit 1
  fi
  exec taskset -c "$cpus" "$@"
fi
exec "$@"
