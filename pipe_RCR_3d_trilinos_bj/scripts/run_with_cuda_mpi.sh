#!/usr/bin/env bash
# run_with_cuda_mpi.sh
# Simple launcher to run svmultiphysics with Kokkos/CUDA per-rank device assignment.
# Usage:
#  ./scripts/run_with_cuda_mpi.sh -n <np> -- /full/path/to/solver.xml [mpirun-args]
# Examples:
#  ./scripts/run_with_cuda_mpi.sh -n 2 -- /home/sujal/research/testcases/pipe_RCR_3d_trilinos_bj/solver.xml
#  ./scripts/run_with_cuda_mpi.sh -n 4 -- /path/to/solver.xml --bind-to none

set -euo pipefail

# Default mpirun (uses mpirun from PATH). You can override via MPI_RUN env var.
MPI_RUN=${MPI_RUN:-mpirun}
BINARY="/home/sujal/research/svGPU/svMultiPhysics/build-debugForGPUCompilation/svMultiPhysics-build/bin/svmultiphysics"

usage() {
  cat <<EOF
Usage: $0 -n <np> -- <solver.xml> [extra mpirun args]

This launcher sets per-process KOKKOS_DEVICE_ID and CUDA_VISIBLE_DEVICES using
OpenMPI-provided local-rank environment variables. It assumes OpenMPI sets
OMPI_COMM_WORLD_LOCAL_RANK and OMPI_COMM_WORLD_RANK for each process.

Notes:
 - If you use another MPI (MVAPICH2, Intel MPI), the local-rank variable name may differ.
 - The wrapper runs the binary via a per-rank shell that sets:
     KOKKOS_DEVICE_ID=$OMPI_COMM_WORLD_LOCAL_RANK
     CUDA_VISIBLE_DEVICES=$KOKKOS_DEVICE_ID
     KOKKOS_NUM_DEVICES=1
 - If you prefer a different mapping (e.g., multiple ranks per GPU), update the script.

EOF
}

if [[ "$#" -lt 3 ]]; then
  usage
  exit 1
fi

NP=1
# parse -n <np>
while getopts ":n:h" opt; do
  case $opt in
    n) NP="$OPTARG" ;;
    h) usage; exit 0 ;;
    \?) echo "Invalid option: -$OPTARG" >&2; usage; exit 1 ;;
  esac
done
shift $((OPTIND-1))

# Expect -- before solver.xml
if [[ "$1" != "--" ]]; then
  echo "Expected -- before solver xml path" >&2
  usage
  exit 1
fi
shift

SOLVER_XML="$1"
shift || true
MPI_EXTRA_ARGS=("$@")

# Show available GPUs (if nvidia-smi present)
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "Detected GPUs:"
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
else
  echo "nvidia-smi not found in PATH; cannot list GPUs"
fi

echo "Launching: ${MPI_RUN} -n ${NP} ${MPI_EXTRA_ARGS[*]} ${BINARY} ${SOLVER_XML}"

# The actual per-rank shell uses OpenMPI-provided env vars (OMPI_COMM_WORLD_LOCAL_RANK).
# Using mpirun to execute /bin/bash -lc so each rank will evaluate the OMPI env vars
# and set Kokkos/CUDA envs accordingly.

${MPI_RUN} -n ${NP} "bash" -lc $'\
  # per-rank starter (expanded on the remote rank):\n  echo "[rank $OMPI_COMM_WORLD_RANK local $OMPI_COMM_WORLD_LOCAL_RANK] starting...";\n  export KOKKOS_DEVICE_ID=$OMPI_COMM_WORLD_LOCAL_RANK;\n  export CUDA_VISIBLE_DEVICES=$KOKKOS_DEVICE_ID;\n  export KOKKOS_NUM_DEVICES=1;\n  echo "[rank $OMPI_COMM_WORLD_RANK] MAPPING -> KOKKOS_DEVICE_ID=$KOKKOS_DEVICE_ID CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES";\n  exec "${BINARY}" "${SOLVER_XML}"\n' "${MPI_EXTRA_ARGS[@]}"

exit_code=$?
if [[ $exit_code -ne 0 ]]; then
  echo "svmultiphysics exited with code $exit_code"
fi
exit $exit_code
