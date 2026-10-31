#!/bin/bash
# Rerun every earlier A40 suite (schur, pc, pipe, scale, physics) with a new
# svMultiPhysics build, e.g. Trilinos 17.2 with Kokkos_ARCH_AMPERE86. The schur
# and pipe jobs also run the old build in the same allocation, as a
# before/after control free of node-to-node variation.
#
# Run from the repository root on a Sherlock login node, with the modules that
# the new build needs loaded (sbatch copies this environment into every job):
#   benchmarks/trilinos_comparison/submit_all.sh
#
# Environment:
#   SVMP_NEW_BUILD    new build directory (default: ./build2)
#   SVMP_OLD_BUILD    old build directory (default: ./build)
#   SVMP_OUTPUT_ROOT  new results root (default: ./build2/benchmarks/rerun-DATE)
#   SVMP_SUITES       subset/order (default: "schur pc pipe scale physics")
#   SVMP_CHAIN=1      run the jobs one after another instead of concurrently
#   SVMP_PIPE_CASE    pipe case (default: /scratch/users/$USER/pipe-case if present)
#   SVMP_PYTHON       Python with NumPy, only needed to generate a missing pipe case
#   SVMP_DRY_RUN=1    print the sbatch commands without submitting
# After all jobs finish, copy SVMP_OUTPUT_ROOT back and run compare_builds.py.
set -euo pipefail

repo=$(cd "$(dirname "$0")/../.." && pwd)
cd "$repo"   # every suite uses SLURM_SUBMIT_DIR as the repository root
new_build=$(cd "${SVMP_NEW_BUILD:-$repo/build2}" && pwd)
old_build=$(cd "${SVMP_OLD_BUILD:-$repo/build}" && pwd)
new_exe=$new_build/svMultiPhysics-build/bin/svmultiphysics
old_exe=$old_build/svMultiPhysics-build/bin/svmultiphysics
root=${SVMP_OUTPUT_ROOT:-$new_build/benchmarks/rerun-$(date +%Y%m%d-%H%M%S)}
suites=${SVMP_SUITES:-"schur pc pipe scale physics"}
pipe_case=${SVMP_PIPE_CASE:-}
if [[ -z $pipe_case && -d /scratch/users/$USER/pipe-case ]]; then
  pipe_case=/scratch/users/$USER/pipe-case
fi

for exe in "$new_exe" "$old_exe"; do
  [[ -x $exe ]] || { echo "ERROR: missing executable $exe" >&2; exit 1; }
  if ldd "$exe" | grep 'not found' > /dev/null; then
    echo "ERROR: unresolved libraries for $exe in this environment:" >&2
    ldd "$exe" | grep 'not found' >&2
    echo "Load the modules this build needs, then resubmit." >&2
    exit 1
  fi
done
[[ $new_exe -ef $old_exe ]] && { echo "ERROR: old and new executables are the same file" >&2; exit 1; }
[[ ! -e $root ]] || { echo "ERROR: $root exists; choose a new SVMP_OUTPUT_ROOT" >&2; exit 1; }
mkdir -p "$root"

# Record which Trilinos/Kokkos each build links and which GPU arch Kokkos targets.
describe_build() {
  local build=$1 name=$2 cache prefix config
  cache=$build/svMultiPhysics-build/CMakeCache.txt
  {
    echo "build=$build"
    sha256sum "$build/svMultiPhysics-build/bin/svmultiphysics"
    grep -E '^(Trilinos_DIR|Kokkos_DIR|CMAKE_BUILD_TYPE|CMAKE_CXX_COMPILER)[:=]' "$cache" 2>/dev/null || true
    prefix=$(grep -E '^Trilinos_DIR[:=]' "$cache" 2>/dev/null | sed 's/^[^=]*=//; s#/lib6\{0,1\}4\{0,1\}/cmake/Trilinos$##' || true)
    for config in "$prefix"/include/KokkosCore_config.h "$prefix"/include/kokkos/KokkosCore_config.h; do
      [[ -f $config ]] && grep -E 'define KOKKOS_(ARCH_[A-Z0-9_]+|ENABLE_CUDA|VERSION)\b' "$config"
    done
    grep -hE 'set\(Trilinos_VERSION ' "$prefix"/lib*/cmake/Trilinos/TrilinosConfig.cmake 2>/dev/null || true
    echo "--- ldd (Trilinos/Kokkos/CUDA/MPI)"
    ldd "$build/svMultiPhysics-build/bin/svmultiphysics" | grep -iE 'tpetra|kokkoscore|muelu|belos|cudart|libmpi' || true
  } > "$root/build-$name.txt" 2>&1
}
describe_build "$new_build" new
describe_build "$old_build" old
module list > "$root/modules.txt" 2>&1 || true
(git rev-parse HEAD && git status --short) > "$root/source.txt" 2>&1 || true

# Each job starts from a clean SVMP_* environment plus the variables given here.
submit() {
  local name=$1 script=$2 time=$3; shift 3
  local cmd=(sbatch --parsable --time="$time")
  [[ ${SVMP_CHAIN:-0} == 1 && -n ${last_job:-} ]] && cmd+=(--dependency=afterany:$last_job)
  cmd+=("$script")
  local vars=(SVMP_BUILD_DIR="$new_build" ${SVMP_TRILINOS_BUILD:+SVMP_TRILINOS_BUILD="$SVMP_TRILINOS_BUILD"} "$@")
  echo "[$name] ${vars[*]} ${cmd[*]}" | tee -a "$root/submissions.txt"
  if [[ ${SVMP_DRY_RUN:-0} == 1 ]]; then return; fi
  local id
  id=$(env $(compgen -e | grep '^SVMP_' | sed 's/^/-u /') "${vars[@]}" "${cmd[@]}")
  id=${id%%;*}
  echo "$name $id" >> "$root/jobs.txt"
  last_job=$id
}

last_job=
for suite in $suites; do
  case $suite in
    schur)   submit schur benchmarks/solver_optimization/sherlock_a40.sbatch 00:45:00 \
               SVMP_OUTPUT="$root/schur" SVMP_BASELINE_EXE="$old_exe" SVMP_STEPS=20 SVMP_REPEAT=2 ;;
    pc)      submit pc benchmarks/solver_optimization/sherlock_preconditioners.sbatch 00:40:00 \
               SVMP_OUTPUT="$root/pc" SVMP_STEPS=20 SVMP_REPEAT=2 ;;
    pipe)    submit pipe benchmarks/large_pipe/sherlock_a40.sbatch 02:00:00 \
               SVMP_OUTPUT="$root/pipe" SVMP_BASELINE_EXE="$old_exe" \
               ${pipe_case:+SVMP_PIPE_CASE="$pipe_case"} ${SVMP_PYTHON:+SVMP_PYTHON="$SVMP_PYTHON"} ;;
    # The October 4-GPU group exceeded the 30-minute limit; allow two hours.
    scale)   submit scale benchmarks/gpu_scaling/sherlock_a40_scaling.sbatch 02:00:00 \
               SVMP_OUTPUT="$root/scale" SVMP_STEPS=20 SVMP_REPEAT=3 ;;
    physics) submit physics benchmarks/multiphysics/sherlock_a40.sbatch 04:00:00 \
               SVMP_OUTPUT="$root/physics" SVMP_STEPS=20 SVMP_REPEAT=2 SVMP_TIMEOUT=180 ;;
    *) echo "ERROR: unknown suite $suite" >&2; exit 1 ;;
  esac
done

cat <<EOF

Results root: $root
Build descriptions: $root/build-new.txt, $root/build-old.txt
After every job in $root/jobs.txt finishes, copy the root and the job logs
(svmp-*-JOBID.log/.err in $repo) back, then run locally:
  python3 benchmarks/trilinos_comparison/compare_builds.py /local/copy/$(basename "$root") \\
    --old-root /Users/sujaldave/Documents/ChatGPT/svmpWithGPU/oct27-tests
EOF
