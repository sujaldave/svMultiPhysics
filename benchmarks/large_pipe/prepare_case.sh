#!/bin/bash
# Generate Lasso's 1M-tetrahedra pipe case for run_benchmark.py --case.
# Usage: prepare_case.sh OUTPUT_DIR [PYTHON_WITH_NUMPY]
# OUTPUT_DIR receives solver.xml and mesh/; an existing complete case is kept.
set -euo pipefail

out=${1:?usage: prepare_case.sh OUTPUT_DIR [PYTHON_WITH_NUMPY]}
python=${2:-python3}
here=$(cd "$(dirname "$0")" && pwd)

if [[ -f "$out/mesh/mesh-complete.mesh.vtu" && -f "$out/mesh/mesh-surfaces/wall.vtp" ]]; then
  echo "Using existing pipe case: $out"
else
  "$python" -c 'import numpy' || { echo "ERROR: $python lacks NumPy" >&2; exit 1; }
  mkdir -p "$out/mesh"
  "$python" "$here/make_pipe.py" "$out/mesh" | tee "$out/mesh/make_pipe.log"
fi
cp "$here/solver.xml" "$out/solver.xml"
