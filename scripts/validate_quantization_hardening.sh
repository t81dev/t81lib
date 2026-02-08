#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-build-hardening}"
WITH_PYTHON=0
FULL=0

usage() {
  cat <<'EOF'
Usage: scripts/validate_quantization_hardening.sh [--with-python] [--full]

Runs the Phase 1 hardening validation lanes for t81lib:
  1) C++ arithmetic/core lane (required)
  2) Python quantization lane (optional, enabled with --with-python)
Default C++ lane excludes the long-running fuzz target (`bigint_bitops_fuzz`).
Use --full to include it.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-python)
      WITH_PYTHON=1
      shift
      ;;
    --full)
      FULL=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

echo "[hardening] lane A: C++ core arithmetic/tests"
cmake -S "$ROOT_DIR" -B "$ROOT_DIR/$BUILD_DIR" -DT81LIB_BUILD_TESTS=ON
cmake --build "$ROOT_DIR/$BUILD_DIR" --parallel
if [[ "$FULL" -eq 1 ]]; then
  ctest --test-dir "$ROOT_DIR/$BUILD_DIR" --output-on-failure
else
  ctest --test-dir "$ROOT_DIR/$BUILD_DIR" -E "bigint_bitops_fuzz" --output-on-failure
fi

if [[ "$WITH_PYTHON" -eq 1 ]]; then
  echo "[hardening] lane B: Python quantization/tests"
  if ! python3 -c "import numpy" >/dev/null 2>&1; then
    echo "Missing numpy. Install Python deps first (for example: pip install \".[torch]\")." >&2
    exit 1
  fi
  if ! python3 -c "import t81lib" >/dev/null 2>&1; then
    echo "Missing importable t81lib module. Install package first (for example: pip install \".[torch]\")." >&2
    exit 1
  fi
  (cd "$ROOT_DIR" && PYTHONPATH="$ROOT_DIR/python:$ROOT_DIR/src:${PYTHONPATH:-}" pytest -q tests/python/test_bindings.py tests/python/test_torch_ternary.py)
fi

echo "[hardening] all selected lanes passed"
