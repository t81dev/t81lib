#!/ usr / bin / env bash
set - euo pipefail

          BUILD_DIR =
    "${BUILD_DIR:-build-python}" cmake - S.- B "$BUILD_DIR" - DT81LIB_BUILD_TESTS =
        ON - DT81LIB_BUILD_PYTHON_BINDINGS = ON cmake-- build "$BUILD_DIR" --parallel
