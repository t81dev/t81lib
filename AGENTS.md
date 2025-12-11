<!--
AGENTS.md — Guidance for AI agents exploring and modifying this repository.
-->

# AGENTS

This file helps AI agents discover and understand how to work with this repository.

## Discovery

- **Primary entry points:** `README.md`, `include/`, `src/`, and `tests/` describe the architecture and entry points for this library. Use `rg` to locate interesting symbols before jumping into implementation.
- **Python bindings:** The new `python/` directory holds the pybind11-based module and `tests/python/test_bindings.py` exercises it; toggle `T81LIB_BUILD_PYTHON_BINDINGS` when configuring CMake to build the module.
- **Build tooling:** The project uses CMake. Inspect `CMakeLists.txt` and related files in `cmake/` or `doc/` for build and test instructions before making changes.

## Agent guidelines

- Follow the existing coding style in `include/t81/core/` and use ASCII-only edits unless a file already includes other Unicode characters.
- Prefer `rg` for searching and avoid destructive operations (`git reset --hard`, etc.).
- Respect non-AI manual edits in the working tree; do not revert unless asked.

## Suggested workflow

1. Run any relevant unit tests in `tests/unit/` via CTest or the provided scripts whenever you touch critical paths to verify behavior.
2. Document significant algorithm changes in `doc/` or `README.md` as appropriate.
3. Mention new files or important updates back in this file so future agents can find your work quickly.

## Recent updates

- Balanced ternary bigint logic in `include/t81/core/bigint.hpp` now normalizes signed limbs more efficiently and fixes `~`/division helpers so later agents can spot the modern bitwise/division flow.
- `tests/unit/test_numeric_types.cpp` now exercises `Complex`, `Polynomial`, and `F2m` helpers so the umbrella numeric helpers stay locked down.
