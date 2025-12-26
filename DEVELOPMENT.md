# Developer Onboarding Guide

This project already boasts extensive documentation (`README.md`, `docs/index.md`, `AGENTS.md`, `docs/ROADMAP.md`), but this guide focuses on the concrete steps a new contributor needs to take so they can build, test, and iterate quickly.

## 1. Clone & configure

```bash
git clone https://github.com/t81dev/t81lib.git
cd t81lib
```

The repository uses CMake for every build target, so keep a spare build directory for each configuration (e.g., `build`, `build-python`).

## 2. Quickstart matrix

Pick the smallest path that matches your task:

**C++ core only** (no Python bindings)

```bash
cmake -S . -B build -DT81LIB_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build
```

**C++ + Python bindings (pybind11)**

```bash
cmake -S . -B build-python -DT81LIB_BUILD_PYTHON_BINDINGS=ON -DT81LIB_BUILD_TESTS=ON
cmake --build build-python
PYTHONPATH=build-python python tests/python/test_bindings.py
```

**CLI helpers (torch/transformers extras)**

```bash
pipx install .[torch]
pipx ensurepath
t81 --help
```

## 3. Native C++ build & tests

```bash
./run-tests.sh
```

`run-tests.sh` (located in the repo root) configures CMake with `-DT81LIB_BUILD_TESTS=ON`, builds the default targets, and runs `ctest`. The script respects the `BUILD_DIR` environment variable, so you can run `BUILD_DIR=build-debug ./run-tests.sh` if you want a custom path.

## 4. Python bindings

```bash
./build-python.sh
PYTHONPATH=build-python python tests/python/test_bindings.py
```

`build-python.sh` configures CMake with `-DT81LIB_BUILD_PYTHON_BINDINGS=ON` and builds the targets needed for the pybind11 module. After the build completes, point `PYTHONPATH` at the build directory before running the Python tests or importing `t81lib` in a script.

If you prefer isolation, install the package via `pipx`:

```bash
pipx install .[torch]
pipx ensurepath
```

The console scripts `t81-convert`, `t81-gguf`, and `t81-qat` become available after the pipx installation.

## 5. Common workflows

Use the scripts in the repo root for the most common tasks:

```bash
# clean configure + build + ctest
./run-tests.sh

# configure + build the pybind11 module
./build-python.sh

# custom build dir
BUILD_DIR=build-debug ./run-tests.sh
```

Expected outputs:

- `run-tests.sh` prints the CMake configure summary, build steps, and `ctest` results.
- `build-python.sh` emits the binding target and leaves the extension module in the build dir.

## 6. CLI helpers

All CLI workflow documentation lives in `docs/references/cli-usage.md`, and the Mermaid diagrams are in `docs/diagrams/cli-workflows-mermaid.md`. Consult those docs for flag explanations, input requirements, and usage examples before writing CLI-focused contributions.

## 7. Documentation & roadmap

If you're updating architecture or proposing a major change, refer to `docs/ROADMAP.md` for the current vision and the recommended initiatives that maintainers are tracking. Document your work in the nearest relevant doc (README, docs/index, AGENTS, etc.).

## 8. Developer container

Preferred setup is a native local toolchain. Use the `.devcontainer` only if you need a fully reproducible VS Code + Docker environment with CMake, Ninja, Clang, Python, and pipx pre-installed. After opening the folder in VS Code, select **Reopen in Container** and let the container build once.

## 9. Additional tips

* Keep your changes small and test locally before opening a PR.
* Run `clang-format` on files you touch (see `.clang-format` for the style configuration).
* When touching bindings or Python helpers, re-run the relevant tests in `tests/python/` to confirm the public interface behaves as expected.

Welcome to `t81lib`! Let us know if anything in this guide needs clarification.
