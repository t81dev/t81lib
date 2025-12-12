<!--
AGENTS.md — Guidance for AI agents exploring and modifying this repository.
-->

# AGENTS

This file helps AI agents discover and understand how to work with this repository.

## Discovery

- **Primary entry points:** `README.md`, `include/`, `src/`, and `tests/` describe the architecture and entry points for this library. Use `rg` to locate interesting symbols before jumping into implementation.
- **Python bindings:** The new `python/` directory holds the pybind11-based module and `tests/python/test_bindings.py` exercises it; toggle `T81LIB_BUILD_PYTHON_BINDINGS` when configuring CMake to build the module.
- **Build tooling:** The project uses CMake. Inspect `CMakeLists.txt` and related files in `cmake/` or `docs/` for build and test instructions before making changes.

## Agent guidelines

- Follow the existing coding style in `include/t81/core/` and use ASCII-only edits unless a file already includes other Unicode characters.
- Prefer `rg` for searching and avoid destructive operations (`git reset --hard`, etc.).
- Respect non-AI manual edits in the working tree; do not revert unless asked.

## Suggested workflow

1. Run any relevant unit tests in `tests/unit/` via CTest or the provided scripts whenever you touch critical paths to verify behavior.
2. Document significant algorithm changes in `docs/` or `README.md` as appropriate.
3. Mention new files or important updates back in this file so future agents can find your work quickly.

## Recent updates

- Reworked the top-level `CMakeLists.txt`, rewrote `run-tests.sh` to execute configure/build/tests, and reordered `tests/unit/test_limb_basic.cpp` so `t81/t81lib.hpp` is included before the SIMD helpers to keep `limb` defined.
- Balanced ternary bigint logic in `include/t81/core/bigint.hpp` now normalizes signed limbs more efficiently and fixes `~`/division helpers so later agents can spot the modern bitwise/division flow.
- `tests/unit/test_numeric_types.cpp` now exercises `Complex`, `Polynomial`, and `F2m` helpers so the umbrella numeric helpers stay locked down.
- `README.md` now documents the high-level helpers (`Float`, `Ratio`, `Complex`, `Polynomial`, `F2m`, `Fixed<N>`, `Modulus`, and `MontgomeryInt`) plus the `t81::Int` alias exposed through `t81/t81lib.hpp`.
- `include/t81/t81lib.hpp` now exposes `Float::from_string`, a `Ratio`→`Float` conversion, the `Int81` `Fixed<48>` alias, and `std::hash` hooks for `limb`/`bigint` so hashing and string-based floats land in the umbrella header.
- `README.md` plus the umbrella header now document the `FloatN` template, ternary `_t3` literal, R3 NTT helpers, and `std::formatter` specializations so overlined ternary floats behave nicely in `std::format`, and `t81::Vector` provides a ready-to-use coefficient container with arithmetic helpers.
- `README.md`/umbrella header now mention `t81::Matrix<Element, R, C>` and how it complements `Vector` for linear algebra over `FloatN`/`Fixed` scalars.
- `F2m` now lives in `include/t81/gf2m.hpp` (still re-exported through `t81/t81lib.hpp`), and `Fixed<N>` gained balanced `/` and `%` helpers so division/magnitude math stays accessible in the umbrella header.
- Added `t81::linalg::gemm_ternary` and the Python binding `t81lib.gemm_ternary` so packed ternary GEMMs with alpha/beta semantics are exposed across the C++/Python API surface.
- Documented the `t81.torch`/`t81.nn` PyTorch helpers in `README.md` and `docs/index.md`, pointing to the `examples/demo_llama_conversion.py`,
  `examples/scaling_laws_ternary.py`, and `examples/ternary_sparse_preview.py` demos so future agents can locate the torch bridge.
- Added production-ready Python bindings (`python/bindings.cpp`) plus packaging helpers (`setup.py`, `pyproject.toml`) that expose `Limb`/`BigInt` helpers, Montgomery contexts, NumPy quantization utilities, and a tutorial notebook `examples/ternary_quantization_demo.ipynb`.
- Added `t81.hardware.TernaryEmulator`, documentation for hardware simulation, and `examples/ternary_hardware_sim_demo.ipynb` so agents can explore ternary gate/circuit modeling, fuzzy AI decisions, and power-aware PyTorch inference workflows.
- Added `docs/references/cli-usage.md` (linked from `docs/index.md`) to cover `t81-convert`, `t81-gguf`, and `t81-qat` usage with the CPU/offloading tips we surfaced for low-memory Apple Silicon.
- Added `docs/diagrams/cli-workflows-mermaid.md` to visualize the `t81-convert`, `t81-gguf`, and `t81-qat` workflows for future contributors looking at the CLI surface.
- Extended `examples/ternary_qat_inference_comparison.py` so it now runs train + validation loops, logs compression ratios + per-step losses, and correlates the ternary threshold history with measured GEMM latencies.
- Added `scripts/quantize_measure.py`, which chains `t81-convert` → `AutoModel.from_pretrained_t81` → latency/compression stats so you can automate quantize→measure in other pipelines.
- Added `docs/references/hardware-emulation.md` to explain how `t81.hardware.TernaryEmulator`, the Python quantization helpers, and the CLI automation fit together for energy-aware AI reasoning.
- Added `scripts/quantize_energy_benchmark.py` to orchestrate quantize→latency+energy benchmarks, logging compression, timing, and emulator energy stats into CSV/JSON outputs for reuse in reports.
- Added `examples/quantization_config_report.py` so you can sweep synthetic datasets (dims, thresholds, sizes) and capture accuracy, latency, and storage comparisons for multi-module configs before quantizing real models.
- Added `t81/cli_validator.py` plus a `--validate` flag for `t81-convert`/`t81-gguf` so the CLI reruns `gguf.read_gguf` (and llama.cpp’s `gguf_validate`/`gguf_to_gguf` when available) to ensure exported GGUF bundles stay compatible before a run returns success.
- Added `t81/cli_progress.py` plus progress logging to `t81-convert`, `t81-gguf`, and `t81-qat` so the CLIs print bar/percentage updates while converting, exporting, or fine-tuning checkpoints.
- Documented the automation scripts (`scripts/quantize_measure.py`, `scripts/quantize_energy_benchmark.py`) plus the CLI telemetry/progress experience so future agents can quickly measure quantization impact, latency, and hardware energy from the console.
- Added `examples/cli-examples.md` with ready-to-copy CLI snippets showing conversion, GGUF export, and QAT flows for the three helpers.
- Updated `README.md` to highlight the CLI docs/diagrams/examples so newcomers can find the new references through the main overview.
- Added `docs/ROADMAP.md` to capture an executive summary, analysis, and next-step recommendations for steering t81lib toward wider adoption and smoother contributions.
- Added `mkdocs.yml`, `docs/python-api.md`, and `docs/python-cookbook.md` so MkDocs + mkdocstrings can publish the Python API reference and cookbook, and linked them from `docs/index.md`.
- Expanded `python/t81/__init__.py` so the higher-level `t81` package re-exports the compiled binding helpers (`t81lib`, `BigInt`, `Limb`, `gemm_ternary`, etc.) while staying import-safe when the extension is unavailable.
- Added `scripts/ternary_quantization_benchmark.py` plus `BENCHMARKS.md` so contributors can reproduce a Fashion-MNIST FP32/PTQ/QAT benchmark and log accuracy/latency/storage for each mode; README now links the benchmark doc.
- Rewrote `pyproject.toml` with valid TOML sections so editable installs (and `pip install -e '.[torch]'`) can parse the metadata cleanly before building the extension.
- Restructured `README.md` into a onboarding-focused front door and added companion docs (`docs/use-cases.md`, `docs/hardware.md`, `docs/api-overview.md`, `docs/python-install.md`, `docs/torch.md`, `examples/README.md`) so heavy reference material lives outside the visitor-facing overview.
