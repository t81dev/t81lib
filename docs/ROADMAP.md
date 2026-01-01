# Project Roadmap & Analysis: t81lib

## 1. Executive Summary

`t81lib` is a high-performance, header-first C++20 library that brings deterministic balanced ternary arithmetic (via `t81::core::limb`, `Int`, and `BigInt`) together with SIMD-accelerated GEMM kernels, Python bindings (`t81lib`, `t81`, and `t81.torch` helpers), and CLI tooling for packed ternary quantization. The repository is technically mature, well-documented, and already ships production-ready conversion/quantization pipelines, which makes it attractive for both systems developers and AI/ML teams. The most valuable growth areas are making the contributor journey more welcoming, automating CI/code-quality feedback, and spotlighting the Python API so the broader Python/torch community can find the right entry points faster.

## 2. Overview Analysis

### Project Purpose, Goals, and Target Users

* **Purpose**: Offer a deterministic, mathematically rich ternary arithmetic foundation (limbs, bigints, balanced `Fixed<N>`, `Float`/`FloatN`, etc.) plus practical quantization tools and pybind11 bindings that serve AI workflows.
* **Goals**: Keep the C++ core razor-fast (SIMD/AVX/NEON, balanced ternary kernels, canonical encodings) while shipping Python/CLI helpers that smoothly quantize, pack, and run ternary GEMMs in PyTorch and CLI conversions.
* **Target Users**:
  1. **C++ engineers** needing high-performance arbitrary precision/ternary numerics (crypto, DSP, specialized hardware).
  2. **AI/ML researchers and engineers** who want deterministic, space-efficient quantization through `t81lib`/`t81`/`t81.torch` helpers and CLI tools like `t81-convert`, `t81-gguf`, and `t81-qat`.

### Tech Stack

* **Languages**: Modern C++20 (mostly header-only) plus Python bindings (`pybind11`) and CLI wrappers.
* **Build system**: CMake with optional Python binding flag `T81LIB_BUILD_PYTHON_BINDINGS`, plus Vcpkg integration for distributed builds.
* **Python dependencies**: PyTorch, Transformers, Accelerate, NumPy for quantization and CLI helpers.
* **Testing/benchmarking**: Google Test/Benchmark, CTest, and scripted Python tests under `tests/python`.
* **Dev tooling**: `clang-format`, docs site via `docs/index.md`, and CLI docs/workflow diagrams (`docs/references`, `docs/diagrams`).

### Repository Structure

| Path | Role |
| --- | --- |
| `include/` & `src/` | Core C++ API (limbs, bigints, `linalg`, `Float`, `Fixed`, `Matrix`, etc.) and implementation.
| `python/` | Pybind11 bindings (`bindings.cpp`) plus packaging helpers (`setup.py`, `pyproject.toml`).
| `t81/` | Pure-Python helpers (`t81.torch`, `t81.nn`, hardware emulation, CLI validation/progress helpers) that build on the C++ core.
| `tests/` | C++ unit tests (`tests/unit`), Python bindings/gguf tests, and integration suites.
| `examples/` | Notebooks/scripts for quantization demos, QAT inference comparisons, CLI examples, and hardware emulation.
| `docs/` | Reference docs, diagrams, CLI usage guides, and roadmap/spec content for contributors.
| `scripts/` | Automation for quantize+measure, energy benchmarking, and reporting.
| `AGENTS.md` / `ARCHITECTURE.md` | Onboarding/help for contributors and architecture overview.

### Code Quality & Documentation

The codebase is thoughtfully commented, consistently formatted by `clang-format`, and split into cohesive modules (core, `linalg`, CLI integration, hardware emulation). A comprehensive testing matrix already exists and the docs cover high-level concepts (`README.md`, `ARCHITECTURE.md`, `docs/index.md`) alongside CLI/usage references, making it easier for users to understand the feature set.

### Strengths & Weaknesses

#### Strengths

* Exceptional documentation (README, ARCHITECTURE, CLI references, diagrams, and AGENTS instructions).
* Balanced focus on high-performance C++ numerics and practical Python/CLI AI tooling.
* Already ships Python CLI helpers (`t81-convert`, `t81-gguf`, `t81-qat`) and quantization examples/notebooks.
* Modular repo structure keeps bounds between core arithmetic, bindings, scripts, and docs.

#### Weaknesses

* Developer onboarding still complex because of the many CMake options, Python extras, and optional dependencies.
* CI could better enforce format/lint rules and test permutations (compilers, binding configurations, and Python coverage).
* Python API discoverability is fragmented between `t81lib` and the higher-level `t81` helpers; documentation could unify them.

## 3. Next Steps Recommendations

### Progress update (2024)

Recent work has delivered parts of this roadmap:

* **Recommendation 1** — quickstart matrix + common workflows added to `DEVELOPMENT.md`.
* **Recommendation 2** — CI matrix expanded for OS/build types and SIMD guards; Python tests standardized on Linux.
* **Recommendation 3** — Python entry-points table added to `docs/python-api.md` and `docs/python-cookbook.md`, with links from `docs/index.md`. **In progress (benchmark visibility added in `README.md`, `BENCHMARKS.md`, and the Phi-3 notebook).**
* **GGUF compatibility** — Phi-3 export validated (`phi3-tq1-fixed12.gguf`); QKV split experiment reverted for llama.cpp parity.
* **QAT benchmark groundwork** — ViT CIFAR-10 PTQ/QAT script added with size-only baseline captured; Phi-3 baseline PPL captured (PTQ PPL/QAT pending).
* **GPU fallback safety** — `t81.torch` now warns + falls back to CPU for PTQ when tensors originate on GPU; smoke test added and troubleshooting docs updated.

### Status timeline (recent highlights)

* Python entry-point discoverability refreshed (docs landing page + cookbook + API entry table).
* Phi-3 GGUF export validated with llama.cpp baseline metrics captured for reference.
* CLI documentation updated to call out Phi-3 GGUF compatibility expectations.
* ViT size-only baseline logged; Phi-3 baseline PPL captured with PTQ PPL/QAT queued.
* GPU fallback behavior documented; `.gitignore` hardened against GGUF/cache artifacts.

### High-impact next priorities (effort vs. impact)

1. **Recommendation 4 — Standardized QAT benchmark (high effort, high impact)**: define a reproducible suite (e.g., Phi-3 Mini fine-tune on OpenAssistant/oasst1 or a small ViT on CIFAR-10) that compares FP16 → PTQ → QAT. Capture perplexity/accuracy, model size, and tok/s on CPU (llama.cpp) plus GPU (when bindings are ready), and publish baselines in `BENCHMARKS.md` with JSON artifacts.
2. **Recommendation 5 — GPU tensor metadata + dispatcher hardening (medium effort, high impact)**: stabilize the `TensorMetadata` ABI for safe `device_ptr` handling, add broadcasting/contiguous fallbacks, and certify CUDA/ROCm kernels in CI with latency/accuracy parity against CPU.
3. **Polish & community**: add issue templates + “good first issue” labels, ship pre-built wheels (including CUDA-enabled variants), and publish the Phi-3 TQ1_0 GGUF for community testing.

Remaining items are listed below with the next steps still required.

### Recommendation 1: Streamline the Developer Onboarding Experience

* **Why**: New contributors struggle with CMake options, Python extras, and building bindings.
* **Benefits**: Faster first-time builds, fewer setup questions, more contributions.
* **Effort**: Medium.
* **Implementation**:
  1. Expand `DEVELOPMENT.md` with a quickstart matrix (CMake-only vs. bindings vs. torch extras) and explicit `T81LIB_BUILD_PYTHON_BINDINGS` examples. **Done.**
  2. Add a short "common workflows" section that references `run-tests.sh` and `build-python.sh`, plus expected outputs/flags. **Done.**
  3. Decide whether a `.devcontainer` is still needed or document the current preferred local setup to avoid duplicate paths. **Done.**

### Recommendation 2: Expand CI and Code Quality Automation

* **Why**: The existing CI could do more to prevent regressions and enforce style across languages.
* **Benefits**: Higher confidence in main, earlier detection of regressions, better cross-platform coverage.
* **Effort**: Medium.
* **Implementation**:
  1. Extend `.github/workflows/ci.yml` with a richer build matrix (GCC/Clang, bindings vs. minimal configuration, AVX vs. scalar) and run both C++ and Python test suites. **Done (matrix + SIMD guard updates).**
  2. Add format/lint steps (`clang-format` check, `ruff`/`black` for Python) to gate style. **Done.**
  3. Publish coverage/artifacts (e.g., via Codecov or GHA artifacts) so maintainers can monitor test completeness. **Done (coverage artifact upload).**

### Recommendation 3: Unify and Document the Python API Surface

* **Why**: Python users currently discover helpers across `t81lib`, `t81`, and CLI docs.
* **Benefits**: Easier discoverability, faster adoption, clearer path from C++ bindings to Torch wrappers.
* **Effort**: Low-Medium.
* **Status**: In progress (benchmark visibility added in `README.md`, `BENCHMARKS.md`, and the Phi-3 notebook).
* **Implementation**:
  1. Expand MkDocs coverage by generating the Python API reference via mkdocstrings and ensuring key modules are linked from `docs/index.md`. **Done (entry-point links + extra directives).**
  2. Keep the “Python Cookbook” up to date with end-to-end recipes (bindings + `t81.torch` + CLI), and add a short "choose your entry point" table. **Done (entry points table).**
  3. Validate that the `t81` re-exports stay in sync with `t81lib` bindings and add a quick API surface checklist. **Done (checklist added).**

### Recommendation 4: Introduce a Standardized Quantization-Aware Training Benchmark

* **Why**: QAT features are powerful but lack standard public benchmarks to illustrate benefits.
* **Benefits**: Demonstrates effectiveness, guides performance tuning, attracts AI users.
* **Effort**: High.
* **Status**: In progress (ViT size-only baseline logged, Phi-3 baseline PPL captured, JSON artifacts added; PTQ PPL/QAT pending).
* **Implementation**:
  1. Define a benchmark (e.g., small BERT or ViT on GLUE/CIFAR subsets) with FP32, PTQ, and QAT runs.
  2. Extend the existing `scripts/` benchmark tooling to log accuracy, model size, and latency in a standardized JSON schema.
  3. Update `BENCHMARKS.md` with reproducible baseline results and link the dataset/model artifacts used for comparisons.

### Recommendation 5: Harden the GPU Tensor Metadata Path

* **Why**: GPU acceleration for ternary ops unlocks real inference speedups only if the dispatcher can safely read raw device pointers and keep outputs on-device, and the Python bindings need a stable contract (TensorMetadata) to wrap Torch/NumPy tensors the same way.
* **Benefits**: Enables CUDA/ROCm kernels (where/clamp/lerp/addcmul) to run without host copies, makes GPU/CPU fallbacks explicit, and gives PyTorch users a predictable ABI.
* **Effort**: Medium.
* **Implementation**:
  1. Continue documenting the `t81::TensorMetadata` ABI (`device_type`, `dtype`, `sizes`, `strides`, `data_ptr`, `storage_offset`, `requires_sync`) so future kernels and Python helpers rely on the same struct.
  2. Expand the dispatcher to record broadcasting/non-contiguous layouts (or copy them to contiguous buffers when needed) so `backend_available == CUDA/ROCm` paths no longer require perfect contiguity.
  3. Run the CUDA 12.4 / ROCm 6+ CI jobs, validate the latency/accuracy guardrails in `tests/python/test_gpu_ops.py`, and flag any ABI mismatches between `torch.Tensor` and `TensorMetadata` so the GPU helpers can be certified before release.

---

The above roadmap tightly ties into the existing structure (`docs/`, `scripts/`, CLI helpers) while making the project more approachable, ensuring quality, highlighting the Python story, and showcasing real-world value through benchmarks.
