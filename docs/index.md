<!--
docs/index.md — Primary landing page for the documentation set.
-->

# t81lib docs portal

This landing page highlights the most helpful resources for people discovering `t81lib` or wanting
to understand the balanced ternary engine without digging through specs immediately.

## Core resources

- **Landing & Quick Start** — [`README.md`](../README.md) contains the hero content, badges, and a comprehensive quick
  start section for builds, subprojects, pip/pipx consumers, CLI helpers, and the new “Start here” router.
- **Architecture guide** — [`ARCHITECTURE.md`](../ARCHITECTURE.md) walks through the component layers and data flow.
- **Normative spec** — [`docs/t81lib-spec-v1.0.0.md`](t81lib-spec-v1.0.0.md) defines the API guarantees for `limb`, `bigint`, and helpers.
- **Design notes** — [`docs/design/limb-design.md`](design/limb-design.md), [`docs/design/bigint-design.md`](design/bigint-design.md), [`docs/design/montgomery-design.md`](design/montgomery-design.md) explain internal
  choices, algorithms, and invariants.
- **Examples** — [`examples/`](../examples/) hosts runnable demos that mirror the README snippets.
- **Quantization reports** — [`examples/quantization_config_report.py`](../examples/quantization_config_report.py) enumerates synthetic parameter sweeps (dims, thresholds, sample counts) so you can compare multi-module accuracy/latency/size numbers before touching a checkpoint.
- **PyTorch bridge** — `t81.torch`/`t81.nn` expose the custom `t81.trit` dtype, `TernaryTensor`,
  and GEMM-backed helpers; see [`examples/demo_llama_conversion.py`](../examples/demo_llama_conversion.py), [`examples/scaling_laws_ternary.py`](../examples/scaling_laws_ternary.py), and
  [`examples/ternary_sparse_preview.py`](../examples/ternary_sparse_preview.py) for runnable workflows.
- **Python API reference** — [`docs/python-api.md`](python-api.md) lays out how MkDocs plus `mkdocstrings` auto-generate the binding reference.
- **Python cookbook** — [`docs/python-cookbook.md`](python-cookbook.md) gathers recipes that mix `t81lib.pack_dense_matrix`, `t81.torch.TernaryTensor`, and the CLI helpers.
- **Python install paths** — [`docs/python-install.md`](python-install.md) explains pip/pipx builds, validation tips, and CLI helper installs.
- **PyTorch how-to** — [`docs/torch.md`](torch.md) walks through `t81.torch`, `t81.nn`, conversion helpers, and how the CLI scripts mirror the Python flows.
- **CLI reference** — [`docs/references/cli-usage.md`](references/cli-usage.md) lists the `t81-convert`, `t81-gguf`, and `t81-qat` helpers
  plus the common flags for exporting GGUF bundles and running QAT.
- **Hardware & energy reference** — [`docs/references/hardware-emulation.md`](references/hardware-emulation.md) connects `t81.hardware.TernaryEmulator`
  with the Python quantization helpers plus the new [`scripts/quantize_measure.py`](../scripts/quantize_measure.py) automation that chains
  `t81-convert` → measurement.
- **Python demos** — the [`examples/`](../examples/) scripts/notebooks track `t81.torch` + `t81.nn` workflows; add
  [`examples/ternary_qat_inference_comparison.py`](../examples/ternary_qat_inference_comparison.py) to kick off a mini `t81.trainer` QAT loop, print the ternary
  threshold schedule, and compare `torch.matmul` vs. `t81lib.gemm_ternary` latency so you can prototype
  entirely inside Python before launching the CLI helpers.
- **CLI automation & energy benchmarking** — [`scripts/quantize_measure.py`](../scripts/quantize_measure.py) and [`scripts/quantize_energy_benchmark.py`](../scripts/quantize_energy_benchmark.py)
  chain `t81-convert`/`t81-gguf` runs with latency/energy measurement so you can report quantization impact directly
  from command-line workflows.
- **Use cases & demos** — [`docs/use-cases.md`](use-cases.md) and [`examples/README.md`](../examples/README.md) capture the canonical scripts, notebooks, and research stories.
- **Hardware simulation** — [`docs/hardware.md`](hardware.md) details `t81.hardware.TernaryEmulator`, fuzzy helpers, and the visualizer notebook.
- **GPU backends** — [`docs/gpu.md`](gpu.md) explains the CUDA/ROCm build flags and tensor metadata routing.
- **API overview** — [`docs/api-overview.md`](api-overview.md) summarizes the numeric containers and helpers exposed via `<t81/t81lib.hpp>`.
- **Tests & benchmarks** — [`tests/`](../tests/) documents the unit/property coverage while [`bench/`](../bench/) shows throughput patterns.

## Stay aligned

1. Review `CONTRIBUTING.md` before opening a PR—workflows, invariants, and branch expectations are documented there.
2. Check `CHANGELOG.md` to understand recent breaking changes or stabilization notes.
3. Run `cmake -S . -B build -DT81LIB_BUILD_TESTS=ON` + `ctest` after local changes to keep deterministic behavior intact.

## Want to present `t81lib`?

Use this portal when pitching the library internally or prepping release notes. The combination of
`README.md`, `ARCHITECTURE.md`, and `docs/` creates a cohesive narrative that balances hands-on examples,
design rationale, and testing expectations.
