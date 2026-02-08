<!--
docs/index.md — Primary landing page for the documentation set.
-->

# t81lib docs portal

This landing page highlights the most helpful resources for people discovering `t81lib` or wanting
to understand the balanced ternary engine without digging through specs immediately.

## Featured example

Try the compact, end-to-end PTQ + QAT notebook that measures size, latency, and perplexity on Phi-3-mini:
[`examples/ternary_phi3_ptq_qat_demo.ipynb`](../examples/ternary_phi3_ptq_qat_demo.ipynb).

**Featured demo** — [Try Phi-3-mini PTQ + QAT](../examples/ternary_phi3_ptq_qat_demo.ipynb)

## Getting started for Torch users

If you are arriving from PyTorch or Hugging Face, use `t81` as the single entry point and alias it once:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install ".[torch]"
```

```python
import t81 as t8

tensor = t8.torch.TernaryTensor.from_float(weight, threshold=0.45)
output = tensor.matmul_input(input_tensor, bias=bias)
```

From here: `t8.nn.Linear` for drop-in layers, `t8.convert`/`t8.gguf` for scripted conversion, and `t81 convert`/`t81 gguf` for CLI workflows.

## Core resources

- **Landing & Quick Start** — [`README.md`](../README.md) contains the hero content, badges, and a comprehensive quick
  start section for builds, subprojects, pip/pipx consumers, CLI helpers, and the new “Start here” router.
- **Architecture guide** — [`ARCHITECTURE.md`](../ARCHITECTURE.md) walks through the component layers and data flow.
- **Normative spec** — [`docs/t81lib-spec-v0.1.0.md`](t81lib-spec-v0.1.0.md) defines the API guarantees for `limb`, `bigint`, and helpers.
- **Design notes** — [`docs/design/limb-design.md`](design/limb-design.md), [`docs/design/bigint-design.md`](design/bigint-design.md), [`docs/design/montgomery-design.md`](design/montgomery-design.md) explain internal
  choices, algorithms, and invariants.
- **Examples** — [`examples/`](../examples/) hosts runnable demos that mirror the README snippets.
- **Quantization reports** — [`examples/quantization_config_report.py`](../examples/quantization_config_report.py) enumerates synthetic parameter sweeps (dims, thresholds, sample counts) so you can compare multi-module accuracy/latency/size numbers before touching a checkpoint.
- **PyTorch bridge** — `t81.torch`/`t81.nn` expose the custom `t81.trit` dtype, `TernaryTensor`,
  and GEMM-backed helpers; see [`examples/demo_llama_conversion.py`](../examples/demo_llama_conversion.py), [`examples/scaling_laws_ternary.py`](../examples/scaling_laws_ternary.py), and
  [`examples/ternary_sparse_preview.py`](../examples/ternary_sparse_preview.py) for runnable workflows.
- **Python API reference** — [`docs/python-api.md`](python-api.md) lays out how MkDocs plus `mkdocstrings` auto-generate the binding reference.
- **Python cookbook** — [`docs/python-cookbook.md`](python-cookbook.md) gathers recipes that mix `t81lib.pack_dense_matrix`, `t81.torch.TernaryTensor`, and the CLI helpers.
- **Python entry points** — [`docs/python-api.md`](python-api.md) and [`docs/python-cookbook.md`](python-cookbook.md) now include a quick table showing which module to import for each workflow.
- **Python install paths** — [`docs/python-install.md`](python-install.md) explains pip/pipx builds, validation tips, and CLI helper installs.
- **PyTorch how-to** — [`docs/torch.md`](torch.md) walks through `t81.torch`, `t81.nn`, conversion helpers, and how the CLI scripts mirror the Python flows.
- **CLI reference** — [`docs/references/cli-usage.md`](references/cli-usage.md) lists the unified `t81 convert`/`t81 gguf` helpers (with legacy `t81-convert`/`t81-gguf` aliases) plus `t81-qat`
  plus the common flags for exporting GGUF bundles and running QAT.
- **Hardware & energy reference** — [`docs/references/hardware-emulation.md`](references/hardware-emulation.md) connects `t81.hardware.TernaryEmulator`
  with the Python quantization helpers plus the new [`scripts/quantize_measure.py`](../scripts/quantize_measure.py) automation that chains
  `t81 convert` → measurement.
- **Python demos** — the [`examples/`](../examples/) scripts/notebooks track `t81.torch` + `t81.nn` workflows; add
  [`examples/ternary_qat_inference_comparison.py`](../examples/ternary_qat_inference_comparison.py) to kick off a mini `t81.trainer` QAT loop, print the ternary
  threshold schedule, and compare `torch.matmul` vs. `t81lib.gemm_ternary` latency so you can prototype
  entirely inside Python before launching the CLI helpers.
- **CLI automation & energy benchmarking** — [`scripts/quantize_measure.py`](../scripts/quantize_measure.py) and [`scripts/quantize_energy_benchmark.py`](../scripts/quantize_energy_benchmark.py)
  chain `t81 convert`/`t81 gguf` runs with latency/energy measurement so you can report quantization impact directly
  from command-line workflows.
- **Use cases & demos** — [`docs/use-cases.md`](use-cases.md) and [`examples/README.md`](../examples/README.md) capture the canonical scripts, notebooks, and research stories.
- **Hardware simulation** — [`docs/hardware.md`](hardware.md) details `t81.hardware.TernaryEmulator`, fuzzy helpers, and the visualizer notebook.
- **GPU backends** — [`docs/gpu.md`](gpu.md) explains the CUDA/ROCm build flags and tensor metadata routing.
- **API overview** — [`docs/api-overview.md`](api-overview.md) summarizes the numeric containers and helpers exposed via `<t81/t81lib.hpp>`.
- **Tests & benchmarks** — [`tests/`](../tests/) documents the unit/property coverage while [`bench/`](../bench/) shows throughput patterns.
- **Phase 1 hardening checklist** — [`docs/quantization-hardening.md`](quantization-hardening.md) defines the arithmetic/quantization stabilization gates and reproducible validation flow.
- **Docs sitemap** — the [`docs/diagrams/docs-sitemap.mermaid.md`](diagrams/docs-sitemap.mermaid.md) mind map visualizes the content hierarchy referenced on this page.

## Stay aligned

1. Review `CONTRIBUTING.md` before opening a PR—workflows, invariants, and branch expectations are documented there.
2. Check `CHANGELOG.md` to understand recent breaking changes or stabilization notes.
3. Run `cmake -S . -B build -DT81LIB_BUILD_TESTS=ON` + `ctest` after local changes to keep deterministic behavior intact.

## Want to present `t81lib`?

Use this portal when pitching the library internally or prepping release notes. The combination of
`README.md`, `ARCHITECTURE.md`, and `docs/` creates a cohesive narrative that balances hands-on examples,
design rationale, and testing expectations.
