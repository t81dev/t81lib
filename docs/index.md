<!--
docs/index.md — Primary landing page for the documentation set.
-->

# t81lib docs portal

This landing page highlights the most helpful resources for people discovering `t81lib` or wanting
to understand the balanced ternary engine without digging through specs immediately.

## Core resources

- **Landing & Quick Start** — `README.md` contains the hero content, badges, and a comprehensive quick
  start section for builds, subprojects, and vcpkg consumers.
- **Architecture guide** — `ARCHITECTURE.md` walks through the component layers and data flow.
- **Normative spec** — `docs/t81lib-spec-v1.0.0.md` defines the API guarantees for `limb`, `bigint`, and helpers.
- **Design notes** — `docs/design/limb-design.md`, `.../bigint-design.md`, `.../montgomery-design.md` explain internal
  choices, algorithms, and invariants.
- **Examples** — `examples/` hosts runnable demos that mirror the README snippets.
- **PyTorch bridge** — `t81.torch`/`t81.nn` expose the custom `t81.trit` dtype, `TernaryTensor`,
  and GEMM-backed helpers; see `examples/demo_llama_conversion.py`, `examples/scaling_laws_ternary.py`, and
  `examples/ternary_sparse_preview.py` for runnable workflows.
- **Tests & benchmarks** — `tests/` documents the unit/property coverage while `bench/` shows throughput patterns.

## Stay aligned

1. Review `CONTRIBUTING.md` before opening a PR—workflows, invariants, and branch expectations are documented there.
2. Check `CHANGELOG.md` to understand recent breaking changes or stabilization notes.
3. Run `cmake -S . -B build -DT81LIB_BUILD_TESTS=ON` + `ctest` after local changes to keep deterministic behavior intact.

## Want to present `t81lib`?

Use this portal when pitching the library internally or prepping release notes. The combination of
`README.md`, `ARCHITECTURE.md`, and `docs/` creates a cohesive narrative that balances hands-on examples,
design rationale, and testing expectations.
