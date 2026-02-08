# Quantization and Arithmetic Hardening

Roadmap linkage:
- `t81-roadmap#2`
- `t81-roadmap/PHASE1_STABILIZATION_MATRIX.md` (`P1-S2`)

This document is the Phase 1 hardening checklist for `t81lib`.

## Hardening Checklist

- [x] Define a reproducible core arithmetic validation lane (C++ build + unit tests).
- [x] Define a reproducible quantization validation lane (Python tests with explicit deps).
- [x] Add one command entry point that runs both lanes in a consistent order.
- [x] Require changed quantization claims to include benchmark artifact references.
- [x] Route contract-impacting changes back to roadmap tracking.

## Reproducible Validation Paths

### Lane A: Core arithmetic (required)

Runs deterministic C++ checks through CMake/CTest:

```bash
scripts/validate_quantization_hardening.sh
```

### Lane B: Python quantization surface (optional but required for Python/API changes)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install ".[torch]"
scripts/validate_quantization_hardening.sh --with-python
```

## Benchmark Artifact Rule

If a change alters quantization behavior or performance claims, include:

1. Command used (for example `scripts/ternary_quantization_benchmark.py` args).
2. Output artifact path (`benchmarks/*.csv` or `benchmarks/*.json`).
3. Before/after summary in PR or issue notes.

## Contract/Governance Rule

When quantization behavior changes impact cross-repo assumptions, open or update linked tracking in:

1. `t81-roadmap` (Phase 1 tracker),
2. downstream contract consumers as needed (`t81-python`, `t81-lang`).
