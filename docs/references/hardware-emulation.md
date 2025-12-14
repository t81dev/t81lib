<!--
docs/references/hardware-emulation.md — How hardware emulation ties to python quantization helpers.
-->

# Hardware emulation & ternary tooling

`TernaryEmulator` in `t81.hardware` is a lightweight trit-level simulator that records wire transitions, energy traces, and fuzzy/binary fallbacks so you can reason about different deployment budgets before burning silicon. Each wire stores a `t81lib.Limb`, transitions call `_estimate_energy`, and helpers such as `ripple_adder`, `balanced_adder`, and `clock_tick` preserve the power-trace bookkeeping you'll want to compare against the packed GEMMs used in `t81.torch`.

## Correlating quantization metrics with energy budgets

1. **Quantize via the CLI** — run `t81 convert` (or the new `scripts/quantize_measure.py` helper) to turn a Hugging Face checkpoint into ternary `t81.nn.Linear` layers. This script reuses the CLI for conversion and then loads the converted checkpoint via `AutoModel.from_pretrained_t81`, timing the first ternary layer so you can see the float vs. ternary latency before you reach for hardware counters.
2. **Pack + unpack weights** — once you have weights or a `TernaryTensor`, call `t81lib.pack_dense_matrix` and `t81lib.unpack_packed_limbs` to recover the raw trits. Feed those trits into `TernaryEmulator.set_wire` (for single wires) or `ripple_adder` layers to tally how often each trit flips when an AVX/NEON-friendly GEMM runs on the host.
3. **Inspect the trace** — `emulator.energy_consumed`, `emulator.transition_counter`, and `emulator.power_trace` are simple sequences you can plot or log alongside the training/validation reports from `examples/ternary_qat_inference_comparison.py` so quantization loss and energy-per-step live in the same notebook.

## Example: scoring a small GEMM

```python
import torch
import t81lib
from t81.hardware import TernaryEmulator

weights = torch.randn(32, 32, dtype=torch.float32)
packed = t81lib.pack_dense_matrix(weights, threshold=0.45)
trits = t81lib.unpack_packed_limbs(packed, rows=packed.shape[0], cols=packed.shape[1])

emulator = TernaryEmulator(name="gemm-emulator", power_model={"ternary": 1.0, "binary": 0.6})
for row_idx, row in enumerate(trits):
    emulator.set_wire(f"gate-row-{row_idx}", int(row[0]))
    emulator.ripple_adder(row, row[::-1], name_prefix=f"sum-{row_idx}")
emulator.clock_tick()
print("energy", emulator.energy_consumed)
```

This snippet shows how to link the same packed matrix that feeds `t81lib.gemm_ternary` into a traceable circuit (the `ripple_adder` generates carries and forces fan-in that matches AVX-parallel mix). Swap in `t81_torch.TernaryTensor.from_float` when you only have a PyTorch weight matrix; you can then pass the tensor directly into the emulator by iterating its `limbs` buffer or by quantizing each row separately.

## See also

- `examples/ternary_hardware_sim_demo.ipynb` for a notebook-based emulator walkthrough covering balanced-adders, fuzzy thresholds, and gate-level power curves.
- `examples/ternary_qat_inference_comparison.py` for a QAT loop that logs validation metrics, compression ratios, and GEMM timings so you can overlay the emulator-derived energy on top of the same Python story.
- `scripts/quantize_measure.py` to chain `t81 convert`, `AutoModel.from_pretrained_t81`, `t81lib.pack_dense_matrix`, and inference latencies when you want to automate quantize→measure in a production pipeline.
- `scripts/quantize_energy_benchmark.py` for a higher-level pipeline that runs `t81 convert`, measures float/ternary latencies, and traces `TernaryEmulator` energy so you get a single CSV/JSON row with compression, latency, and power numbers.
