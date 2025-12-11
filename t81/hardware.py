"""
t81.hardware — Balanced ternary hardware emulator and energy model used by ai researchers.
"""

from __future__ import annotations

import collections
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:
    from graphviz import Digraph

    GRAPHVIZ_AVAILABLE = True
except ImportError:
    Digraph = None  # type: ignore[assignment]
    GRAPHVIZ_AVAILABLE = False

import t81lib

TritValue = int
WireName = str


def _clamp_trit(value: Any) -> TritValue:
    """Clamp a numeric value to {-1, 0, 1} and coerce limbs to ints."""
    if isinstance(value, t81lib.Limb):
        try:
            numeric = int(value.to_int())
        except Exception:  # safe-fallback
            numeric = 0
        return max(-1, min(1, numeric))
    if isinstance(value, bool):
        return 1 if value else -1
    if isinstance(value, (int, float)):
        candidate = int(round(value))
        return max(-1, min(1, candidate))
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return 0
        return _clamp_trit(value.flat[0])
    raise ValueError(f"unable to coerce {value!r} into a ternary trit")


class TernaryEmulator:
    """Simple emulator for ternary logic with hybrid, fuzzy, and power-aware helpers."""

    def __init__(
        self,
        name: str = "ternary-chip",
        power_model: Optional[Mapping[str, float]] = None,
        fuzzy_threshold: float = 0.25,
    ) -> None:
        self.name = name
        self.mode = "ternary"
        self.wires: Dict[WireName, TritValue] = {}
        self.limb_states: Dict[WireName, t81lib.Limb] = {}
        self.gates: List[Dict[str, Any]] = []
        self.registers: Dict[WireName, TritValue] = {}
        self.register_inputs: Dict[WireName, TritValue] = {}
        self.history: List[Tuple[str, TritValue]] = []
        self.power_trace: List[float] = []
        self.transition_counter: collections.Counter[str] = collections.Counter()
        self.energy_consumed = 0.0
        self.fuzzy_threshold = fuzzy_threshold
        self.power_model = dict(power_model or {"ternary": 1.0, "binary": 0.6})

    def _estimate_energy(self, old: TritValue, new: TritValue) -> float:
        """Charge energy based on the number of trit switches and selected mode."""
        base = self.power_model.get(self.mode, 1.0)
        if old == new:
            return base * 0.05
        if old == 0 or new == 0:
            return base * 0.25
        return base * 0.45

    def _record_transition(self, wire: WireName, previous: TritValue, current: TritValue) -> None:
        cost = self._estimate_energy(previous, current)
        self.energy_consumed += cost
        self.power_trace.append(cost)
        self.transition_counter[wire] += 1
        self.history.append((wire, current))

    def set_wire(self, name: WireName, value: Any) -> None:
        """Drive a wire, tracking limbs and energy penalties."""
        trit = _clamp_trit(value)
        previous = self.wires.get(name, 0)
        self.wires[name] = trit
        self.limb_states[name] = t81lib.Limb.from_value(trit)
        self._record_transition(name, previous, trit)

    def add_gate(
        self,
        gate_type: str,
        output: WireName,
        inputs: Iterable[WireName],
        name: Optional[str] = None,
    ) -> None:
        """Add a combinational gate to the emulator."""
        gate = {
            "name": name or f"{gate_type}-{len(self.gates)}",
            "type": gate_type.upper(),
            "inputs": tuple(inputs),
            "output": output,
        }
        self.gates.append(gate)

    def evaluate_circuit(self, inputs: Mapping[WireName, Any]) -> Dict[WireName, TritValue]:
        """Run combinational logic using the previously registered gates."""
        for key, value in inputs.items():
            self.set_wire(key, value)
        for gate in self.gates:
            resolved_inputs = [self.wires.get(inp, 0) for inp in gate["inputs"]]
            result = self._evaluate_gate(gate["type"], resolved_inputs)
            self.set_wire(gate["output"], result)
        return {gate["output"]: self.wires[gate["output"]] for gate in self.gates}

    def _evaluate_gate(self, gate_type: str, inputs: Sequence[TritValue]) -> TritValue:
        gate_type = gate_type.upper()
        if gate_type == "AND":
            return min(inputs)
        if gate_type == "OR":
            return max(inputs)
        if gate_type == "NOT":
            if not inputs:
                return 0
            return -inputs[0]
        if gate_type == "MUX":
            if len(inputs) < 3:
                raise ValueError("MUX needs select, true, false inputs")
            sel, true_in, false_in = inputs[:3]
            if sel > 0:
                return true_in
            if sel < 0:
                return false_in
            return 0
        if gate_type == "XOR":
            return 1 if sum(inputs) == 1 else -1 if sum(inputs) == -1 else 0
        raise ValueError(f"unknown gate type {gate_type}")

    def add_flipflop(self, name: WireName, initial: Any = -1) -> None:
        """Create a simple D-style flip-flop (register) that latches on clock_tick."""
        stable = _clamp_trit(initial)
        self.registers[name] = stable
        self.register_inputs[name] = stable
        self.set_wire(name, stable)

    def feed_register(self, name: WireName, value: Any) -> None:
        """Stage a value for the next clock tick."""
        if name not in self.registers:
            raise KeyError(f"{name} is not registered")
        self.register_inputs[name] = _clamp_trit(value)

    def clock_tick(self) -> None:
        """Commit staged register values, emulating sequential state machines."""
        for name, value in self.register_inputs.items():
            self.registers[name] = value
            self.set_wire(name, value)

    def balanced_adder(
        self, operand_a: TritValue, operand_b: TritValue, carry_in: TritValue = 0
    ) -> Tuple[TritValue, TritValue]:
        """Single-trit balanced adder used within ripple_adder."""
        total = operand_a + operand_b + carry_in
        carry_out = 0
        if total > 1:
            total -= 3
            carry_out = 1
        elif total < -1:
            total += 3
            carry_out = -1
        return total, carry_out

    def ripple_adder(
        self, a_bits: Sequence[Any], b_bits: Sequence[Any], name_prefix: str = "sum"
    ) -> Dict[str, TritValue]:
        """Build and evaluate a ripple adder storing sums in wires for tracing."""
        max_len = max(len(a_bits), len(b_bits))
        carry = 0
        result: Dict[str, TritValue] = {}
        for index in range(max_len):
            a = _clamp_trit(a_bits[index]) if index < len(a_bits) else 0
            b = _clamp_trit(b_bits[index]) if index < len(b_bits) else 0
            total, carry = self.balanced_adder(a, b, carry)
            key = f"{name_prefix}_{index}"
            result[key] = total
            self.set_wire(key, total)
        if carry:
            result[f"{name_prefix}_carry"] = carry
            self.set_wire(f"{name_prefix}_carry", carry)
        return result

    def fuzzy_and(self, *values: Any) -> TritValue:
        """Treat 0 as uncertainty; prefer strict consensus when values align."""
        trits = [_clamp_trit(v) for v in values]
        if any(v == -1 for v in trits):
            return -1
        if any(v == 0 for v in trits):
            return 0
        return 1

    def fuzzy_or(self, *values: Any) -> TritValue:
        trits = [_clamp_trit(v) for v in values]
        if any(v == 1 for v in trits):
            return 1
        if any(v == 0 for v in trits):
            return 0
        return -1

    def fuzzy_not(self, value: Any) -> TritValue:
        trit = _clamp_trit(value)
        if trit == 0:
            return 0
        return -trit

    def fuzzy_decision(self, beliefs: Sequence[float]) -> TritValue:
        """Return a trit representing an agent's belief: -1 false, 0 maybe, 1 true."""
        if not beliefs:
            return 0
        avg = sum(beliefs) / len(beliefs)
        if avg >= self.fuzzy_threshold:
            return 1
        if avg <= -self.fuzzy_threshold:
            return -1
        return 0

    def hybrid_mode_decision(
        self, latency_ms: float, energy_budget: float, switch_margin: float = 0.2
    ) -> str:
        """Switch between binary and ternary modes for edge AI workloads."""
        if energy_budget < latency_ms * 0.02:
            self.mode = "binary"
        elif latency_ms < energy_budget * switch_margin:
            self.mode = "ternary"
        else:
            self.mode = "ternary"
        return self.mode

    def simulate_torch_forward(
        self,
        inputs: "torch.Tensor",
        weights: "torch.Tensor",
        bias: Optional["torch.Tensor"] = None,
        threshold: float = 0.4,
    ) -> "torch.Tensor":
        """Emulate a small ternary accelerator that replaces an inner product."""
        if torch is None:
            raise RuntimeError("PyTorch is required to run simulate_torch_forward")
        if inputs.dim() != 2 or weights.dim() != 2:
            raise ValueError("inputs and weights must be 2D tensors")
        if inputs.size(1) != weights.size(0):
            raise ValueError("input channels must match weight rows")
        device = inputs.device
        inputs_np = inputs.detach().to("cpu").numpy().astype(np.float32)
        weights_np = weights.detach().to("cpu").numpy().astype(np.float32)
        trits_in = t81lib.quantize_to_trits(inputs_np, threshold)
        trits_w = t81lib.quantize_to_trits(weights_np, threshold)
        active_switches = int(np.count_nonzero(trits_in)) + int(np.count_nonzero(trits_w))
        self.energy_consumed += active_switches * self.power_model.get(self.mode, 1.0) * 0.01
        accum = np.matmul(trits_in.astype(np.int16), trits_w.astype(np.int16))
        if bias is not None:
            bias_np = bias.detach().to("cpu").numpy().astype(np.float32)
            accum += bias_np
        clamped = np.clip(accum, -1, 1).astype(np.float32)
        outputs = torch.from_numpy(clamped).to(device)
        for idx, value in enumerate(clamped.flatten()):
            self.set_wire(f"infer_{idx}", value)
        return outputs

    def compare_with_binary(self, steps: int = 64) -> Dict[str, float]:
        """Simple benchmark: ternary energy vs binary for vector toggles."""
        ternary_energy = 0.0
        binary_energy = 0.0
        prev = -1
        for step in range(steps):
            current = -1 if step % 2 == 0 else 1
            ternary_energy += self._estimate_energy(prev, current)
            binary_energy += self.power_model.get("binary", 0.5) * (0.1 if prev == current else 0.3)
            prev = current
        return {"ternary": ternary_energy, "binary": binary_energy}

    def power_summary(self) -> Dict[str, Any]:
        """Return aggregate info about energy consumed during simulation."""
        total_events = len(self.power_trace)
        return {
            "mode": self.mode,
            "energy": self.energy_consumed,
            "events": total_events,
            "average": self.energy_consumed / total_events if total_events else 0.0,
            "transitions": dict(self.transition_counter),
        }

    def visualize_circuit(
        self,
        backend: str = "matplotlib",
        figsize: Tuple[int, int] = (10, 4),
        annotate_wires: bool = True,
    ) -> Any:
        """Draw the registered gates either with matplotlib or graphviz (if available)."""
        if backend == "graphviz" and GRAPHVIZ_AVAILABLE:
            dot = Digraph(name=self.name, format="svg")
            for gate in self.gates:
                dot.node(gate["name"], label=f'{gate["name"]}\n{gate["type"]}', shape="box")
                for inp in gate["inputs"]:
                    dot.edge(inp, gate["name"])
                dot.edge(gate["name"], gate["output"])
            return dot
        fig, ax = plt.subplots(figsize=figsize)
        if not self.gates:
            ax.text(0.5, 0.5, "no gates defined", ha="center", va="center")
            ax.axis("off")
            return fig
        ys = np.linspace(0, -len(self.gates), len(self.gates))
        for idx, gate in enumerate(self.gates):
            x = idx
            y = ys[idx]
            ax.scatter([x], [y], s=600, facecolors="lightcyan", edgecolors="navy")
            ax.text(x, y, f"{gate['name']}\n{gate['type']}", ha="center", va="center")
            for offset, name in enumerate(gate["inputs"]):
                y_input = y + ((offset - 0.5) * 0.3)
                ax.plot([x - 0.5, x], [y_input, y], color="gray")
                if annotate_wires:
                    ax.text(x - 0.5, y_input, name, ha="right", va="center", fontsize=8)
            if annotate_wires:
                ax.text(x + 0.2, y, gate["output"], ha="left", va="center", fontsize=9)
        ax.set_xticks([])  # Hide axis lines; purely illustrative
        ax.set_yticks([])
        ax.set_title(f"{self.name} combinational graph", fontsize=10)
        plt.tight_layout()
        return fig
