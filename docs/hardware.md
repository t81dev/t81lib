# Hardware simulation

`T81.hardware.TernaryEmulator` lets you sketch ternary chips, evaluate fuzzy decisions, and estimate energy costs for AI edge deployments. It mirrors the core library by keeping wires as `t81lib.Limb` values so trit-level bookkeeping stays consistent with the rest of the stack.

Highlights include:

- `visualize_circuit()` for matplotlib-friendly diagrams with optional Graphviz exports so you can illustrate ternary datapaths or neuromorphic operators.
- Fuzzy helpers such as `fuzzy_and`, `fuzzy_decision`, and `fuzzy_not` so agents can reason about “false/maybe/true” beliefs before dispatching ternary or binary schedules.
- `simulate_torch_forward()` plus power-tracing hooks that tally trit flips, emulate hybrid forward passes, and let you compare ternary energy to binary energy budgets.

See `examples/ternary_hardware_sim_demo.ipynb` for a guided walkthrough that builds a ternary adder, runs a small inference, records virtual power/latency metrics, and highlights how balanced ternary drops switching energy for drones or tiny neuromorphic chips.
