<!--
docs/diagrams/cli-workflows-mermaid.md — Visual guide to the t81 CLI helper workflows.
-->

## CLI workflows

Each helper (`t81-convert`, `t81-gguf`, `t81-qat`) interfaces with Hugging Face + `transformers` to deliver ternary-ready bundles or quantization-aware checkpoints. The diagrams below highlight the entry points, where shared helpers are invoked, and what artifacts land on disk.

### `t81-convert`

```mermaid
graph TD
    HF["Hugging Face checkpoint (model + weights)"]
    CLI["t81-convert CLI (threshold, device-map, dtype, output-gguf, etc.)"]
    Convert["t81.convert.convert + tensor pass"]
    Rewriter["Rewrite every nn.Linear → t81.nn.Linear"]
    ForceCPU["Optional --force-cpu-device-map (keep on host)"]
    Metadata["t81_metadata.json + stats + linear metadata"]
    Output["Converted dir (ternary tensors, biases, helpers)"]
    GGUFOut["Optional gguf.write_gguf (if --output-gguf)"]

    HF --> Convert
    CLI --> Convert
    Convert --> Rewriter --> Output
    Convert --> Metadata
    CLI --> ForceCPU --> Convert
    CLI --> GGUFOut
    Convert --> GGUFOut
```

### `t81-gguf`

```mermaid
graph TD
    FromHF["--from-hf meta-llama/... model path"]
    FromT81["--from-t81 path/to/converted"]
    CLI["t81-gguf CLI (threshold, dtype, force-cpu-device-map)"]
    Convert["(internal) t81-convert + tensor prep"]
    Metadata["Reuse t81_metadata.json if present"]
    Writer["gguf.write_gguf"]
    GGUF["LLM-friendly GGUF bundle"]

    FromHF --> Convert
    CLI --> Convert
    FromT81 --> Metadata --> Writer
    Convert --> Metadata --> Writer
    Writer --> GGUF
```

### `t81-qat`

```mermaid
graph TD
    Model["Hugging Face model + config"]
    Dataset["datasets + transformers source"]
    CLI["t81-qat CLI (per-device batch size, LR, ternary flags)"]
    Trainer["Quantization-aware trainer (t81.trainer + torch)"]
    Schedules["Ternary threshold, stochastic rounding, warmup"]
    Output["Output dir with QAT checkpoints + logs"]

    Model --> Trainer
    Dataset --> Trainer
    CLI --> Trainer
    CLI --> Schedules --> Trainer
    Trainer --> Output
```
