### Reference index

This folder collects lightweight guides for the console scripts and the GGUF
plugin surface. For a broader overview of the CLI helpers (`t81 convert`,
`t81 gguf`, and `t81-qat` plus the legacy `t81-convert`/`t81-gguf` aliases), see `cli-usage.md`.

### GGUF export (llama.cpp / Ollama / LM Studio)

```bash
t81 convert meta-llama/Llama-3.2-3B-Instruct llama3.2-3b-t81.gguf --quant TQ1_0
```

When writing a converted checkpoint or GGUF bundle, append `--force-cpu-device-map`
so that Accelerate keeps parameters on CPU/disk instead of dispatching them to
`meta`. The default `device_map="auto"` path can offload modules to disk and
triggers `NotImplementedError: Cannot copy out of meta tensor` when `save_pretrained`
runs. Using the new flag ensures everything stays serializable, and you can rerun
`t81 gguf`/`t81 convert` (or the legacy `t81-gguf`/`t81-convert` scripts) with it whenever you hit that error.
