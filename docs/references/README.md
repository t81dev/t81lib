### GGUF Export (llama.cpp / Ollama / LM Studio)

```bash
t81-convert meta-llama/Llama-3.2-3B-Instruct llama3.2-3b-t81.gguf --quant TQ1_0
```

When writing a converted checkpoint or GGUF bundle, append `--force-cpu-device-map`
so that Accelerate keeps parameters on CPU/disk instead of dispatching them to
`meta`. The default `device_map="auto"` path can offload modules to disk and
triggers `NotImplementedError: Cannot copy out of meta tensor` when `save_pretrained`
runs. Using the new flag ensures everything stays serializable, and you can rerun
`t81-gguf`/`t81-convert` with it whenever you hit that error.
