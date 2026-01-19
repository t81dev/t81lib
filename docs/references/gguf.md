# GGUF Support in t81lib

`t81lib` implements **bit-perfect** read/write support for GGUF files with balanced ternary quantization:

| Type   | bpw     | Status       | llama.cpp compatible |
|-------|---------|--------------|----------------------|
| TQ1_0 | ~1.625  | Fully supported | Yes                |
| TQ2_0 | ~2.0625 | Fully supported | Yes                |

### Usage

```bash
# Convert + export directly to GGUF
t81 convert meta-llama/Llama-3.2-3B-Instruct llama3.2-3b-t81.gguf \
  --quant TQ1_0 --threshold 0.45
```

### Export profiles

For a no-knobs compression-first export, use the `compression-first` profile via the CLI (`--gguf-profile` or `--profile`). It stamps `t81.profile=compression-first` in metadata and pins the GGUF quant scheme to TQ1_0 for maximum compression.

### Experimental TQ1_1 profile

`tq1_1-draft` is available for header-size testing only. It requires `T81_ENABLE_TQ1_1=1` and writes payloads that are not yet loadable by llama.cpp, so use it for experiments rather than production GGUF bundles.

### Repacking + dequantizing existing GGUF files

`t81.gguf.repack_gguf` re-quantizes an existing GGUF file (float tensors only) and preserves the metadata, so you can take a float32 or float16 bundle and emit a ternary one without running the full conversion pipeline. For compatibility with runtimes that do not support ternary types, `t81.gguf.dequantize_gguf` (and the convenience `t81.dequantize_gguf_to_float`) converts TQ1_0/TQ2_0 payloads into float32 or float16 GGUF files.

If you need to inspect a GGUF without loading everything into RAM, `t81.gguf.read_gguf` streams metadata and tensor payloads from the file handle and can return raw bytes instead of dequantized tensors.
