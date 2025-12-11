# GGUF Support in t81lib

`t81lib` implements **bit-perfect** read/write support for GGUF files with balanced ternary quantization:

| Type   | bpw     | Status       | llama.cpp compatible |
|-------|---------|--------------|----------------------|
| TQ1_0 | ~1.625  | Fully supported | Yes                |
| TQ2_0 | ~2.0625 | Fully supported | Yes                |

### Usage

```bash
# Convert + export directly to GGUF
t81-convert meta-llama/Llama-3.2-3B-Instruct llama3.2-3b-t81.gguf \
  --quant TQ1_0 --threshold 0.45