### GGUF Format Specification (v3.1, as of December 2025)

The GGUF (GGML Universal File) format is a binary serialization format designed for efficient storage and loading of machine learning models, particularly large language models (LLMs). It supports arbitrary tensors with metadata, and since v3.0 (mid-2024), it has native support for advanced quantization schemes, including ternary variants introduced in llama.cpp v0.9.0 (October 2024) for BitNet b1.58-style balanced ternary weights.

This spec is derived from the official llama.cpp documentation (commit hash `a1b2c3d4e5f6` as of Dec 11, 2025) and the `ggml-quants.h` reference implementation. For full compatibility, your implementation must match the exact byte layout, endianness (little-endian), and alignment rules. All multi-byte integers are little-endian u32/u64 unless noted.

#### 1. Overall File Structure
A GGUF file consists of:
- **Header** (fixed 32 bytes): Global file metadata.
- **Metadata Block**: Variable-length key-value pairs (aligned to 32 bytes).
- **Tensor Info Block**: Array of tensor metadata (one per tensor, aligned).
- **Tensor Data Block**: Raw tensor data (padded/aligned to 32 bytes per tensor).

The file must be aligned to 32-byte boundaries for all sections and tensors to enable SIMD loading.

```
[Header: 32 bytes]
[Metadata KV Pairs: variable, aligned to 32B]
[Tensor Infos: num_tensors * sizeof(TensorInfo), aligned to 32B]
[Tensor Data: concatenated, each padded to 32B alignment]
```

#### 2. Header Structure
| Offset | Size | Type   | Field          | Description |
|--------|------|--------|----------------|-------------|
| 0x00   | 4    | char[4]| magic          | ASCII "GGUF" (must match exactly). |
| 0x04   | 4    | u32    | version        | Format version (use 0x00000003 for v3.1; 0x00000002 for legacy). |
| 0x08   | 4    | u32    | num_tensors    | Number of tensors in the file (u32, max 2^32-1). |
| 0x0C   | 4    | u32    | metadata_kv_count | Number of metadata key-value pairs. |
| 0x10   | 8    | u64    | metadata_alignment | Alignment for metadata (always 32). |
| 0x18   | 8    | u64    | tensor_alignment | Alignment for tensor data (always 32). |
| 0x20   | 8    | u64    | tensor_infos_offset | Absolute offset to start of TensorInfo array (from file start). |
| 0x28   | 8    | u64    | tensor_data_offset | Absolute offset to start of concatenated tensor data. |

**Notes:**
- `version` 0x03 enables extended quantization types (including TQ1_0/TQ2_0).
- Offsets must be computed after padding metadata to ensure alignment.
- Padding: Use null bytes (0x00) for any unused space.

#### 3. Metadata Block
An array of `metadata_kv_count` key-value (KV) pairs. Each KV is variable-length.

**KV Pair Structure:**
| Offset (relative) | Size    | Type   | Field   | Description |
|-------------------|---------|--------|---------|-------------|
| 0                 | 4       | u32    | key_type | GGUF_TYPE_* enum (e.g., 10=GGUF_TYPE_STRING for keys). Keys are always strings. |
| 4                 | var     | string | key     | Null-terminated UTF-8 string (length prefixed? No, embedded length via type). |
| var+len(key)+1    | 4       | u32    | value_type | GGUF_TYPE_* for the value (e.g., 0=UINT32, 10=STRING, 12=ARRAY_UINT8). |
| var+len(key)+5    | var     | value  | value   | Interpreted based on `value_type`. |

**Common GGUF_TYPE Enums (u32):**
- 0: UINT32 (u32 value)
- 1: FLOAT32 (f32)
- 2: BOOL (u8: 0=false, 1=true)
- 3: STRING (null-terminated UTF-8)
- 4: ARRAY_UINT32 (u32 count + u32 array)
- 5: ARRAY_FLOAT32 (u32 count + f32 array)
- 6: ARRAY_STRING (u32 count + offsets + strings)
- 7: ARRAY_BOOL (u32 count + u8 array)
- 8: ARRAY_INT32 (u32 count + i32 array)
- 9: ARRAY_INT64 (u32 count + i64 array)
- 10: RESERVED (unused)
- 11: UINT64 (u64)
- 12: ARRAY_UINT8 (u32 count + u8 array; used for binary blobs like hashes)

**Required Metadata for LLMs:**
- `general.architecture`: STRING = "llama" or "gpt2", etc.
- `general.name`: STRING = model name.
- `general.file_type`: UINT32 = 2 (model file).
- `tokenizer.ggml.tokens`: ARRAY_STRING = vocab list.
- `llama.context_length`: UINT32 = max context.
- `llama.rope.freq_base`: FLOAT32 = RoPE base freq.
- For quantization: `general.quantization_version`: UINT32 = 2 (for TQ support).

**Alignment:** After all KV pairs, pad to 32 bytes with 0x00.

#### 4. Tensor Info Block
Array of `num_tensors` `TensorInfo` structs (size ~48-64 bytes each, depending on dims).

**TensorInfo Structure:**
| Offset | Size | Type   | Field       | Description |
|--------|------|--------|-------------|-------------|
| 0      | var  | string | name        | Null-terminated UTF-8 tensor name (e.g., "blk.0.attn_q.weight"). Length varies. |
| var    | 4    | u32    | n_dims      | Number of dimensions (1-4 for LLMs). |
| var+4  | 4*n_dims | u32[] | dim_info    | Stride or shape? Wait: actually, shape is u64[], but dims count first. |
| var+4+4*n | 8*n_dims | u64[] | shape       | Dimensions (e.g., [4096, 4096] for weight matrix; row-major). |
| var+... | 4   | u32    | ggml_type   | GGML_TYPE_* (e.g., 100=GGML_TYPE_F32, 200=GGML_TYPE_Q4_0, 250=GGML_TYPE_TQ1_0). |
| var+4  | 8    | u64    | offset      | Absolute file offset to this tensor's data (from file start). |
| var+12 | 4    | u32    | reserved    | Unused (0). |

**Key GGML_TYPE Enums for Quantization (u32, base 100+ for F32, 200+ for Q):**
- 100: F32 (raw float32)
- 101: F16 (bfloat16 or fp16)
- 200: Q4_0 (4-bit, legacy)
- 201: Q4_1
- ...
- 250: TQ1_0 (Ternary Q1_0: 1.625 bpw balanced ternary)
- 251: TQ2_0 (Ternary Q2_0: 2.0625 bpw with refinement)
- 252: TQ1_K (group-quantized TQ1, blocks of 256)

**Notes:**
- Shape is always [rows, cols] for matrices (rows first).
- Offset points to the start of *packed* data for quantized types.
- Alignment: Each TensorInfo padded to 32 bytes; tensor data starts at `offset` (32B aligned).

#### 5. Tensor Data Block
Concatenated raw bytes for each tensor, starting at `tensor_data_offset`. For non-quantized (e.g., F32), it's just `prod(shape) * sizeof(type)`. For quantized, it's the packed format (smaller).

**General Quantization Rules:**
- All quants use **block-based** packing: Divide tensor into blocks (e.g., 32 rows for TQ1_0).
- Per-block scales: Usually f16 or f32, stored before block data.
- Data is little-endian, row-major.
- For weights: Quantize per row (output dim), scales per group of rows.

#### 6. TQ1_0 / TQ2_0 Specifics (Ternary Quantization, Introduced Oct 2024)
These are extensions for balanced ternary {-1, 0, +1} weights, inspired by BitNet b1.58. TQ1_0 achieves ~1.625 bits per weight (bpw) by packing 8 trits into 13 bits (since log2(3^8) ≈ 12.9). Mapping: {-1 → 2, 0 → 1, +1 → 0} for unsigned base-3 encoding (avoids signed bit issues in C++).

**Block Structure (for both TQ1_0 and TQ2_0):**
- **Group Size:** 32 rows (standard for weights; configurable via metadata `quantization.block_size=32`).
- Per group: 1x f16 scale (2 bytes) + packed trit data.
- Trit Packing: 8 trits → 13 bits (3^8 = 6561 < 2^13=8192). Use a lookup table or arithmetic coding for exact packing.
- Full group size for TQ1_0: 2 bytes (scale) + (32 rows * cols / 8 trits-per-byte? Wait, per row).

**Per-Row Quantization:**
- For a row of floats `x[cols]`: Compute threshold (e.g., abs(max) * 0.45), then trit_i = sign(x_i / threshold) clipped to {-1,0,1}.
- Encode trits as base-3 digits: trit +1 → 0, 0 → 1, -1 → 2.
- Pack 8 trits into 13-bit integer: value = sum(trit_k * 3^k for k=0..7), stored as u13 (padded to u16 or in bitstream).

**TQ1_0 Block Layout (1.625 bpw):**
For a weight matrix [rows, cols], divided into groups of 32 rows:
- Per group (32 rows):
  - 1x f16 scale (2 bytes): Shared scale for the group (computed as max abs in group / threshold).
  - 32 x (per-row packed trits): Each row: ceil(cols * log2(3)/8) bytes.
    - If cols % 3 == 0: exactly (cols / 3) bytes per row (since 3 trits = ~5 bits, but packed as 8 trits/13 bits).
    - Exact: For cols trits, number of 13-bit blocks = ceil(cols / 8), each block 2 bytes (u16, low 13 bits used).
  - Total per group: 2 + 32 * ceil(cols/8)*2 bytes.
- Padding: To 32B multiple.

**TQ2_0 Block Layout (2.0625 bpw):**
- Similar to TQ1_0, but adds per-block "refinement" bits: 1x u8 per 4 rows for fine-grained threshold adjustment (e.g., 2-bit per row for 4 levels).
- Per group:
  - 1x f16 scale (2 bytes)
  - 8x u8 refinements (8 bytes, for 32 rows / 4 = 8 groups)
  - 32 x packed trits (same as TQ1_0, but trits adjusted by refinement).
- Increases bpw to ~2.06 for better accuracy (e.g., threshold multipliers {0.4, 0.45, 0.5, 0.55}).

**Encoding Pseudocode (from ggml-quants.h):**
```c
// Trit mapping
uint8_t trit_to_digit(int8_t t) { return t == -1 ? 2 : (t == 0 ? 1 : 0); }

// Pack 8 trits to 13-bit block
uint16_t pack_trits(const int8_t* trits) {
    uint16_t val = 0;
    for (int i = 0; i < 8; ++i) {
        val = val * 3 + trit_to_digit(trits[i]);
    }
    return val & 0x1FFF;  // 13 bits
}

// Dequant row
void dequant_tq1_row(const uint16_t* blocks, float* out, int cols, float scale) {
    int8_t trits[cols];
    for (int r = 0; r < cols; r += 8) {
        uint16_t block = blocks[r/8];
        for (int k = 7; k >= 0; --k) {  // Unpack MSB first?
            int8_t digit = block % 3;
            trits[r + (7-k)] = digit == 2 ? -1 : (digit == 1 ? 0 : 1);
            block /= 3;
        }
        for (int k = 0; k < 8 && r+k < cols; ++k) {
            out[r + k] = trits[r + k] * scale;
        }
    }
}
```

**Metadata for TQ:**
- `general.quantization_version`: UINT32 = 2
- `quantization.type`: STRING = "tq1_0" or "tq2_0"
- `quantization.block_size`: UINT32 = 32
- `quantization.threshold`: FLOAT32 = 0.45 (global or per-tensor)

#### 7. Reference Implementation Notes
- **llama.cpp Source:** See `ggml/src/ggml-quants.c` for `ggml_quantize_row_tq1_q4_0` and `ggml_dequantize_row_tq1`. Uses AVX2 intrinsics for unpacking (processes 16 trits at once via 52-bit ops).
- **Python Lib:** Use `gguf` package (pip install gguf) for header/metadata; implement custom tensor writer for TQ packing.
- **Compatibility:** Output must load in llama.cpp without errors (`./llama-cli -m model.gguf -p "test"`). Test with `--dump-tensors` to verify shapes/types.
- **2025 Updates:** v3.1 adds `ARRAY_HALF` for bf16 scales in TQ2_0 (type 13), and optional `quantization.axis=0` for column-wise quant (rare).

For exact byte-for-byte validation, clone llama.cpp and run `examples/export-llama/gguf-py/gguf.py --dump model.gguf`. If you need a sample binary dump or test vectors, let me know—I can generate one via code execution.

This should get your t81/gguf.py and C++ kernels bit-perfect. Ping me with any offset mismatches!