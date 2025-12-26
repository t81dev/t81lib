#!/usr/bin/env python3
"""
Estimate per-tensor size savings for a GGUF file when moving TQ1_0 -> TQ1_1.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import struct

from t81 import gguf


def _tensor_nbytes(info: gguf._TensorInfo, tq1_block_bytes: int) -> int:
    if not info.shape:
        return 0
    if info.ggml_type == gguf.GGML_TYPE_F16:
        return int(np.prod(info.shape)) * 2
    if info.ggml_type == gguf.GGML_TYPE_F32:
        return int(np.prod(info.shape)) * 4
    if info.ggml_type in {gguf.GGML_TYPE_TQ1_0, gguf.GGML_TYPE_TQ1_1}:
        rows = info.shape[0]
        cols = info.shape[1] if len(info.shape) > 1 else 1
        blocks_per_row = (cols + gguf.GGUF_QUANT_BLOCK_ROWS - 1) // gguf.GGUF_QUANT_BLOCK_ROWS
        return rows * blocks_per_row * tq1_block_bytes
    if info.ggml_type == gguf.GGML_TYPE_TQ2_0:
        rows = info.shape[0]
        cols = info.shape[1] if len(info.shape) > 1 else 1
        blocks_per_row = (cols + gguf.GGUF_QUANT_BLOCK_ROWS - 1) // gguf.GGUF_QUANT_BLOCK_ROWS
        return rows * blocks_per_row * 66
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit GGUF tensor sizes for TQ1_1 savings.")
    parser.add_argument("--gguf", required=True, help="Path to the GGUF file.")
    parser.add_argument("--tq1-0-block-bytes", type=int, default=54)
    parser.add_argument("--tq1-1-block-bytes", type=int, default=53)
    args = parser.parse_args()

    gguf_path = Path(args.gguf)
    if not gguf_path.exists():
        raise SystemExit(f"GGUF path does not exist: {gguf_path}")

    with gguf_path.open("rb") as handle:
        header = handle.read(gguf.HEADER_SIZE)
        if len(header) < gguf.HEADER_SIZE:
            raise SystemExit("file too short to be a valid GGUF blob")
        magic, version, num_tensors, metadata_kv_count = gguf.HEADER_STRUCT.unpack_from(header, 0)
        if magic != gguf.HEADER_MAGIC or version != gguf.HEADER_VERSION:
            raise SystemExit("GGUF header mismatch")
        metadata, metadata_length = gguf._parse_metadata_from_file(handle, metadata_kv_count)
        alignment = int(metadata.get("general.alignment", gguf.HEADER_ALIGNMENT))
        metadata_end = handle.tell()
        peek = handle.read(8)
        if len(peek) != 8:
            raise SystemExit("metadata truncated before tensor infos")
        name_len = struct.unpack("<Q", peek)[0]
        handle.seek(-8, 1)
        if name_len == 0 or name_len > 4096:
            aligned_end = gguf.HEADER_SIZE + gguf._align(metadata_length, alignment)
            if aligned_end > metadata_end:
                handle.seek(aligned_end - metadata_end, 1)
        tensor_infos = gguf._parse_tensor_infos_from_file(handle, num_tensors)

    total_current = 0
    total_tq1_1 = 0
    print("name,ggml_type,current_bytes,tq1_1_bytes,delta_bytes")
    for info in tensor_infos:
        current_bytes = _tensor_nbytes(info, args.tq1_0_block_bytes)
        if info.ggml_type in {gguf.GGML_TYPE_TQ1_0, gguf.GGML_TYPE_TQ1_1}:
            tq1_bytes = _tensor_nbytes(info, args.tq1_1_block_bytes)
        else:
            tq1_bytes = current_bytes
        delta = tq1_bytes - current_bytes
        total_current += current_bytes
        total_tq1_1 += tq1_bytes
        print(f"{info.name},{info.ggml_type},{current_bytes},{tq1_bytes},{delta}")

    total_delta = total_tq1_1 - total_current
    print(f"TOTAL,,{total_current},{total_tq1_1},{total_delta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
