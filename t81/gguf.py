"""
GGUF helpers for writing and reading the llama.cpp TQ1_0/TQ2_0 tensors.

The module targets the 2025 BitNet-style balanced ternary layout (8 trits packed
into 13-bit blocks, shared scale per 32-row group, and 32-byte alignment) so the
resulting files can be consumed by unmodified llama.cpp builds.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from transformers import PreTrainedModel

import t81lib
from .nn import Linear as TernaryLinear

HEADER_STRUCT = struct.Struct("<4sIIIIQQQQ")
HEADER_MAGIC = b"GGUF"
HEADER_VERSION = 0x00000003
HEADER_ALIGNMENT = 32
HEADER_SIZE = HEADER_STRUCT.size

GGML_TYPE_TQ1_0 = 250
GGML_TYPE_TQ2_0 = 251

METADATA_KEY_TYPE_STRING = 3
METADATA_VALUE_UINT32 = 0
METADATA_VALUE_FLOAT32 = 1
METADATA_VALUE_BOOL = 2
METADATA_VALUE_STRING = 3

GGUF_QUANT_BLOCK_ROWS = 32


@dataclass(frozen=True)
class _TensorPayload:
    name: str
    shape: tuple[int, ...]
    ggml_type: int
    data: bytes


@dataclass(frozen=True)
class _TensorInfo:
    name: str
    shape: tuple[int, ...]
    ggml_type: int
    offset: int


def _align(length: int, alignment: int = HEADER_ALIGNMENT) -> int:
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    return (length + alignment - 1) // alignment * alignment


def _encode_metadata_entry(key: str, value: Any) -> bytes:
    if isinstance(value, str):
        payload = value.encode("utf-8") + b"\x00"
        value_type = METADATA_VALUE_STRING
    elif isinstance(value, bool):
        payload = struct.pack("<B", 1 if value else 0)
        value_type = METADATA_VALUE_BOOL
    elif isinstance(value, int):
        if value < 0:
            raise ValueError("metadata integer values must be non-negative")
        payload = struct.pack("<I", value)
        value_type = METADATA_VALUE_UINT32
    elif isinstance(value, float):
        payload = struct.pack("<f", value)
        value_type = METADATA_VALUE_FLOAT32
    else:
        raise ValueError(f"unsupported metadata value {value!r}")
    return (
        struct.pack("<I", METADATA_KEY_TYPE_STRING)
        + key.encode("utf-8")
        + b"\x00"
        + struct.pack("<I", value_type)
        + payload
    )


def _collect_metadata(model: PreTrainedModel, quant: str, threshold: float) -> tuple[bytes, int]:
    config = getattr(model, "config", None)
    architecture = None
    if config is not None:
        architecture = getattr(config, "architectures", None)
        if isinstance(architecture, Iterable):
            architecture = next(iter(architecture), None)
        architecture = architecture or getattr(config, "model_type", None)
    architecture = architecture or type(model).__name__
    name = getattr(model, "name_or_path", architecture) or architecture
    threshold = float(max(0.0, min(0.9999, threshold)))
    entries = [
        ("general.architecture", architecture),
        ("general.name", name),
        ("general.file_type", 2),
        ("general.alignment", HEADER_ALIGNMENT),
        ("general.quantized_by", "t81lib"),
        ("general.quantization_version", 2),
        ("quantization.type", quant.lower()),
        ("quantization.block_size", GGUF_QUANT_BLOCK_ROWS),
        ("quantization.threshold", threshold),
    ]
    buffer = bytearray()
    for key, value in entries:
        buffer.extend(_encode_metadata_entry(key, value))
    return bytes(buffer), len(entries)


def _tensor_info_length(name: str, shape: tuple[int, ...]) -> int:
    length = len(name.encode("utf-8")) + 1
    length += 4  # n_dims
    length += 4 * len(shape)
    length += 8 * len(shape)
    length += 4 + 8 + 4  # ggml_type + offset + reserved
    return _align(length)


def _serialize_tensor_info(name: str, shape: tuple[int, ...], ggml_type: int, offset: int) -> bytes:
    encoded = bytearray()
    encoded.extend(name.encode("utf-8"))
    encoded.append(0)
    encoded.extend(struct.pack("<I", len(shape)))
    for _ in shape:
        encoded.extend(struct.pack("<I", 0))
    for dim in shape:
        encoded.extend(struct.pack("<Q", dim))
    encoded.extend(struct.pack("<I", ggml_type))
    encoded.extend(struct.pack("<Q", offset))
    encoded.extend(struct.pack("<I", 0))
    padding = _align(len(encoded)) - len(encoded)
    if padding:
        encoded.extend(b"\x00" * padding)
    return bytes(encoded)


def _float_to_half_bytes(value: float) -> bytes:
    return np.float16(value).tobytes()


def _quantize_tensor(tensor: torch.Tensor, quant: str, threshold: float) -> bytes:
    array = tensor.cpu().to(dtype=torch.float32, copy=False).numpy()
    rows, cols = array.shape
    serialized = bytearray()
    for group_start in range(0, rows, GGUF_QUANT_BLOCK_ROWS):
        group = array[group_start : group_start + GGUF_QUANT_BLOCK_ROWS]
        scale = float(np.max(np.abs(group))) if group.size else 0.0
        serialized.extend(_float_to_half_bytes(scale))
        if quant == "TQ2_0":
            serialized.extend(b"\x00" * 8)
        for row in group:
            if cols == 0:
                continue
            packed = t81lib.quantize_row_tq1_0(np.asarray(row, dtype=np.float32), threshold, scale)[1]
            serialized.extend(packed.tobytes(order="C"))
    return bytes(serialized)


def _collect_linears(model: PreTrainedModel) -> list[tuple[str, torch.Tensor]]:
    seen: set[str] = set()
    result: list[tuple[str, torch.Tensor]] = []
    for module_name, module in model.named_modules():
        if isinstance(module, TernaryLinear):
            tensor_name = f"{module_name}.weight" if module_name else "weight"
            if tensor_name in seen:
                continue
            seen.add(tensor_name)
            result.append((tensor_name, module.weight.detach()))
    return result


def write_gguf(
    model: PreTrainedModel,
    path: str | Path,
    *,
    quant: str = "TQ1_0",
    threshold: float = 0.45,
) -> None:
    """
    Write a GGUF file from a converted model built from ``t81.nn.Linear`` modules.
    """
    quant = quant.upper()
    if quant not in {"TQ1_0", "TQ2_0"}:
        raise ValueError("quant must be one of 'TQ1_0' or 'TQ2_0'")
    entries = _collect_linears(model)
    if not entries:
        raise ValueError("model does not contain any t81.nn.Linear layers")

    tensor_payloads: list[_TensorPayload] = []
    for name, tensor in entries:
        data = _quantize_tensor(tensor, quant, threshold)
        tensor_payloads.append(
            _TensorPayload(
                name=name,
                shape=tuple(tensor.shape),
                ggml_type=GGML_TYPE_TQ1_0 if quant == "TQ1_0" else GGML_TYPE_TQ2_0,
                data=data,
            )
        )

    metadata_bytes, metadata_count = _collect_metadata(model, quant, threshold)
    metadata_section = metadata_bytes + b"\x00" * (_align(len(metadata_bytes)) - len(metadata_bytes))
    metadata_size = len(metadata_section)
    tensor_infos_offset = _align(HEADER_SIZE + metadata_size)

    infos_length = sum(_tensor_info_length(payload.name, payload.shape) for payload in tensor_payloads)
    tensor_infos_size = _align(infos_length)
    tensor_data_offset = _align(tensor_infos_offset + tensor_infos_size)

    serialized_infos = bytearray()
    tensor_data_section = bytearray()
    current_data_offset = tensor_data_offset
    for payload in tensor_payloads:
        serialized_infos.extend(
            _serialize_tensor_info(payload.name, payload.shape, payload.ggml_type, current_data_offset)
        )
        tensor_data_section.extend(payload.data)
        padding = _align(len(payload.data)) - len(payload.data)
        if padding:
            tensor_data_section.extend(b"\x00" * padding)
        current_data_offset += len(payload.data) + padding

    if len(serialized_infos) < tensor_infos_size:
        serialized_infos.extend(b"\x00" * (tensor_infos_size - len(serialized_infos)))

    header = HEADER_STRUCT.pack(
        HEADER_MAGIC,
        HEADER_VERSION,
        len(tensor_payloads),
        metadata_count,
        HEADER_ALIGNMENT,
        HEADER_ALIGNMENT,
        tensor_infos_offset,
        tensor_data_offset,
        0,
    )

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        handle.write(header)
        handle.write(metadata_section)
        metadata_padding = tensor_infos_offset - (HEADER_SIZE + metadata_size)
        if metadata_padding > 0:
            handle.write(b"\x00" * metadata_padding)
        handle.write(serialized_infos)
        infos_padding = tensor_data_offset - (tensor_infos_offset + len(serialized_infos))
        if infos_padding > 0:
            handle.write(b"\x00" * infos_padding)
        handle.write(tensor_data_section)


def _parse_metadata(buffer: bytes, offset: int, count: int) -> Mapping[str, Any]:
    metadata: dict[str, Any] = {}
    cursor = offset
    for _ in range(count):
        key_type = struct.unpack_from("<I", buffer, cursor)[0]
        cursor += 4
        if key_type != METADATA_KEY_TYPE_STRING:
            raise ValueError("unexpected metadata key type")
        key_end = buffer.index(0, cursor)
        key = buffer[cursor:key_end].decode("utf-8")
        cursor = key_end + 1
        value_type = struct.unpack_from("<I", buffer, cursor)[0]
        cursor += 4
        if value_type == METADATA_VALUE_UINT32:
            metadata[key] = struct.unpack_from("<I", buffer, cursor)[0]
            cursor += 4
        elif value_type == METADATA_VALUE_FLOAT32:
            metadata[key] = struct.unpack_from("<f", buffer, cursor)[0]
            cursor += 4
        elif value_type == METADATA_VALUE_BOOL:
            metadata[key] = bool(buffer[cursor])
            cursor += 1
        elif value_type == METADATA_VALUE_STRING:
            value_end = buffer.index(0, cursor)
            metadata[key] = buffer[cursor:value_end].decode("utf-8")
            cursor = value_end + 1
        else:
            raise ValueError(f"unsupported metadata value type {value_type}")
    return metadata


def _parse_tensor_infos(buffer: bytes, offset: int, count: int) -> list[_TensorInfo]:
    infos: list[_TensorInfo] = []
    cursor = offset
    for _ in range(count):
        start = cursor
        name_end = buffer.index(0, cursor)
        name = buffer[cursor:name_end].decode("utf-8")
        cursor = name_end + 1
        n_dims = struct.unpack_from("<I", buffer, cursor)[0]
        cursor += 4
        cursor += 4 * n_dims
        shape: list[int] = []
        for _ in range(n_dims):
            dim = struct.unpack_from("<Q", buffer, cursor)[0]
            cursor += 8
            shape.append(dim)
        ggml_type = struct.unpack_from("<I", buffer, cursor)[0]
        cursor += 4
        data_offset = struct.unpack_from("<Q", buffer, cursor)[0]
        cursor += 8
        cursor += 4
        length = cursor - start
        cursor = start + _align(length)
        infos.append(_TensorInfo(name, tuple(shape), ggml_type, data_offset))
    return infos


def _decode_quant_tensor(
    chunk: bytes,
    shape: Sequence[int],
    ggml_type: int,
    block_rows: int,
    dequantize: bool,
) -> torch.Tensor | bytes:
    rows = shape[0] if shape else 0
    cols = shape[1] if len(shape) > 1 else 1
    if not dequantize:
        return chunk
    decoder = t81lib.dequant_tq1_0 if ggml_type == GGML_TYPE_TQ1_0 else t81lib.dequant_tq2_0
    array = decoder(memoryview(chunk), rows, cols, block_rows)
    return torch.from_numpy(np.asarray(array, dtype=np.float32))


def read_gguf(path: str | Path, *, dequantize: bool = True) -> Mapping[str, torch.Tensor | bytes]:
    """
    Read a GGUF file. Quantized tensors are dequantized when requested.
    """
    buffer = Path(path).read_bytes()
    if len(buffer) < HEADER_SIZE:
        raise ValueError("file too short to be a valid GGUF blob")
    (
        magic,
        version,
        num_tensors,
        metadata_kv_count,
        _metadata_alignment,
        _tensor_alignment,
        tensor_infos_offset,
        tensor_data_offset,
    ) = HEADER_STRUCT.unpack_from(buffer, 0)
    if magic != HEADER_MAGIC or version != HEADER_VERSION:
        raise ValueError("GGUF header mismatch")

    metadata = _parse_metadata(buffer, HEADER_SIZE, metadata_kv_count)
    block_rows = int(metadata.get("quantization.block_size", GGUF_QUANT_BLOCK_ROWS))
    tensor_infos = _parse_tensor_infos(buffer, tensor_infos_offset, num_tensors)
    sorted_infos = sorted(tensor_infos, key=lambda info: info.offset)
    payload: dict[str, torch.Tensor | bytes] = {}
    for index, info in enumerate(sorted_infos):
        next_offset = (
            sorted_infos[index + 1].offset if index + 1 < len(sorted_infos) else len(buffer)
        )
        chunk = buffer[info.offset:next_offset]
        if info.ggml_type not in {GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0}:
            raise ValueError(f"unsupported tensor type {info.ggml_type}")
        decoded = _decode_quant_tensor(chunk, info.shape, info.ggml_type, block_rows, dequantize)
        payload[info.name] = decoded
    return payload
