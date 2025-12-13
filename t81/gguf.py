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

HEADER_MAGIC = b"GGUF"
HEADER_VERSION = 0x00000003
HEADER_ALIGNMENT = 32
HEADER_STRUCT = struct.Struct("<4sIQQ")
HEADER_SIZE = HEADER_STRUCT.size

GGML_TYPE_TQ1_0 = 250
GGML_TYPE_TQ2_0 = 251
GGML_TYPE_F32 = 100
GGML_TYPE_F16 = 101

GGML_TYPE_F32 = 100

GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

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
    key_bytes = key.encode("utf-8")
    payload = struct.pack("<Q", len(key_bytes)) + key_bytes
    if isinstance(value, str):
        value_bytes = value.encode("utf-8")
        value_type = GGUF_TYPE_STRING
        payload += struct.pack("<i", value_type)
        payload += struct.pack("<Q", len(value_bytes)) + value_bytes
        return payload
    elif isinstance(value, bool):
        value_type = GGUF_TYPE_BOOL
        payload += struct.pack("<i", value_type)
        payload += struct.pack("<b", 1 if value else 0)
        return payload
    elif isinstance(value, int):
        if value < 0:
            raise ValueError("metadata integer values must be non-negative")
        value_type = GGUF_TYPE_UINT32
        payload += struct.pack("<i", value_type)
        payload += struct.pack("<I", value)
        return payload
    elif isinstance(value, float):
        value_type = GGUF_TYPE_FLOAT32
        payload += struct.pack("<i", value_type)
        payload += struct.pack("<f", value)
        return payload
    else:
        raise ValueError(f"unsupported metadata value {value!r}")


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


def _tensor_info_length(name: str, shape: tuple[int, ...], alignment: int) -> int:
    encoded_name = name.encode("utf-8")
    length = 8 + len(encoded_name)  # name length (u64) + bytes
    length += 4  # n_dims
    length += 8 * len(shape)  # dims
    length += 4  # ggml_type
    length += 8  # offset
    return _align(length, alignment)


def _serialize_tensor_info(
    name: str,
    shape: tuple[int, ...],
    ggml_type: int,
    offset: int,
    alignment: int,
) -> bytes:
    encoded = bytearray()
    name_bytes = name.encode("utf-8")
    encoded.extend(struct.pack("<Q", len(name_bytes)))
    encoded.extend(name_bytes)
    encoded.extend(struct.pack("<I", len(shape)))
    for dim in shape:
        encoded.extend(struct.pack("<Q", dim))
    encoded.extend(struct.pack("<i", ggml_type))
    encoded.extend(struct.pack("<Q", offset))
    padding = _align(len(encoded), alignment) - len(encoded)
    if padding:
        encoded.extend(b"\x00" * padding)
    return bytes(encoded)


def _serialize_metadata_from_mapping(metadata: Mapping[str, Any]) -> tuple[bytes, int]:
    buffer = bytearray()
    for key, value in metadata.items():
        buffer.extend(_encode_metadata_entry(key, value))
    return bytes(buffer), len(metadata)


def _write_payloads(
    payloads: list[_TensorPayload],
    metadata_bytes: bytes,
    metadata_count: int,
    alignment: int,
    path: Path,
) -> None:
    metadata_padding = _align(len(metadata_bytes), alignment) - len(metadata_bytes)
    metadata_section = metadata_bytes + b"\x00" * metadata_padding

    metadata_size = len(metadata_section)
    tensor_infos_offset = HEADER_SIZE + metadata_size

    infos_length = sum(_tensor_info_length(payload.name, payload.shape, alignment) for payload in payloads)
    tensor_infos_size = _align(infos_length, alignment)
    tensor_data_offset = _align(tensor_infos_offset + tensor_infos_size, alignment)

    serialized_infos = bytearray()
    tensor_data_section = bytearray()
    current_data_offset = tensor_data_offset
    for payload in payloads:
        serialized_infos.extend(
            _serialize_tensor_info(payload.name, payload.shape, payload.ggml_type, current_data_offset, alignment)
        )
        tensor_data_section.extend(payload.data)
        padding = _align(len(payload.data), alignment) - len(payload.data)
        if padding:
            tensor_data_section.extend(b"\x00" * padding)
        current_data_offset += len(payload.data) + padding

    if len(serialized_infos) < tensor_infos_size:
        serialized_infos.extend(b"\x00" * (tensor_infos_size - len(serialized_infos)))

    header = HEADER_STRUCT.pack(
        HEADER_MAGIC,
        HEADER_VERSION,
        len(payloads),
        metadata_count,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(metadata_section)
        handle.write(serialized_infos)
        infos_padding = tensor_data_offset - (tensor_infos_offset + len(serialized_infos))
        if infos_padding > 0:
            handle.write(b"\x00" * infos_padding)
        handle.write(tensor_data_section)


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

    alignment = HEADER_ALIGNMENT
    metadata_bytes, metadata_count = _collect_metadata(model, quant, threshold)
    _write_payloads(tensor_payloads, metadata_bytes, metadata_count, alignment, Path(path))


def _parse_metadata(
    buffer: bytes,
    offset: int,
    count: int,
) -> tuple[Mapping[str, Any], int]:
    metadata: dict[str, Any] = {}
    cursor = offset
    for _ in range(count):
        if cursor + 8 > len(buffer):
            raise ValueError("metadata truncated")
        key_len = struct.unpack_from("<Q", buffer, cursor)[0]
        cursor += 8
        key = buffer[cursor : cursor + key_len].decode("utf-8")
        cursor += key_len
        if cursor + 4 > len(buffer):
            raise ValueError("metadata truncated")
        value_type = struct.unpack_from("<i", buffer, cursor)[0]
        cursor += 4
        if value_type == GGUF_TYPE_UINT32:
            metadata[key] = struct.unpack_from("<I", buffer, cursor)[0]
            cursor += 4
        elif value_type == GGUF_TYPE_FLOAT32:
            metadata[key] = struct.unpack_from("<f", buffer, cursor)[0]
            cursor += 4
        elif value_type == GGUF_TYPE_BOOL:
            metadata[key] = bool(struct.unpack_from("<b", buffer, cursor)[0])
            cursor += 1
        elif value_type == GGUF_TYPE_STRING:
            if cursor + 8 > len(buffer):
                raise ValueError("metadata truncated")
            value_len = struct.unpack_from("<Q", buffer, cursor)[0]
            cursor += 8
            metadata[key] = buffer[cursor : cursor + value_len].decode("utf-8")
            cursor += value_len
        else:
            raise ValueError(f"unsupported metadata value type {value_type}")
    return metadata, cursor


def _parse_tensor_infos(
    buffer: bytes,
    offset: int,
    count: int,
    alignment: int,
) -> list[_TensorInfo]:
    infos: list[_TensorInfo] = []
    cursor = offset
    for _ in range(count):
        start = cursor
        if cursor + 8 > len(buffer):
            raise ValueError("tensor info truncated")
        name_len = struct.unpack_from("<Q", buffer, cursor)[0]
        cursor += 8
        name = buffer[cursor : cursor + name_len].decode("utf-8")
        cursor += name_len
        n_dims = struct.unpack_from("<I", buffer, cursor)[0]
        cursor += 4
        shape: list[int] = []
        for _ in range(n_dims):
            dim = struct.unpack_from("<Q", buffer, cursor)[0]
            cursor += 8
            shape.append(dim)
        ggml_type = struct.unpack_from("<i", buffer, cursor)[0]
        cursor += 4
        data_offset = struct.unpack_from("<Q", buffer, cursor)[0]
        cursor += 8
        length = cursor - start
        cursor = start + _align(length, alignment)
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
    if ggml_type == GGML_TYPE_F32:
        dtype = np.float32
        data = np.frombuffer(memoryview(chunk), dtype=dtype)
        if shape:
            data = data.reshape(shape)
        return torch.from_numpy(data)
    decoder = t81lib.dequant_tq1_0 if ggml_type == GGML_TYPE_TQ1_0 else t81lib.dequant_tq2_0
    array = decoder(memoryview(chunk), rows, cols, block_rows)
    return torch.from_numpy(np.asarray(array, dtype=np.float32))


def read_gguf(
    path: str | Path,
    *,
    dequantize: bool = True,
    return_metadata: bool = False,
) -> Mapping[str, torch.Tensor | bytes] | tuple[Mapping[str, torch.Tensor | bytes], Mapping[str, Any]]:
    """
    Read a GGUF file. Quantized tensors are dequantized when requested.
    """
    buffer = Path(path).read_bytes()
    if len(buffer) < HEADER_SIZE:
        raise ValueError("file too short to be a valid GGUF blob")
    magic, version, num_tensors, metadata_kv_count = HEADER_STRUCT.unpack_from(buffer, 0)
    if magic != HEADER_MAGIC or version != HEADER_VERSION:
        raise ValueError("GGUF header mismatch")

    metadata, metadata_end = _parse_metadata(buffer, HEADER_SIZE, metadata_kv_count)
    metadata_length = metadata_end - HEADER_SIZE
    alignment = int(metadata.get("general.alignment", HEADER_ALIGNMENT))
    if alignment <= 0:
        raise ValueError("invalid GGUF alignment")
    tensor_infos_offset = HEADER_SIZE + _align(metadata_length, alignment)
    block_rows = int(metadata.get("quantization.block_size", GGUF_QUANT_BLOCK_ROWS))
    tensor_infos = _parse_tensor_infos(buffer, tensor_infos_offset, num_tensors, alignment)
    sorted_infos = sorted(tensor_infos, key=lambda info: info.offset)
    payload: dict[str, torch.Tensor | bytes] = {}
    for index, info in enumerate(sorted_infos):
        next_offset = (
            sorted_infos[index + 1].offset if index + 1 < len(sorted_infos) else len(buffer)
        )
        chunk = buffer[info.offset:next_offset]
        if info.ggml_type not in {GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0, GGML_TYPE_F32}:
            raise ValueError(f"unsupported tensor type {info.ggml_type}")
        decoded = _decode_quant_tensor(chunk, info.shape, info.ggml_type, block_rows, dequantize)
        payload[info.name] = decoded
    if return_metadata:
        return payload, metadata
    return payload


def dequantize_gguf(
    source: str | Path,
    destination: str | Path,
    *,
    dtype: np.dtype = np.float32,
    ggml_type: int = GGML_TYPE_F32,
) -> None:
    """
    Dequantize a TQ1_0/TQ2_0 GGUF and write a float-compatible bundle of the given dtype.
    """
    payload, metadata = read_gguf(source, dequantize=True, return_metadata=True)
    metadata_bytes, metadata_count = _serialize_metadata_from_mapping(metadata)
    alignment = int(metadata.get("general.alignment", HEADER_ALIGNMENT))
    tensor_payloads: list[_TensorPayload] = []
    for name, tensor in payload.items():
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("expected torch.Tensor while dequantizing GGUF")
        array = torch.as_tensor(tensor, dtype=torch.float32, device="cpu")
        if dtype == np.float16:
            array = array.to(dtype=torch.float16)
            numpy_dtype = np.float16
        elif dtype == np.float32:
            numpy_dtype = np.float32
        else:
            raise ValueError(f"unsupported target dtype {dtype}")
        numpy_array = array.cpu().numpy(dtype=numpy_dtype, copy=False)
        tensor_payloads.append(
            _TensorPayload(
                name=name,
                shape=tuple(numpy_array.shape),
                ggml_type=ggml_type,
                data=numpy_array.tobytes(order="C"),
            )
        )
    _write_payloads(tensor_payloads, metadata_bytes, metadata_count, alignment, Path(destination))


def dequantize_gguf_to_float(source: str | Path, destination: str | Path) -> None:
    """Helper to keep the existing float32-specific API."""
    dequantize_gguf(source, destination, dtype=np.float32, ggml_type=GGML_TYPE_F32)
