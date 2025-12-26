"""
GGUF helpers for writing and reading the llama.cpp TQ1_0/TQ2_0 tensors.

The module targets the 2025 BitNet-style balanced ternary layout (8 trits packed
into 13-bit blocks, shared scale per 32-row group, and 32-byte alignment) so the
resulting files can be consumed by unmodified llama.cpp builds.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import struct
from typing import Any, BinaryIO, Iterable, Mapping, Sequence

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

GGML_TYPE_TQ1_0 = 34
GGML_TYPE_TQ2_0 = 35
GGML_TYPE_TQ1_1 = 38
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1

GGUF_PROFILE_COMPRESSION_FIRST = "compression-first"
GGUF_PROFILE_TQ1_1_DRAFT = "tq1_1-draft"

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

GGUF_QUANT_BLOCK_ROWS = 256  # ggml TQ block size
TQ1_TRITS_PER_BLOCK = 8
TQ2_TRITS_PER_BYTE = 4
TQ1_1_SCALE_BIAS = 7
TQ1_1_SCALE_BITS = 3


@dataclass(frozen=True)
class GGUFExportProfile:
    name: str
    quant: str
    threshold: float
    metadata: Mapping[str, Any]


GGUF_EXPORT_PROFILES: dict[str, GGUFExportProfile] = {
    GGUF_PROFILE_COMPRESSION_FIRST: GGUFExportProfile(
        name=GGUF_PROFILE_COMPRESSION_FIRST,
        quant="TQ1_0",
        threshold=0.45,
        metadata={
            "t81.profile": GGUF_PROFILE_COMPRESSION_FIRST,
            "t81.profile.intent": "compression",
            "t81.profile.quant": "tq1_0",
            "t81.profile.bpw": 1.625,
        },
    ),
    GGUF_PROFILE_TQ1_1_DRAFT: GGUFExportProfile(
        name=GGUF_PROFILE_TQ1_1_DRAFT,
        quant="TQ1_0",
        threshold=0.45,
        metadata={
            "t81.profile": GGUF_PROFILE_TQ1_1_DRAFT,
            "t81.profile.intent": "header-hygiene",
            "t81.profile.layout": "tq1_1-draft",
            "t81.profile.scale_format": "fp8-e4m3",
            "t81.profile.scale_cadence_blocks": 1,
            "t81.profile.scale_shared_exponent": False,
            "t81.profile.block_bytes_tq1_0": 54,
            "t81.profile.block_bytes_tq1_1_est": 53,
        },
    ),
}


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


def _encode_metadata_value(value: Any) -> tuple[int, bytes]:
    if isinstance(value, str):
        value_bytes = value.encode("utf-8")
        return GGUF_TYPE_STRING, struct.pack("<Q", len(value_bytes)) + value_bytes
    if isinstance(value, bool):
        return GGUF_TYPE_BOOL, struct.pack("<b", 1 if value else 0)
    if isinstance(value, int):
        if value < 0:
            return GGUF_TYPE_INT32, struct.pack("<i", value)
        return GGUF_TYPE_UINT32, struct.pack("<I", value)
    if isinstance(value, float):
        return GGUF_TYPE_FLOAT32, struct.pack("<f", value)
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("metadata arrays must be non-empty")
        first = value[0]
        element_type, _ = _encode_metadata_value(first)
        element_payloads = []
        for entry in value:
            entry_type, entry_payload = _encode_metadata_value(entry)
            if entry_type != element_type:
                raise ValueError("metadata arrays must use a single element type")
            element_payloads.append(entry_payload)
        payload = struct.pack("<i", element_type)
        payload += struct.pack("<Q", len(value))
        payload += b"".join(element_payloads)
        return GGUF_TYPE_ARRAY, payload
    raise ValueError(f"unsupported metadata value {value!r}")


def _encode_metadata_entry(key: str, value: Any) -> bytes:
    key_bytes = key.encode("utf-8")
    payload = struct.pack("<Q", len(key_bytes)) + key_bytes
    if key == "tokenizer.ggml.token_type" and isinstance(value, (list, tuple)):
        payload += struct.pack("<i", GGUF_TYPE_ARRAY)
        payload += struct.pack("<i", GGUF_TYPE_INT32)
        payload += struct.pack("<Q", len(value))
        payload += b"".join(struct.pack("<i", int(entry)) for entry in value)
        return payload
    value_type, value_payload = _encode_metadata_value(value)
    payload += struct.pack("<i", value_type)
    payload += value_payload
    return payload


def _collect_metadata(
    model: PreTrainedModel,
    quant: str,
    threshold: float,
    extra_metadata: Mapping[str, Any] | None = None,
) -> tuple[bytes, int]:
    config = getattr(model, "config", None)
    architecture = None
    if config is not None:
        model_type = getattr(config, "model_type", None)
        if model_type:
            architecture = model_type
        else:
            architecture = getattr(config, "architectures", None)
            if isinstance(architecture, Iterable):
                architecture = next(iter(architecture), None)
    architecture = architecture or type(model).__name__
    name = getattr(model, "name_or_path", architecture) or architecture
    quant_prefix = quant.lower()
    if quant == "TQ1_0":
        file_type = 36
    elif quant == "TQ2_0":
        file_type = 37
    else:
        file_type = 38
    quant_version = 3 if quant == "TQ2_0" else 2
    if quant == "TQ1_1":
        quant_version = 4
    entries = [
        ("general.architecture", architecture),
        ("general.name", name),
        ("general.file_type", file_type),
        ("general.alignment", HEADER_ALIGNMENT),
        ("general.quantized_by", "t81lib"),
        ("general.quantization_version", quant_version),
        ("quantization.type", quant_prefix),
        ("quantization.block_size", GGUF_QUANT_BLOCK_ROWS),
        ("quantization.threshold", threshold),
        (f"{quant_prefix}.threshold", threshold),
        (f"{quant_prefix}.version", 1),
    ]
    if architecture == "llama":
        entries.extend(_collect_llama_metadata(model, name))
    if extra_metadata:
        seen = {key for key, _ in entries}
        for key, value in extra_metadata.items():
            if key in seen:
                continue
            entries.append((key, value))
            seen.add(key)
    buffer = bytearray()
    for key, value in entries:
        buffer.extend(_encode_metadata_entry(key, value))
    return bytes(buffer), len(entries)


def _collect_llama_metadata(model: PreTrainedModel, name_or_path: str) -> list[tuple[str, Any]]:
    config = getattr(model, "config", None)
    if config is None:
        return []
    max_ctx = getattr(config, "max_position_embeddings", None) or getattr(
        config, "max_sequence_length", 2048
    )
    hidden_size = getattr(config, "hidden_size", 0)
    n_layers = getattr(config, "num_hidden_layers", 0)
    n_heads = getattr(config, "num_attention_heads", 0)
    n_kv_heads = getattr(config, "num_key_value_heads", n_heads)
    head_dim = hidden_size // n_heads if n_heads else 0
    ff_size = getattr(config, "intermediate_size", 0)
    rms_eps = getattr(config, "rms_norm_eps", 1e-5)
    rope_theta = getattr(config, "rope_theta", 10000.0)

    tokenizer_metadata = _collect_tokenizer_metadata(name_or_path, config)
    entries = [
        ("llama.context_length", int(max_ctx)),
        ("llama.embedding_length", int(hidden_size)),
        ("llama.block_count", int(n_layers)),
        ("llama.feed_forward_length", int(ff_size)),
        ("llama.rope.dimension_count", int(head_dim)),
        ("llama.rope.freq_base", float(rope_theta)),
        ("llama.attention.head_count", int(n_heads)),
        ("llama.attention.head_count_kv", int(n_kv_heads)),
        ("llama.attention.layer_norm_rms_epsilon", float(rms_eps)),
        ("llama.vocab_size", int(getattr(config, "vocab_size", 0))),
    ]
    entries.extend(tokenizer_metadata)
    return entries


def _collect_tokenizer_metadata(name_or_path: str, config: Any) -> list[tuple[str, Any]]:
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return []
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, use_fast=True)
    vocab_size = getattr(tokenizer, "vocab_size", None) or getattr(config, "vocab_size", 0)
    tokens = [tokenizer.convert_ids_to_tokens(i) for i in range(vocab_size)]
    scores = [0.0] * vocab_size
    token_types = [1] * vocab_size
    for idx, token in enumerate(tokens):
        if (
            len(token) == 6
            and token.startswith("<0x")
            and token.endswith(">")
            and all(ch in "0123456789abcdefABCDEF" for ch in token[3:5])
        ):
            token_types[idx] = 6
    unk_id = getattr(tokenizer, "unk_token_id", None)
    bos_id = getattr(tokenizer, "bos_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if unk_id is not None and 0 <= unk_id < vocab_size:
        token_types[unk_id] = 2
    for tok_id in (bos_id, eos_id, pad_id):
        if tok_id is not None and 0 <= tok_id < vocab_size:
            token_types[tok_id] = 3

    merges = []
    tokenizer_json = Path(name_or_path) / "tokenizer.json"
    if tokenizer_json.exists():
        try:
            import json

            with tokenizer_json.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            raw_merges = data.get("model", {}).get("merges", [])
            if raw_merges and isinstance(raw_merges[0], list):
                merges = [" ".join(pair) for pair in raw_merges]
            else:
                merges = [str(entry) for entry in raw_merges]
        except (OSError, json.JSONDecodeError):
            merges = []

    model_type = "llama"
    chat_template = None
    config_path = Path(name_or_path) / "tokenizer_config.json"
    if config_path.exists():
        try:
            import json

            with config_path.open("r", encoding="utf-8") as handle:
                config_data = json.load(handle)
            chat_template = config_data.get("chat_template")
        except (OSError, json.JSONDecodeError):
            chat_template = None
    entries = [
        ("tokenizer.ggml.model", model_type),
        ("tokenizer.ggml.tokens", tokens),
        ("tokenizer.ggml.scores", scores),
        ("tokenizer.ggml.token_type", token_types),
        ("tokenizer.ggml.bos_token_id", int(getattr(tokenizer, "bos_token_id", -1))),
        ("tokenizer.ggml.eos_token_id", int(getattr(tokenizer, "eos_token_id", -1))),
        ("tokenizer.ggml.unknown_token_id", int(getattr(tokenizer, "unk_token_id", -1))),
        ("tokenizer.ggml.padding_token_id", int(getattr(tokenizer, "pad_token_id", -1))),
    ]
    if merges:
        entries.append(("tokenizer.ggml.merges", merges))
    if chat_template:
        entries.append(("tokenizer.chat_template", chat_template))
    return entries


def _tensor_info_length(name: str, shape: tuple[int, ...]) -> int:
    encoded_name = name.encode("utf-8")
    length = 8 + len(encoded_name)  # name length (u64) + bytes
    length += 4  # n_dims
    length += 8 * len(shape)  # dims
    length += 4  # ggml_type
    length += 8  # offset
    return length


def _serialize_tensor_info(
    name: str,
    shape: tuple[int, ...],
    ggml_type: int,
    offset: int,
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
    return bytes(encoded)


def _read_bytes(handle: BinaryIO, count: int) -> bytes:
    data = handle.read(count)
    if len(data) != count:
        raise ValueError("metadata truncated")
    return data


def _read_metadata_value(handle: BinaryIO, value_type: int) -> Any:
    if value_type == GGUF_TYPE_UINT32:
        return struct.unpack("<I", _read_bytes(handle, 4))[0]
    if value_type == GGUF_TYPE_INT32:
        return struct.unpack("<i", _read_bytes(handle, 4))[0]
    if value_type == GGUF_TYPE_FLOAT32:
        return struct.unpack("<f", _read_bytes(handle, 4))[0]
    if value_type == GGUF_TYPE_BOOL:
        return bool(struct.unpack("<b", _read_bytes(handle, 1))[0])
    if value_type == GGUF_TYPE_STRING:
        length = struct.unpack("<Q", _read_bytes(handle, 8))[0]
        return _read_bytes(handle, length).decode("utf-8")
    if value_type == GGUF_TYPE_ARRAY:
        element_type = struct.unpack("<i", _read_bytes(handle, 4))[0]
        count = struct.unpack("<Q", _read_bytes(handle, 8))[0]
        return [_read_metadata_value(handle, element_type) for _ in range(count)]
    if value_type == GGUF_TYPE_UINT64:
        return struct.unpack("<Q", _read_bytes(handle, 8))[0]
    if value_type == GGUF_TYPE_INT64:
        return struct.unpack("<q", _read_bytes(handle, 8))[0]
    if value_type == GGUF_TYPE_FLOAT64:
        return struct.unpack("<d", _read_bytes(handle, 8))[0]
    raise ValueError(f"unsupported metadata value type {value_type}")


def _parse_metadata_from_file(handle: BinaryIO, count: int) -> tuple[Mapping[str, Any], int]:
    metadata: dict[str, Any] = {}
    start = handle.tell()
    for _ in range(count):
        key_len = struct.unpack("<Q", _read_bytes(handle, 8))[0]
        key = _read_bytes(handle, key_len).decode("utf-8")
        value_type = struct.unpack("<i", _read_bytes(handle, 4))[0]
        metadata[key] = _read_metadata_value(handle, value_type)
    return metadata, handle.tell() - start


def _parse_tensor_infos_from_file(
    handle: BinaryIO,
    count: int,
) -> list[_TensorInfo]:
    infos: list[_TensorInfo] = []
    for _ in range(count):
        name_len = struct.unpack("<Q", _read_bytes(handle, 8))[0]
        name_bytes = _read_bytes(handle, name_len)
        try:
            name = name_bytes.decode("utf-8")
        except UnicodeDecodeError:
            name = name_bytes.decode("latin-1")
        n_dims = struct.unpack("<I", _read_bytes(handle, 4))[0]
        shape: list[int] = []
        for _ in range(n_dims):
            shape.append(struct.unpack("<Q", _read_bytes(handle, 8))[0])
        ggml_type = struct.unpack("<i", _read_bytes(handle, 4))[0]
        data_offset = struct.unpack("<Q", _read_bytes(handle, 8))[0]
        infos.append(_TensorInfo(name, tuple(shape), ggml_type, data_offset))
    return infos

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
    metadata_size = len(metadata_bytes)
    tensor_infos_offset = HEADER_SIZE + metadata_size

    infos_length = sum(_tensor_info_length(payload.name, payload.shape) for payload in payloads)
    tensor_infos_size = infos_length
    tensor_data_offset = _align(tensor_infos_offset + tensor_infos_size, alignment)

    serialized_infos = bytearray()
    tensor_data_section = bytearray()
    current_data_offset = tensor_data_offset
    for payload in payloads:
        serialized_infos.extend(
            _serialize_tensor_info(
                payload.name,
                payload.shape,
                payload.ggml_type,
                current_data_offset - tensor_data_offset,
            )
        )
        tensor_data_section.extend(payload.data)
        current_data_offset += len(payload.data)

    header = HEADER_STRUCT.pack(
        HEADER_MAGIC,
        HEADER_VERSION,
        len(payloads),
        metadata_count,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(metadata_bytes)
        handle.write(serialized_infos)
        infos_padding = tensor_data_offset - (tensor_infos_offset + len(serialized_infos))
        if infos_padding > 0:
            handle.write(b"\x00" * infos_padding)
        handle.write(tensor_data_section)


def _float_to_half_bytes(value: float) -> bytes:
    return np.float16(value).tobytes()


def _round_away_from_zero(values: np.ndarray) -> np.ndarray:
    return np.where(values >= 0, np.floor(values + 0.5), np.ceil(values - 0.5))


def _quantize_trits(values: np.ndarray, threshold: float | None) -> np.ndarray:
    if threshold is None:
        trits = _round_away_from_zero(values)
        return np.clip(trits, -1, 1).astype(np.int8)
    return np.where(np.abs(values) >= threshold, np.sign(values), 0).astype(np.int8)


def _quantize_blocks_tq1(blocks: np.ndarray, threshold: float | None) -> np.ndarray:
    n_blocks = blocks.shape[0]
    d = np.abs(blocks).max(axis=-1, keepdims=True)
    with np.errstate(divide="ignore"):
        inv_d = np.where(d == 0, 0, 1.0 / d)
    trits = _quantize_trits(blocks * inv_d, threshold)
    qs = (trits + 1).astype(np.uint8)

    qs0, qs1, qh = qs[..., :(32 * 5)], qs[..., (32 * 5) : (48 * 5)], qs[..., (48 * 5) :]
    weights5 = np.array([81, 27, 9, 3, 1], dtype=np.uint8).reshape((1, 1, 5, 1))
    qs0 = (qs0.reshape((n_blocks, -1, 5, 32)) * weights5).sum(axis=-2).reshape((n_blocks, -1))
    qs1 = (qs1.reshape((n_blocks, -1, 5, 16)) * weights5).sum(axis=-2).reshape((n_blocks, -1))
    weights4 = np.array([81, 27, 9, 3], dtype=np.uint8).reshape((1, 1, 4, 1))
    qh = (qh.reshape((n_blocks, -1, 4, 4)) * weights4).sum(axis=-2).reshape((n_blocks, -1))

    qs = np.concatenate([qs0, qs1, qh], axis=-1)
    qs = (qs.astype(np.uint16) * 256 + (243 - 1)) // 243
    qs = qs.astype(np.uint8)
    d_bytes = d.astype(np.float16).view(np.uint8).reshape((n_blocks, 2))
    return np.concatenate([qs, d_bytes], axis=-1)


def _float_to_fp8_e4m3(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    out = np.zeros(values.shape, dtype=np.uint8)
    mask = values > 0
    if not np.any(mask):
        return out
    v = values[mask]
    exp = np.floor(np.log2(v)).astype(np.int32)
    exp = np.clip(exp, -6, 7)
    mant = np.round((v / np.exp2(exp) - 1.0) * (1 << TQ1_1_SCALE_BITS)).astype(np.int32)
    carry = mant >= (1 << TQ1_1_SCALE_BITS)
    if np.any(carry):
        mant = np.where(carry, 0, mant)
        exp = np.where(carry, np.minimum(exp + 1, 7), exp)
    mant = np.clip(mant, 0, (1 << TQ1_1_SCALE_BITS) - 1)
    exp = np.clip(exp + TQ1_1_SCALE_BIAS, 0, 0x0F)
    out_vals = (exp << TQ1_1_SCALE_BITS) | mant
    out[mask] = out_vals.astype(np.uint8)
    return out


def _fp8_e4m3_to_float(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.uint8, copy=False)
    exp = (values >> TQ1_1_SCALE_BITS) & 0x0F
    mant = values & ((1 << TQ1_1_SCALE_BITS) - 1)
    is_zero = (exp == 0) & (mant == 0)
    exp = exp.astype(np.int32) - TQ1_1_SCALE_BIAS
    scale = (1.0 + mant.astype(np.float32) / (1 << TQ1_1_SCALE_BITS)) * np.exp2(exp)
    scale = np.where(is_zero, 0.0, scale)
    return scale.astype(np.float32)


def _quantize_blocks_tq1_1(blocks: np.ndarray, threshold: float | None) -> np.ndarray:
    n_blocks = blocks.shape[0]
    d = np.abs(blocks).max(axis=-1, keepdims=True)
    with np.errstate(divide="ignore"):
        inv_d = np.where(d == 0, 0, 1.0 / d)
    trits = _quantize_trits(blocks * inv_d, threshold)
    qs = (trits + 1).astype(np.uint8)

    qs0, qs1, qh = qs[..., :(32 * 5)], qs[..., (32 * 5) : (48 * 5)], qs[..., (48 * 5) :]
    weights5 = np.array([81, 27, 9, 3, 1], dtype=np.uint8).reshape((1, 1, 5, 1))
    qs0 = (qs0.reshape((n_blocks, -1, 5, 32)) * weights5).sum(axis=-2).reshape((n_blocks, -1))
    qs1 = (qs1.reshape((n_blocks, -1, 5, 16)) * weights5).sum(axis=-2).reshape((n_blocks, -1))
    weights4 = np.array([81, 27, 9, 3], dtype=np.uint8).reshape((1, 1, 4, 1))
    qh = (qh.reshape((n_blocks, -1, 4, 4)) * weights4).sum(axis=-2).reshape((n_blocks, -1))

    qs = np.concatenate([qs0, qs1, qh], axis=-1)
    qs = (qs.astype(np.uint16) * 256 + (243 - 1)) // 243
    qs = qs.astype(np.uint8)
    d_fp8 = _float_to_fp8_e4m3(d.reshape((n_blocks,))).reshape((n_blocks, 1))
    return np.concatenate([qs, d_fp8], axis=-1)


def _quantize_blocks_tq2(blocks: np.ndarray, threshold: float | None) -> np.ndarray:
    n_blocks = blocks.shape[0]
    d = np.abs(blocks).max(axis=-1, keepdims=True)
    with np.errstate(divide="ignore"):
        inv_d = np.where(d == 0, 0, 1.0 / d)
    trits = _quantize_trits(blocks * inv_d, threshold)
    qs = (trits + 1).astype(np.uint8)

    shifts = np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 1, 4, 1))
    qs = qs.reshape((n_blocks, -1, 4, 32)) << shifts
    qs = qs[..., 0, :] | qs[..., 1, :] | qs[..., 2, :] | qs[..., 3, :]
    qs = qs.reshape((n_blocks, -1))

    d_bytes = d.astype(np.float16).view(np.uint8).reshape((n_blocks, 2))
    return np.concatenate([qs, d_bytes], axis=-1)


def _dequantize_blocks_tq1(blocks: np.ndarray) -> np.ndarray:
    n_blocks = blocks.shape[0]
    qk = GGUF_QUANT_BLOCK_ROWS
    qs, rest = np.hsplit(blocks, [(qk - 4 * qk // 64) // 5])
    qh, d = np.hsplit(rest, [qk // 64])

    d = d.view(np.float16).astype(np.float32).reshape((n_blocks, 1))

    qs0, qs1 = qs[..., :32], qs[..., 32:]
    weights5 = np.array([1, 3, 9, 27, 81], dtype=np.uint8).reshape((1, 1, 5, 1))
    qs0 = (qs0.reshape((n_blocks, -1, 1, 32)) * weights5).reshape((n_blocks, -1))
    qs1 = (qs1.reshape((n_blocks, -1, 1, 16)) * weights5).reshape((n_blocks, -1))
    weights4 = np.array([1, 3, 9, 27], dtype=np.uint8).reshape((1, 1, 4, 1))
    qh = (qh.reshape((n_blocks, -1, 1, 4)) * weights4).reshape((n_blocks, -1))
    qs = np.concatenate([qs0, qs1, qh], axis=-1)
    qs = ((qs.astype(np.uint16) * 3) >> 8).astype(np.int8) - np.int8(1)
    return d * qs.astype(np.float32)


def _dequantize_blocks_tq1_1(blocks: np.ndarray) -> np.ndarray:
    n_blocks = blocks.shape[0]
    qk = GGUF_QUANT_BLOCK_ROWS
    qs, rest = np.hsplit(blocks, [(qk - 4 * qk // 64) // 5])
    qh, d = np.hsplit(rest, [qk // 64])
    d = _fp8_e4m3_to_float(d.reshape((n_blocks,))).reshape((n_blocks, 1))

    qs0, qs1 = qs[..., :32], qs[..., 32:]
    weights5 = np.array([1, 3, 9, 27, 81], dtype=np.uint8).reshape((1, 1, 5, 1))
    qs0 = (qs0.reshape((n_blocks, -1, 1, 32)) * weights5).reshape((n_blocks, -1))
    qs1 = (qs1.reshape((n_blocks, -1, 1, 16)) * weights5).reshape((n_blocks, -1))
    weights4 = np.array([1, 3, 9, 27], dtype=np.uint8).reshape((1, 1, 4, 1))
    qh = (qh.reshape((n_blocks, -1, 1, 4)) * weights4).reshape((n_blocks, -1))
    qs = np.concatenate([qs0, qs1, qh], axis=-1)
    qs = ((qs.astype(np.uint16) * 3) >> 8).astype(np.int8) - np.int8(1)
    return d * qs.astype(np.float32)

def _dequantize_blocks_tq2(blocks: np.ndarray) -> np.ndarray:
    n_blocks = blocks.shape[0]
    qk = GGUF_QUANT_BLOCK_ROWS
    qs, d = np.hsplit(blocks, [qk // 4])
    d = d.view(np.float16).astype(np.float32).reshape((n_blocks, 1))
    shifts = np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 1, 4, 1))
    qs = qs.reshape((n_blocks, -1, 1, 32)) >> shifts
    qs = (qs & 0x03).reshape((n_blocks, -1)).astype(np.int8) - np.int8(1)
    return d * qs.astype(np.float32)


def _quantize_tensor(tensor: torch.Tensor, quant: str, threshold: float) -> bytes:
    """Quantize a 2D tensor into TQ1_0 or TQ2_0 payload bytes."""
    array = tensor.cpu().to(dtype=torch.float32, copy=False).numpy()
    if array.ndim != 2:
        raise ValueError(f"Only 2D tensors supported, got shape {array.shape}")
    rows, cols = array.shape
    serialized = bytearray()
    if cols == 0:
        return bytes(serialized)
    block_size = GGUF_QUANT_BLOCK_ROWS
    threshold_value = float(np.clip(threshold, 0.0, 1.0))
    use_threshold = threshold_value if threshold_value > 0.0 else None
    if quant == "TQ1_0":
        quantize_blocks = _quantize_blocks_tq1
    elif quant == "TQ1_1":
        quantize_blocks = _quantize_blocks_tq1_1
    else:
        quantize_blocks = _quantize_blocks_tq2
    blocks_per_row = (cols + block_size - 1) // block_size
    padded_cols = blocks_per_row * block_size
    for row in array:
        padded = np.pad(row, (0, padded_cols - cols), constant_values=0)
        blocks = padded.reshape((blocks_per_row, block_size))
        packed = quantize_blocks(blocks, use_threshold)
        serialized.extend(packed.astype(np.uint8).tobytes(order="C"))
    return bytes(serialized)


def _collect_quantized_weight_names(model: PreTrainedModel) -> set[str]:
    seen: set[str] = set()
    for module_name, module in model.named_modules():
        if isinstance(module, TernaryLinear):
            tensor_name = f"{module_name}.weight" if module_name else "weight"
            seen.add(tensor_name)
    return seen


def _collect_all_parameters(model: PreTrainedModel) -> list[tuple[str, torch.Tensor]]:
    result: list[tuple[str, torch.Tensor]] = []
    seen: set[str] = set()
    for name, parameter in model.named_parameters():
        if name in seen:
            continue
        seen.add(name)
        result.append((name, parameter.detach()))
    return result


def _map_llama_tensor_name(name: str) -> str:
    if name == "model.embed_tokens.weight":
        return "token_embd.weight"
    if name == "model.norm.weight":
        return "output_norm.weight"
    if name == "lm_head.weight":
        return "output.weight"
    if not name.startswith("model.layers."):
        return name
    parts = name.split(".")
    if len(parts) < 4:
        return name
    layer_id = parts[2]
    suffix = ".".join(parts[3:])
    prefix = f"blk.{layer_id}."
    mapping = {
        "self_attn.q_proj.weight": "attn_q.weight",
        "self_attn.k_proj.weight": "attn_k.weight",
        "self_attn.v_proj.weight": "attn_v.weight",
        "self_attn.o_proj.weight": "attn_output.weight",
        "mlp.gate_proj.weight": "ffn_gate.weight",
        "mlp.up_proj.weight": "ffn_up.weight",
        "mlp.down_proj.weight": "ffn_down.weight",
        "input_layernorm.weight": "attn_norm.weight",
        "post_attention_layernorm.weight": "ffn_norm.weight",
    }
    mapped = mapping.get(suffix)
    if mapped is not None:
        return prefix + mapped
    if suffix.endswith(".bias"):
        return prefix + suffix
    return name


def resolve_gguf_profile(profile: str | None) -> GGUFExportProfile | None:
    if profile is None:
        return None
    key = profile.strip().lower()
    selected = GGUF_EXPORT_PROFILES.get(key)
    if selected is None:
        known = ", ".join(sorted(GGUF_EXPORT_PROFILES))
        raise ValueError(f"unknown GGUF profile '{profile}', expected one of: {known}")
    return selected


def _infer_ggml_type(tensor: torch.Tensor, quantized: bool, quant: str) -> int:
    if quantized:
        if quant == "TQ1_0":
            return GGML_TYPE_TQ1_0
        if quant == "TQ1_1":
            return GGML_TYPE_TQ1_1
        return GGML_TYPE_TQ2_0
    if tensor.dtype in (torch.float16, torch.bfloat16):
        return GGML_TYPE_F16
    return GGML_TYPE_F32


def _float_tensor_bytes(tensor: torch.Tensor, ggml_type: int) -> bytes:
    array = tensor.detach().cpu()
    if ggml_type == GGML_TYPE_F16:
        array = array.to(dtype=torch.float16)
        numpy_dtype = np.float16
    else:
        array = array.to(dtype=torch.float32)
        numpy_dtype = np.float32
    numpy_array = array.numpy()
    if numpy_array.dtype != numpy_dtype:
        numpy_array = numpy_array.astype(numpy_dtype, copy=False)
    return numpy_array.tobytes(order="C")


def write_gguf(
    model: PreTrainedModel,
    path: str | Path,
    *,
    quant: str = "TQ1_0",
    threshold: float = 0.45,
    profile: str | None = None,
) -> None:
    """
    Write a GGUF file from a converted model built from ``t81.nn.Linear`` modules.
    """
    profile_data = resolve_gguf_profile(profile)
    if profile_data is not None:
        quant = profile_data.quant
        threshold = profile_data.threshold
        profile_metadata = profile_data.metadata
    else:
        profile_metadata = {}
    quant = quant.upper()
    if profile_data is not None and profile_data.name == GGUF_PROFILE_TQ1_1_DRAFT:
        if os.getenv("T81_ENABLE_TQ1_1") == "1":
            quant = "TQ1_1"
        else:
            raise ValueError("tq1_1-draft requires T81_ENABLE_TQ1_1=1")
    if quant not in {"TQ1_0", "TQ1_1", "TQ2_0"}:
        raise ValueError("quant must be one of 'TQ1_0', 'TQ1_1', or 'TQ2_0'")
    if quant == "TQ1_1" and os.getenv("T81_ENABLE_TQ1_1") != "1":
        raise ValueError("TQ1_1 is experimental; set T81_ENABLE_TQ1_1=1 to enable")
    threshold = float(np.clip(threshold, 0.0, 1.0))
    quantized_names = _collect_quantized_weight_names(model)
    if not quantized_names:
        raise ValueError("model does not contain any t81.nn.Linear layers")

    parameters = _collect_all_parameters(model)
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", None) if config is not None else None
    use_llama_names = model_type == "llama"
    tensor_payloads: list[_TensorPayload] = []
    for name, tensor in parameters:
        write_tensor = tensor.t() if use_llama_names and tensor.ndim == 2 else tensor
        quantizable = name in quantized_names and tensor.ndim == 2
        if use_llama_names and name in {"model.embed_tokens.weight", "lm_head.weight"}:
            quantizable = False
        ggml_type = _infer_ggml_type(write_tensor, quantizable, quant)
        if quantizable:
            data = _quantize_tensor(write_tensor, quant, threshold)
        else:
            data = _float_tensor_bytes(write_tensor, ggml_type)
        output_name = _map_llama_tensor_name(name) if use_llama_names else name
        tensor_payloads.append(
            _TensorPayload(
                name=output_name,
                shape=tuple(write_tensor.shape),
                ggml_type=ggml_type,
                data=data,
            )
        )

    alignment = HEADER_ALIGNMENT
    metadata_bytes, metadata_count = _collect_metadata(
        model,
        quant,
        threshold,
        extra_metadata=profile_metadata,
    )
    _write_payloads(tensor_payloads, metadata_bytes, metadata_count, alignment, Path(path))


def repack_gguf(
    source: str | Path,
    destination: str | Path,
    *,
    quant: str = "TQ1_0",
    threshold: float = 0.45,
    profile: str | None = None,
) -> None:
    """Repack an existing GGUF file with ternary quantization."""
    quant = quant.upper()
    if quant not in {"TQ1_0", "TQ1_1", "TQ2_0"}:
        raise ValueError("quant must be one of 'TQ1_0', 'TQ1_1', or 'TQ2_0'")
    profile_data = resolve_gguf_profile(profile)
    if profile_data is not None:
        quant = profile_data.quant
        threshold = profile_data.threshold
        profile_metadata = profile_data.metadata
    else:
        profile_metadata = {}
    if profile_data is not None and profile_data.name == GGUF_PROFILE_TQ1_1_DRAFT:
        if os.getenv("T81_ENABLE_TQ1_1") == "1":
            quant = "TQ1_1"
        else:
            raise ValueError("tq1_1-draft requires T81_ENABLE_TQ1_1=1")
    if quant == "TQ1_1" and os.getenv("T81_ENABLE_TQ1_1") != "1":
        raise ValueError("TQ1_1 is experimental; set T81_ENABLE_TQ1_1=1 to enable")
    threshold = float(np.clip(threshold, 0.0, 1.0))
    quant_prefix = quant.lower()

    source_path = Path(source)
    file_size = source_path.stat().st_size
    with source_path.open("rb") as handle:
        header = handle.read(HEADER_SIZE)
        if len(header) < HEADER_SIZE:
            raise ValueError("file too short to be a valid GGUF blob")
        magic, version, num_tensors, metadata_kv_count = HEADER_STRUCT.unpack_from(header, 0)
        if magic != HEADER_MAGIC or version != HEADER_VERSION:
            raise ValueError("GGUF header mismatch")
        metadata, metadata_length = _parse_metadata_from_file(handle, metadata_kv_count)
        alignment = int(metadata.get("general.alignment", HEADER_ALIGNMENT))
        if alignment <= 0:
            raise ValueError("invalid GGUF alignment")
        metadata_end = handle.tell()
        peek = handle.read(8)
        if len(peek) != 8:
            raise ValueError("metadata truncated before tensor infos")
        name_len = struct.unpack("<Q", peek)[0]
        handle.seek(-8, 1)
        if name_len == 0 or name_len > 4096:
            aligned_end = HEADER_SIZE + _align(metadata_length, alignment)
            if aligned_end > metadata_end:
                handle.seek(aligned_end - metadata_end, 1)
        tensor_infos = _parse_tensor_infos_from_file(handle, num_tensors)
        tensor_infos_end = handle.tell()
        tensor_data_start = _align(tensor_infos_end, alignment)

        offsets_are_relative = any(info.offset < tensor_data_start for info in tensor_infos)

        def _resolve_offset(info: _TensorInfo) -> int:
            if offsets_are_relative:
                return tensor_data_start + info.offset
            return info.offset

        def _tensor_nbytes(info: _TensorInfo) -> int:
            if not info.shape:
                return 0
            if info.ggml_type == GGML_TYPE_F16:
                return int(np.prod(info.shape)) * 2
            if info.ggml_type == GGML_TYPE_F32:
                return int(np.prod(info.shape)) * 4
            if info.ggml_type == GGML_TYPE_TQ1_0:
                rows, cols = info.shape[0], info.shape[1] if len(info.shape) > 1 else 1
                blocks_per_row = (cols + GGUF_QUANT_BLOCK_ROWS - 1) // GGUF_QUANT_BLOCK_ROWS
                return rows * blocks_per_row * 54
            if info.ggml_type == GGML_TYPE_TQ1_1:
                rows, cols = info.shape[0], info.shape[1] if len(info.shape) > 1 else 1
                blocks_per_row = (cols + GGUF_QUANT_BLOCK_ROWS - 1) // GGUF_QUANT_BLOCK_ROWS
                return rows * blocks_per_row * 53
            if info.ggml_type == GGML_TYPE_TQ2_0:
                rows, cols = info.shape[0], info.shape[1] if len(info.shape) > 1 else 1
                blocks_per_row = (cols + GGUF_QUANT_BLOCK_ROWS - 1) // GGUF_QUANT_BLOCK_ROWS
                return rows * blocks_per_row * 66
            return 0

        payloads_by_name: dict[str, _TensorPayload] = {}
        for info in tensor_infos:
            resolved_offset = _resolve_offset(info)
            nbytes = _tensor_nbytes(info)
            if nbytes <= 0:
                raise ValueError(f"unsupported tensor size for {info.name}")
            if resolved_offset + nbytes > file_size:
                raise ValueError("tensor data extends beyond file length")
            handle.seek(resolved_offset)
            data = _read_bytes(handle, nbytes)
            quantizable = len(info.shape) == 2 and info.ggml_type in {GGML_TYPE_F16, GGML_TYPE_F32}
            if quantizable:
                dtype = np.float16 if info.ggml_type == GGML_TYPE_F16 else np.float32
                array = np.frombuffer(memoryview(data), dtype=dtype).reshape(info.shape).astype(np.float32)
                tensor = torch.from_numpy(array)
                data = _quantize_tensor(tensor, quant, threshold)
                if quant == "TQ1_0":
                    ggml_type = GGML_TYPE_TQ1_0
                elif quant == "TQ1_1":
                    ggml_type = GGML_TYPE_TQ1_1
                else:
                    ggml_type = GGML_TYPE_TQ2_0
            else:
                ggml_type = info.ggml_type
            payloads_by_name[info.name] = _TensorPayload(
                name=info.name,
                shape=info.shape,
                ggml_type=ggml_type,
                data=data,
            )

    metadata["general.quantized_by"] = "t81lib"
    if quant == "TQ2_0":
        quant_version = 3
    elif quant == "TQ1_1":
        quant_version = 4
    else:
        quant_version = 2
    metadata["general.quantization_version"] = quant_version
    metadata["quantization.type"] = quant_prefix
    metadata["quantization.block_size"] = GGUF_QUANT_BLOCK_ROWS
    metadata["quantization.threshold"] = threshold
    metadata[f"{quant_prefix}.threshold"] = threshold
    metadata[f"{quant_prefix}.version"] = 1
    for key, value in profile_metadata.items():
        metadata.setdefault(key, value)

    payloads = [payloads_by_name[info.name] for info in tensor_infos]
    metadata_bytes, metadata_count = _serialize_metadata_from_mapping(metadata)
    _write_payloads(payloads, metadata_bytes, metadata_count, alignment, Path(destination))


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
    if ggml_type in {GGML_TYPE_F32, GGML_TYPE_F16}:
        dtype = np.float32 if ggml_type == GGML_TYPE_F32 else np.float16
        data = np.frombuffer(memoryview(chunk), dtype=dtype)
        if shape:
            expected = int(np.prod(shape))
            if data.size < expected:
                raise ValueError("float tensor data truncated")
            data = data[:expected]
            data = data.reshape(shape)
        return torch.from_numpy(data)
    block_size = block_rows
    if block_size <= 0:
        raise ValueError("invalid quant block size")
    if cols == 0 or rows == 0:
        return torch.zeros(shape, dtype=torch.float32)
    if ggml_type == GGML_TYPE_TQ1_0:
        block_bytes = 54
    elif ggml_type == GGML_TYPE_TQ1_1:
        block_bytes = 53
    else:
        block_bytes = 66
    blocks_per_row = (cols + block_size - 1) // block_size
    row_bytes = blocks_per_row * block_bytes
    expected_bytes = rows * row_bytes
    if len(chunk) < expected_bytes:
        raise ValueError("quantized tensor data truncated")
    data = np.frombuffer(memoryview(chunk), dtype=np.uint8, count=expected_bytes)
    data = data.reshape((rows, blocks_per_row, block_bytes))
    if ggml_type == GGML_TYPE_TQ1_0:
        dequantize_blocks = _dequantize_blocks_tq1
    elif ggml_type == GGML_TYPE_TQ1_1:
        dequantize_blocks = _dequantize_blocks_tq1_1
    else:
        dequantize_blocks = _dequantize_blocks_tq2
    decoded_rows = []
    for row_blocks in data:
        dequant = dequantize_blocks(row_blocks)
        decoded_rows.append(dequant.reshape(-1)[:cols])
    return torch.from_numpy(np.stack(decoded_rows, axis=0).astype(np.float32))


def read_gguf(
    path: str | Path,
    *,
    dequantize: bool = True,
    return_metadata: bool = False,
) -> Mapping[str, torch.Tensor | bytes] | tuple[Mapping[str, torch.Tensor | bytes], Mapping[str, Any]]:
    path_obj = Path(path)
    file_size = path_obj.stat().st_size
    with path_obj.open("rb") as handle:
        header = handle.read(HEADER_SIZE)
        if len(header) < HEADER_SIZE:
            raise ValueError("file too short to be a valid GGUF blob")
        magic, version, num_tensors, metadata_kv_count = HEADER_STRUCT.unpack_from(header, 0)
        if magic != HEADER_MAGIC or version != HEADER_VERSION:
            raise ValueError("GGUF header mismatch")
        metadata, metadata_length = _parse_metadata_from_file(handle, metadata_kv_count)
        alignment = int(metadata.get("general.alignment", HEADER_ALIGNMENT))
        if alignment <= 0:
            raise ValueError("invalid GGUF alignment")
        metadata_end = handle.tell()
        tensor_infos_offset = metadata_end
        peek = handle.read(8)
        if len(peek) != 8:
            raise ValueError("metadata truncated before tensor infos")
        name_len = struct.unpack("<Q", peek)[0]
        handle.seek(-8, 1)
        if name_len == 0 or name_len > 4096:
            aligned_end = HEADER_SIZE + _align(metadata_length, alignment)
            if aligned_end > metadata_end:
                handle.seek(aligned_end - metadata_end, 1)
            tensor_infos_offset = handle.tell()
        tensor_infos = _parse_tensor_infos_from_file(handle, num_tensors)
        tensor_infos_end = handle.tell()
        tensor_data_start = _align(tensor_infos_end, alignment)

        offsets_are_relative = any(info.offset < tensor_data_start for info in tensor_infos)

        def _resolve_offset(info: _TensorInfo) -> int:
            if offsets_are_relative:
                return tensor_data_start + info.offset
            return info.offset

        sorted_infos = sorted(tensor_infos, key=_resolve_offset)
        block_rows = int(metadata.get("quantization.block_size", GGUF_QUANT_BLOCK_ROWS))
        payload: dict[str, torch.Tensor | bytes] = {}
        prev_end = 0
        for index, info in enumerate(sorted_infos):
            resolved_offset = _resolve_offset(info)
            if resolved_offset < prev_end:
                raise ValueError("tensor data overlaps or is out of order")
            next_offset = (
                _resolve_offset(sorted_infos[index + 1]) if index + 1 < len(sorted_infos) else file_size
            )
            if next_offset > file_size:
                raise ValueError("tensor data extends beyond file length")
            prev_end = next_offset
            handle.seek(resolved_offset)
            chunk = _read_bytes(handle, next_offset - resolved_offset)
            if info.ggml_type == GGML_TYPE_TQ1_1 and os.getenv("T81_ENABLE_TQ1_1") != "1":
                raise ValueError("TQ1_1 tensors require T81_ENABLE_TQ1_1=1")
            if info.ggml_type not in {
                GGML_TYPE_TQ1_0,
                GGML_TYPE_TQ1_1,
                GGML_TYPE_TQ2_0,
                GGML_TYPE_F32,
                GGML_TYPE_F16,
            }:
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
        numpy_array = array.cpu().numpy()
        if numpy_array.dtype != numpy_dtype:
            numpy_array = numpy_array.astype(numpy_dtype, copy=False)
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
