from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
from torch import nn
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForMultipleChoice,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoModelForTokenClassification,
    PreTrainedModel,
)

from .nn import Linear as TernaryLinear

_METADATA_FILENAME = "t81_metadata.json"


@dataclass
class _ConversionStats:
    float_bytes: int = 0
    ternary_bytes: int = 0


_AUTO_MODEL_CLASSES = (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
    AutoModelForMultipleChoice,
    AutoModel,
)


def _ternary_byte_count(shape: tuple[int, int]) -> int:
    rows, cols = shape
    k_limbs = (cols + 47) // 48
    return rows * k_limbs * 16


def _convert_linear_layer(
    linear: nn.Module,
    threshold: float,
    keep_biases_bf16: bool,
    stats: _ConversionStats,
) -> nn.Module | None:
    if isinstance(linear, TernaryLinear):
        linear.threshold = threshold
        return None
    if not isinstance(linear, nn.Linear):
        return None

    device = linear.weight.device
    dtype = linear.weight.dtype
    ternary = TernaryLinear(
        linear.in_features,
        linear.out_features,
        bias=linear.bias is not None,
        device=device,
        dtype=dtype,
        threshold=threshold,
    )
    with torch.no_grad():
        ternary.weight.copy_(linear.weight)
        if linear.bias is not None and ternary.bias is not None:
            bias_tensor = linear.bias.detach()
            target_dtype = bias_tensor.dtype
            if not keep_biases_bf16 and target_dtype == torch.bfloat16:
                target_dtype = torch.float32
            if target_dtype != ternary.bias.dtype:
                ternary.bias = torch.nn.Parameter(
                    bias_tensor.to(device, dtype=target_dtype),
                    requires_grad=linear.bias.requires_grad,
                )
            else:
                ternary.bias.copy_(bias_tensor)

    stats.float_bytes += linear.weight.numel() * linear.weight.element_size()
    stats.ternary_bytes += _ternary_byte_count(tuple(linear.weight.shape))
    return ternary


def _walk_and_replace(
    module: nn.Module,
    threshold: float,
    keep_biases_bf16: bool,
    stats: _ConversionStats,
) -> None:
    for name, child in list(module.named_children()):
        replacement = _convert_linear_layer(child, threshold, keep_biases_bf16, stats)
        if replacement is not None:
            setattr(module, name, replacement)
        else:
            _walk_and_replace(child, threshold, keep_biases_bf16, stats)


def _attach_metadata(
    model: PreTrainedModel, threshold: float, keep_biases_bf16: bool, stats: _ConversionStats
) -> None:
    setattr(model, "_t81_threshold", threshold)
    setattr(model, "_t81_keep_biases_bf16", keep_biases_bf16)
    setattr(model, "_t81_compression_stats", stats)


def _report_stats(stats: _ConversionStats, replaced: int) -> None:
    if stats.ternary_bytes == 0:
        return
    compressed = stats.ternary_bytes
    original = stats.float_bytes
    ratio = original / compressed if compressed else 0.0
    saving = original - compressed
    miB = 1024**2
    print(
        f"t81.convert: replaced {replaced} nn.Linear modules, "
        f"compressed {original / miB:.2f} MiB -> {compressed / miB:.2f} MiB "
        f"(ratio {ratio:.2f}), estimated VRAM savings {saving / miB:.2f} MiB"
    )


def _select_model_class(model_id_or_path: str, **kwargs: Any) -> PreTrainedModel:
    last_error: Exception | None = None
    for cls in _AUTO_MODEL_CLASSES:
        try:
            return cls.from_pretrained(model_id_or_path, **kwargs)
        except Exception as exc:  # pylint: disable=broad-except
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError("Unable to instantiate model")


def _load_model(
    model_id_or_path: str,
    device_map: str | Mapping[str, Any] | None,
    torch_dtype: torch.dtype | None,
) -> PreTrainedModel:
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": device_map,
        "low_cpu_mem_usage": True,
    }
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    return _select_model_class(model_id_or_path, **kwargs)


def _convert_model_instance(
    model: PreTrainedModel, threshold: float, keep_biases_bf16: bool
) -> _ConversionStats:
    stats = _ConversionStats()
    _walk_and_replace(model, threshold, keep_biases_bf16, stats)
    _attach_metadata(model, threshold, keep_biases_bf16, stats)
    return stats


def convert(
    model_id_or_path: str | PreTrainedModel,
    threshold: float = 0.45,
    keep_biases_bf16: bool = True,
    device_map: str | Mapping[str, Any] | None = "auto",
    torch_dtype: torch.dtype | None = None,
    *,
    inplace: bool = False,
) -> PreTrainedModel:
    """
    Load a Hugging Face model and swap every ``nn.Linear`` for ``t81.nn.Linear`` so the
    weights get quantized lazily via ``t81.torch.TernaryTensor``.
    If an existing ``PreTrainedModel`` is passed, set ``inplace=True`` and the instance
    is modified without reloading from disk.
    """
    if isinstance(model_id_or_path, PreTrainedModel):
        if not inplace:
            raise ValueError("convert() requires inplace=True when passing a PreTrainedModel instance")
        model = model_id_or_path
    else:
        model = _load_model(model_id_or_path, device_map=device_map, torch_dtype=torch_dtype)
    stats = _convert_model_instance(model, threshold, keep_biases_bf16)
    _report_stats(stats, replaced=sum(1 for _ in model.modules() if isinstance(_, TernaryLinear)))
    return model


def _metadata_path(directory: Path) -> Path:
    return directory / _METADATA_FILENAME


def save_pretrained_t81(
    self: PreTrainedModel,
    save_directory: str | Path,
    *,
    threshold: float | None = None,
    keep_biases_bf16: bool | None = None,
    call_save_pretrained: bool = True,
    save_fn: Callable[[PreTrainedModel, str], None] | None = None,
    **kwargs: Any,
) -> None:
    directory = Path(save_directory)
    directory.mkdir(parents=True, exist_ok=True)
    if call_save_pretrained:
        original_save = getattr(type(self), "__t81_original_save_pretrained__", None)
        save_callable = save_fn or original_save or self.save_pretrained
        save_callable(self, directory, **kwargs)
    metadata = {
        "threshold": threshold if threshold is not None else getattr(self, "_t81_threshold", 0.45),
        "keep_biases_bf16": keep_biases_bf16
        if keep_biases_bf16 is not None
        else getattr(self, "_t81_keep_biases_bf16", True),
    }
    with _metadata_path(directory).open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle)


def from_pretrained_t81(
    cls: type[PreTrainedModel],
    pretrained_model_name_or_path: str,
    *model_args: Any,
    threshold: float | None = None,
    keep_biases_bf16: bool | None = None,
    device_map: str | Mapping[str, Any] | None = "auto",
    torch_dtype: torch.dtype | None = None,
    trust_remote_code: bool = True,
    **model_kwargs: Any,
) -> PreTrainedModel:
    metadata: dict[str, Any] = {}
    directory = Path(pretrained_model_name_or_path)
    if _metadata_path(directory).exists():
        with _metadata_path(directory).open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    threshold = threshold if threshold is not None else metadata.get("threshold", 0.45)
    keep_biases_bf16 = (
        keep_biases_bf16
        if keep_biases_bf16 is not None
        else metadata.get("keep_biases_bf16", True)
    )
    model = cls.from_pretrained(
        pretrained_model_name_or_path,
        *model_args,
        trust_remote_code=trust_remote_code,
        device_map=device_map,
        torch_dtype=torch_dtype,
        **model_kwargs,
    )
    _convert_model_instance(model, threshold, keep_biases_bf16)
    return model


def _parse_dtype(value: str) -> torch.dtype:
    try:
        return getattr(torch, value)
    except AttributeError as exc:
        raise argparse.ArgumentTypeError(f"{value} is not a valid torch.dtype") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert a HF model to t81 ternary modules.")
    parser.add_argument("model_id_or_path", help="Pretrained model identifier or local directory.")
    parser.add_argument("output_dir", help="Directory where the converted model will be written.")
    parser.add_argument("--threshold", type=float, default=0.45, help="Ternary quantization threshold.")
    parser.add_argument(
        "--keep-biases-bf16",
        action="store_true",
        dest="keep_biases_bf16",
        default=True,
        help="Keep bias tensors in BF16/FP32 when possible.",
    )
    parser.add_argument(
        "--no-keep-biases-bf16",
        action="store_false",
        dest="keep_biases_bf16",
        help="Force biases to float32.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Device map passed to `transformers.PreTrainedModel.from_pretrained`.",
    )
    parser.add_argument(
        "--torch-dtype",
        type=_parse_dtype,
        help="Optional torch dtype for the underlying float copy.",
    )
    args = parser.parse_args()

    model = convert(
        args.model_id_or_path,
        threshold=args.threshold,
        keep_biases_bf16=args.keep_biases_bf16,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
    )
    model.save_pretrained_t81(args.output_dir)
    return 0


def _bind_pretrained_methods() -> None:
    from transformers import PreTrainedModel as _PreTrainedModel

    setattr(_PreTrainedModel, "save_pretrained_t81", save_pretrained_t81)
    setattr(_PreTrainedModel, "from_pretrained_t81", classmethod(from_pretrained_t81))
    if not hasattr(_PreTrainedModel, "__t81_original_save_pretrained__"):
        setattr(_PreTrainedModel, "__t81_original_save_pretrained__", _PreTrainedModel.save_pretrained)
    if not getattr(_PreTrainedModel, "__t81_save_pretrained_hook__", False):
        original_save = getattr(_PreTrainedModel, "__t81_original_save_pretrained__")

        def _t81_save_pretrained(self, save_directory, *args, **kwargs):
            result = original_save(self, save_directory, *args, **kwargs)
            save_pretrained_t81(
                self,
                save_directory,
                threshold=getattr(self, "_t81_threshold", 0.45),
                keep_biases_bf16=getattr(self, "_t81_keep_biases_bf16", True),
                call_save_pretrained=False,
            )
            return result

        setattr(_PreTrainedModel, "save_pretrained", _t81_save_pretrained)
        setattr(_PreTrainedModel, "__t81_save_pretrained_hook__", True)


_bind_pretrained_methods()
