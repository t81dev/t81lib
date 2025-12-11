"""Helper trainer and arguments for ternary quantization-aware training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import Trainer, TrainingArguments

from .convert import convert
from .nn import Linear


@dataclass
class TernaryTrainingArguments(TrainingArguments):
    """Extends ``TrainingArguments`` with ternary-specific knobs."""

    ternary_threshold: float = field(
        default=0.44,
        metadata={"help": "Target ternary threshold that is enabled after warmup."},
    )
    ternary_stochastic_rounding: bool = field(
        default=False,
        metadata={"help": "Enable stochastic rounding when quantizing weights."},
    )
    ternary_warmup_steps: int = field(
        default=100,
        metadata={"help": "Number of steps to warm up the ternary threshold from 1.0."},
    )


class TernaryTrainer(Trainer):
    """Trainer that converts a HF model to ternary weights before running QAT."""

    args: TernaryTrainingArguments  # type: ignore[assignment]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _args = list(args)
        training_args = kwargs.get("args")
        if training_args is None and len(_args) > 1:
            training_args = _args[1]
        if training_args is None:
            raise ValueError("TernaryTrainer requires TrainingArguments or TernaryTrainingArguments")
        if not isinstance(training_args, TernaryTrainingArguments):
            sanitized = (
                training_args.to_sanitized_dict()
                if hasattr(training_args, "to_sanitized_dict")
                else vars(training_args)
            )
            training_args = TernaryTrainingArguments(**sanitized)
            if len(_args) > 1:
                _args[1] = training_args
            kwargs["args"] = training_args
        else:
            kwargs["args"] = training_args
        args = tuple(_args)

        super().__init__(*args, **kwargs)
        convert(self.model, threshold=self.args.ternary_threshold, inplace=True)
        self._ternary_modules = self._collect_linear_modules()
        self._apply_qat_config()

    def _collect_linear_modules(self) -> list[Linear]:
        return [module for module in self.model.modules() if isinstance(module, Linear)]

    def _effective_threshold(self) -> float:
        warmup = max(0, self.args.ternary_warmup_steps)
        if warmup == 0:
            return self.args.ternary_threshold
        step = getattr(self.state, "global_step", 0)
        progress = min(step / warmup, 1.0)
        target = self.args.ternary_threshold
        return target + (1.0 - target) * (1.0 - progress)

    def _apply_qat_config(self) -> None:
        threshold = self._effective_threshold()
        stochastic = self.args.ternary_stochastic_rounding
        for module in self._ternary_modules:
            module.ternary_threshold = threshold
            module.configure_qat(stochastic_rounding=stochastic)

    def training_step(self, model, inputs: dict[str, Any]) -> torch.Tensor:  # type: ignore[override]
        self._apply_qat_config()
        return super().training_step(model, inputs)

    def save_model(self, output_dir: str | None = None, state_dict: dict[str, Any] | None = None) -> None:
        directory = output_dir or self.args.output_dir
        super().save_model(output_dir=directory, state_dict=state_dict)
        if hasattr(self.model, "save_pretrained_t81"):
            getattr(self.model, "save_pretrained_t81")(directory)
