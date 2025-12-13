"""Shared dummy models used across the GGUF regression suite."""

from __future__ import annotations

import torch
import t81
from transformers import PreTrainedModel, PretrainedConfig


class _DummyConfig(PretrainedConfig):
    model_type = "dummy"
    architectures = ("DummyModel",)


class _DummyModel(PreTrainedModel):
    config_class = _DummyConfig
    base_model_prefix = "dummy"

    def __init__(self, config: PretrainedConfig | None = None):
        config = config or _DummyConfig()
        super().__init__(config)
        self.linear = t81.Linear(4, 3)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:  # pragma: no cover - not executed
        return self.linear(tensor)


class _LargeDummyModel(PreTrainedModel):
    config_class = _DummyConfig
    base_model_prefix = "dummy-large"

    def __init__(self, config: PretrainedConfig | None = None):
        config = config or _DummyConfig()
        super().__init__(config)
        self.linear = t81.Linear(128, 64)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:  # pragma: no cover - not executed
        return self.linear(tensor)


def _populate_linear(model: _DummyModel, low: float, high: float) -> None:
    with torch.no_grad():
        values = torch.linspace(low, high, steps=model.linear.weight.numel(), dtype=torch.float32)
        values = values.reshape(model.linear.weight.shape)
        model.linear.weight.copy_(values)
