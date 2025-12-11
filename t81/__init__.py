"""
t81 package root for the PyTorch helpers and helpers that live alongside ``t81lib``'s bindings.

The torch subpackage is deliberately minimal so consumers can ``import t81`` and access
``t81.trit``/``t81.trt`` helpers together with ``t81.torch`` without conflicting with the
system ``torch`` module.
"""

from . import hardware  # expose ternary hardware helpers alongside torch
from . import torch as torch_integration  # Enables ``import t81``.
from .convert import convert
from .nn import Linear

__all__ = ["torch", "hardware", "convert", "Linear"]
