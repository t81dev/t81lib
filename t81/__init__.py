"""
t81 package root for the PyTorch helpers and helpers that live alongside ``t81lib``'s bindings.

The torch subpackage is deliberately minimal so consumers can ``import t81`` and access
``t81.trit``/``t81.trt`` helpers together with ``t81.torch`` without conflicting with the
system ``torch`` module.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

__all__ = [
    "torch",
    "hardware",
    "gguf",
    "read_gguf",
    "write_gguf",
    "convert",
    "Linear",
    "ternary",
    "TernaryTrainer",
    "TernaryTrainingArguments",
]

_LAZY_MODULES: dict[str, str] = {
    "gguf": ".gguf",
    "hardware": ".hardware",
    "torch": ".torch",
}

_LAZY_MEMBERS: dict[str, tuple[str, str]] = {
    "read_gguf": (".gguf", "read_gguf"),
    "write_gguf": (".gguf", "write_gguf"),
    "convert": (".convert", "convert"),
    "Linear": (".nn", "Linear"),
    "ternary": (".qat", "ternary"),
    "TernaryTrainer": (".trainer", "TernaryTrainer"),
    "TernaryTrainingArguments": (".trainer", "TernaryTrainingArguments"),
}

_MODULE_CACHE: dict[str, ModuleType] = {}


def _import_module(path: str) -> ModuleType:
    module = _MODULE_CACHE.get(path)
    if module is None:
        module = importlib.import_module(path, __name__)
        _MODULE_CACHE[path] = module
    return module


def __getattr__(name: str) -> Any:
    if name in _LAZY_MODULES:
        module = _import_module(_LAZY_MODULES[name])
        globals()[name] = module
        return module
    if name in _LAZY_MEMBERS:
        module_path, attr = _LAZY_MEMBERS[name]
        module = _import_module(module_path)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals().keys()))
