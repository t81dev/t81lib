#!/usr/bin/env python3
"""Generate reproducible quantization reports for multiple synthetic configs."""

from __future__ import annotations

import argparse
import csv
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from t81.nn import Linear as TernaryLinear


DEFAULT_EPOCHS = 2
DEFAULT_BATCH = 32
DEFAULT_VAL_SPLIT = 0.2
DEFAULT_ITERATIONS = 32


class RandomClassificationDataset(Dataset):
    """A fixed random classification corpus used across configs."""

    def __init__(self, samples: int, input_dim: int, num_classes: int, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.inputs = torch.randn(samples, input_dim, generator=generator)
        self.labels = torch.randint(0, num_classes, (samples,), generator=generator)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.labels[index]


@dataclass
class Config:
    """Defines a configuration that will be benchmarked."""

    name: str
    dims: Sequence[int]
    samples: int
    num_classes: int
    threshold: float
    learning_rate: float = 1e-3
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH
    val_split: float = DEFAULT_VAL_SPLIT
    iterations: int = DEFAULT_ITERATIONS
    seed: int = 42


class MultiLinearModel(nn.Module):
    def __init__(self, dims: Sequence[int], threshold: float, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[tuple[nn.Module, str]] = []
        for in_dim, out_dim in zip(dims, dims[1:]):
            layers.append((TernaryLinear(in_dim, out_dim, threshold=threshold), "linear"))
            layers.append((nn.ReLU(inplace=True), "relu"))
        if layers and layers[-1][1] == "relu":
            layers.pop()
        self.layers = nn.ModuleList([layer for layer, _ in layers])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class FloatMultiLinearModel(nn.Module):
    def __init__(self, dims: Sequence[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims, dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU(inplace=True))
        if layers and isinstance(layers[-1], nn.ReLU):
            layers.pop()
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def build_dataset(config: Config) -> tuple[Dataset, Dataset]:
    dataset = RandomClassificationDataset(
        config.samples,
        config.dims[0],
        config.num_classes,
        config.seed,
    )
    val_len = max(1, int(len(dataset) * config.val_split))
    train_len = len(dataset) - val_len
    return tuple(random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(config.seed)))


def evaluate(model: nn.Module, loader: DataLoader) -> tuple[float, float]:
    model.eval()
    total = correct = 0
    loss_total = 0.0
    criterion = F.cross_entropy
    with torch.no_grad():
        for inputs, labels in loader:
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss_total += loss.item() * inputs.size(0)
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)
    return loss_total / total, correct / total


def clone_to_float(model: MultiLinearModel, dims: Sequence[int]) -> FloatMultiLinearModel:
    float_model = FloatMultiLinearModel(dims)
    ternary_layers = [layer for layer in model.layers if isinstance(layer, TernaryLinear)]
    float_layers = [layer for layer in float_model.layers if isinstance(layer, nn.Linear)]
    for ternary_layer, float_layer in zip(ternary_layers, float_layers):
        float_layer.weight.data.copy_(ternary_layer.weight.detach())
        if ternary_layer.bias is not None and float_layer.bias is not None:
            float_layer.bias.data.copy_(ternary_layer.bias.detach())
    return float_model


def train_model(model: MultiLinearModel, loader: DataLoader, config: Config) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = F.cross_entropy
    model.train()
    for _ in range(config.epochs):
        for inputs, labels in loader:
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()


def _input_dim(model: nn.Module) -> int:
    if isinstance(model, MultiLinearModel):
        for layer in model.layers:
            if isinstance(layer, TernaryLinear):
                return layer.in_features
    if isinstance(model, FloatMultiLinearModel):
        for layer in model.layers:
            if isinstance(layer, nn.Linear):
                return layer.in_features
    raise ValueError("unable to determine input dim")


def time_forward(model: nn.Module, batch_size: int, iterations: int) -> float:
    input_dim = _input_dim(model)
    sample = torch.randn(batch_size, input_dim)
    model.eval()
    start = time.perf_counter()
    for _ in range(iterations):
        with torch.no_grad():
            model(sample)
    return (time.perf_counter() - start) / iterations


def format_bytes(n_bytes: int) -> str:
    return f"{n_bytes / 1024:.2f} KiB"


def run_configs(configs: Iterable[Config], output: Path | None) -> None:
    rows: list[dict[str, str]] = []
    for config in configs:
        train_dataset, val_dataset = build_dataset(config)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

        model = MultiLinearModel(config.dims, threshold=config.threshold)
        train_model(model, train_loader, config)

        val_loss, val_acc = evaluate(model, val_loader)
        float_model = clone_to_float(model, config.dims)
        float_loss, float_acc = evaluate(float_model, val_loader)

        ternary_latency = time_forward(model, config.batch_size, config.iterations)
        float_latency = time_forward(float_model, config.batch_size, config.iterations)

        ternary_bytes = sum(
            (layer.weight.numel() * layer.weight.element_size())
            for layer in model.layers
            if isinstance(layer, TernaryLinear)
        )
        float_bytes = sum(layer.weight.numel() * layer.weight.element_size() for layer in float_model.layers if isinstance(layer, nn.Linear))

        rows.append(
            {
                "config": config.name,
                "val_loss_quant": f"{val_loss:.4f}",
                "val_acc_quant": f"{val_acc:.2f}",
                "val_loss_float": f"{float_loss:.4f}",
                "val_acc_float": f"{float_acc:.2f}",
                "ternary_latency_ms": f"{ternary_latency * 1e3:.3f}",
                "float_latency_ms": f"{float_latency * 1e3:.3f}",
                "ternary_bytes": format_bytes(ternary_bytes),
                "float_bytes": format_bytes(float_bytes),
            }
        )

    if output:
        with output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Generated quantization report at {output}")
    else:
        print("Quantization report:")
        for row in rows:
            print(", ".join(f"{k}={v}" for k, v in row.items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark multiple quantization configurations.")
    parser.add_argument("--output", type=Path, help="Optional CSV file to write the report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = [
        Config(name="tiny", dims=[32, 32, 16], samples=256, num_classes=4, threshold=0.45),
        Config(name="wide", dims=[64, 64, 32], samples=512, num_classes=8, threshold=0.5),
        Config(name="deep", dims=[48, 32, 16, 8], samples=384, num_classes=6, threshold=0.4),
    ]
    run_configs(configs, args.output)


if __name__ == "__main__":
    main()
