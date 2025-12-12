#!/usr/bin/env python3
"""Run a benchmark that compares FP32, PTQ, and QAT on a small fashion MNIST model."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader, Dataset, random_split

import t81lib
from t81.trainer import TernaryTrainer, TernaryTrainingArguments
from transformers import PreTrainedModel, PretrainedConfig
from transformers import TrainingArguments


class FlattenedFashionMNIST(Dataset):
    """Fashion MNIST whose samples are flattened to a 1-D vector."""

    def __init__(self, data_dir: Path, train: bool) -> None:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda tensor: tensor.view(-1))]
        )
        self.dataset = FashionMNIST(
            root=str(data_dir),
            train=train,
            download=True,
            transform=transform,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample, label = self.dataset[index]
        return {"input": sample, "labels": label}


class TinyClassifier(nn.Module):
    """Simple linear classifier used for the FP32 baseline."""

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.linear(inputs)
        return {"logits": logits}


class TinyConfig(PretrainedConfig):
    model_type = "tiny_classifier"

    def __init__(self, input_dim: int, num_classes: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.num_labels = num_classes


class TinyPretrainedModel(PreTrainedModel):
    config_class = TinyConfig

    def __init__(self, config: TinyConfig) -> None:
        super().__init__(config)
        self.linear = nn.Linear(config.input_dim, config.num_labels)
        self.post_init()

    def forward(
        self,
        input: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        logits = self.linear(input)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits}


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Compute accuracy and loss over a single dataset split."""

    was_training = model.training
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total = correct = 0
    loss_total = 0.0
    with torch.no_grad():
        for batch in loader:
            inputs = batch["input"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(inputs)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            loss = criterion(logits, labels)
            predictions = logits.argmax(dim=-1)
            loss_total += loss.item() * inputs.size(0)
            correct += (predictions == labels).sum().item()
            total += labels.numel()
    model.train(mode=was_training)
    return {"accuracy": correct / total, "loss": loss_total / total}


def measure_latency(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    iterations: int,
) -> float:
    """Measure average inference latency (seconds) over `iterations` batches."""

    model.eval()
    start = time.perf_counter()
    for idx, batch in enumerate(loader):
        if idx >= iterations:
            break
        inputs = batch["input"].to(device)
        with torch.no_grad():
            _ = model(inputs)
    elapsed = time.perf_counter() - start
    return elapsed / min(iterations, len(loader))


def train_fp32(
    model: TinyClassifier,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> None:
    """Train the FP32 model with a simple SGD optimizer."""

    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for batch in loader:
            optimizer.zero_grad()
            inputs = batch["input"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(inputs)["logits"]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


def quantize_tensor(tensor: torch.Tensor, threshold: float) -> torch.Tensor:
    """Use `t81lib.quantize_to_trits` to clamp a tensor to {-1, 0, 1} and return float32."""

    flattened = tensor.detach().cpu().numpy().astype("float32")
    trits = t81lib.quantize_to_trits(flattened, threshold)
    return torch.from_numpy(trits.astype("float32").reshape(tensor.shape)).to(tensor.device)


def quantize_model(model: TinyClassifier, threshold: float) -> TinyClassifier:
    """Copy the FP32 model parameters, quantize each tensor, and return a new module."""

    quantized = TinyClassifier(model.linear.in_features, model.linear.out_features)
    quantized.load_state_dict(model.state_dict())
    for name, param in quantized.named_parameters():
        param.data.copy_(quantize_tensor(param.data, threshold))
    return quantized


def compute_tensor_bytes(tensor: torch.Tensor) -> int:
    """Estimate ternary storage by counting the required limbs (48 trits per 16 bytes)."""

    limps = (tensor.numel() + 47) // 48
    return limps * 16


def collect_model_size(model: nn.Module) -> int:
    """Return the raw byte footprint of a model's parameters."""

    return sum(p.numel() * p.element_size() for p in model.parameters())


def write_benchmark_results(output: Path, rows: list[dict[str, object]]) -> None:
    """Append benchmark rows to CSV (creates header when missing)."""

    output.parent.mkdir(parents=True, exist_ok=True)
    exists = output.exists()
    with output.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tensor benchmark for FP32, PTQ, and QAT.")
    parser.add_argument("--data-dir", default="data", help="Directory to store Fashion MNIST.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training/eval.")
    parser.add_argument("--fp32-epochs", type=int, default=3, help="FP32 training epochs.")
    parser.add_argument("--qat-epochs", type=int, default=3, help="Quantization-aware training epochs.")
    parser.add_argument("--threshold", type=float, default=0.4, help="Ternary quantization threshold.")
    parser.add_argument("--device", default="cpu", help="Torch device to run the benchmark on.")
    parser.add_argument("--latency-iters", type=int, default=16, help="Batches to use for latency timing.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/quantization_runs.csv"),
        help="CSV file where benchmark rows are appended.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dataset = FlattenedFashionMNIST(Path(args.data_dir), train=True)
    val_size = len(dataset) // 6
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    baseline = TinyClassifier(input_dim=28 * 28, num_classes=10)
    train_fp32(baseline, train_loader, device, args.fp32_epochs, lr=0.05)
    baseline_metrics = evaluate_model(baseline, val_loader, device)
    fp32_latency = measure_latency(baseline, val_loader, device, args.latency_iters)
    fp32_size = collect_model_size(baseline)

    ptq_model = quantize_model(baseline, args.threshold).to(device)
    ptq_metrics = evaluate_model(ptq_model, val_loader, device)
    ptq_latency = measure_latency(ptq_model, val_loader, device, args.latency_iters)
    ptq_ternary_bytes = sum(compute_tensor_bytes(param) for param in ptq_model.parameters())

    config = TinyConfig(input_dim=28 * 28, num_classes=10)
    qat_model = TinyPretrainedModel(config)
    training_args = TernaryTrainingArguments(
        output_dir="scripts/ternary_quantization_benchmark_output",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.qat_epochs,
        learning_rate=0.01,
        ternary_threshold=args.threshold,
    )
    training_args.evaluation_strategy = "epoch"
    training_args.save_strategy = "no"
    training_args.logging_strategy = "epoch"
    trainer = TernaryTrainer(
        model=qat_model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
    )
    trainer.train()
    qat_model = trainer.model.to(device)
    qat_metrics = evaluate_model(qat_model, val_loader, device)
    qat_latency = measure_latency(qat_model, val_loader, device, args.latency_iters)
    qat_linear = trainer.model.linear  # type: ignore[assignment]
    qat_size = sum(compute_tensor_bytes(param) for param in qat_linear.parameters())

    results = [
        {
            "mode": "fp32",
            "accuracy": baseline_metrics["accuracy"],
            "loss": baseline_metrics["loss"],
            "latency_s": fp32_latency,
            "bytes": fp32_size,
        },
        {
            "mode": "ptq",
            "accuracy": ptq_metrics["accuracy"],
            "loss": ptq_metrics["loss"],
            "latency_s": ptq_latency,
            "bytes": ptq_ternary_bytes,
        },
        {
            "mode": "qat",
            "accuracy": qat_metrics["accuracy"],
            "loss": qat_metrics["loss"],
            "latency_s": qat_latency,
            "bytes": qat_size,
        },
    ]

    print("Benchmark summary (accuracy, latency, storage):")
    for row in results:
        print(
            f"  {row['mode']:>3} → acc={row['accuracy']:.3f}, latency={row['latency_s']:.4f}s, "
            f"bytes={row['bytes']:,}"
        )

    write_benchmark_results(args.output, results)


if __name__ == "__main__":
    main()
