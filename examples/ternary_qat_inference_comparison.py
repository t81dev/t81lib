#!/usr/bin/env python3
"""Run a mini ternary QAT loop and measure quantization + inference costs."""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Callable
from typing import Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

import t81.torch as t81_torch
from t81.nn import Linear
from t81.trainer import TernaryTrainer, TernaryTrainingArguments

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RandomClassificationDataset(torch.utils.data.Dataset):
    """Synthetic dataset that feeds random floats + integer labels to the trainer."""

    def __init__(self, samples: int, input_dim: int, num_classes: int) -> None:
        self.inputs = torch.randn(samples, input_dim, dtype=torch.float32)
        self.labels = torch.randint(0, num_classes, (samples,), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"input": self.inputs[index], "labels": self.labels[index]}


class TinyClassifier(nn.Module):
    """Simple linear head that is eligible for ternary conversion."""

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, input: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        logits = self.linear(input)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits}


DEFAULT_INPUT_DIM = 32
DEFAULT_CLASSES = 4
DEFAULT_SAMPLES = 192
DEFAULT_BATCH = 32
DEFAULT_EPOCHS = 3
DEFAULT_VAL_SPLIT = 0.25
DEFAULT_ITERATIONS = 32


def build_datasets(
    samples: int,
    val_split: float,
    input_dim: int,
    num_classes: int,
    seed: int = 42,
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Create a train/validation split from a synthetic classification dataset."""

    if samples < 2:
        raise ValueError("samples must be at least 2 for a train/validation split")
    val_len = max(1, min(samples - 1, int(samples * val_split)))
    train_len = samples - val_len
    dataset = RandomClassificationDataset(samples, input_dim, num_classes)
    generator = torch.Generator().manual_seed(seed)
    return tuple(random_split(dataset, [train_len, val_len], generator=generator))


def evaluate_model(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    batch_size: int,
) -> dict[str, float]:
    """Run a simple eval pass and report accuracy + cross-entropy loss."""

    loader = DataLoader(dataset, batch_size=batch_size or DEFAULT_BATCH)
    device = next(model.parameters()).device if any(model.parameters()) else torch.device("cpu")
    criterion = nn.CrossEntropyLoss()
    was_training = model.training
    model.eval()
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
    return {"loss": loss_total / total, "accuracy": correct / total}


def build_float_baseline(trained_model: TinyClassifier) -> TinyClassifier:
    """Copy the trained ternary-ready parameters into a float-only baseline."""

    baseline = TinyClassifier(
        trained_model.linear.in_features,
        trained_model.linear.out_features,
    )
    baseline.linear.weight.data.copy_(trained_model.linear.weight.detach())
    if trained_model.linear.bias is not None and baseline.linear.bias is not None:
        baseline.linear.bias.data.copy_(trained_model.linear.bias.detach())
    return baseline


def compression_summary(linear: Linear) -> tuple[int, int, float]:
    """Estimate float vs ternary storage for a single linear module."""

    rows, cols = linear.weight.shape
    k_limbs = (cols + 47) // 48
    ternary_bytes = rows * k_limbs * 16
    float_bytes = linear.weight.numel() * linear.weight.element_size()
    ratio = float_bytes / ternary_bytes if ternary_bytes else float("inf")
    return float_bytes, ternary_bytes, ratio


def quantization_impact_report(
    trainer: TernaryTrainer,
    eval_dataset: torch.utils.data.Dataset,
    batch_size: int,
) -> None:
    """Log validation metrics before/after quantization for the trained module."""

    linear = trainer.model.linear
    quant_metrics = evaluate_model(trainer.model, eval_dataset, batch_size)
    baseline = build_float_baseline(trainer.model)
    baseline_metrics = evaluate_model(baseline, eval_dataset, batch_size)
    float_bytes, ternary_bytes, ratio = compression_summary(linear)
    logger.info(
        "validation quantization impact → float loss=%.4f acc=%.2f%%, ternary loss=%.4f acc=%.2f%%",
        baseline_metrics["loss"],
        baseline_metrics["accuracy"] * 100,
        quant_metrics["loss"],
        quant_metrics["accuracy"] * 100,
    )
    logger.info(
        "storage: float=%.2f KiB, ternary=%.2f KiB (%.2fx compression)",
        float_bytes / 1024,
        ternary_bytes / 1024,
        ratio,
    )


def _report_threshold_schedule(trainer: TernaryTrainer) -> None:
    """Print the effective threshold during the warmup phase."""

    warmup_steps = trainer.args.ternary_warmup_steps
    checkpoints = sorted({0, warmup_steps // 2, warmup_steps, warmup_steps + 10})
    saved_step = trainer.state.global_step
    print("\nTernary threshold schedule:")
    for step in checkpoints:
        trainer.state.global_step = max(step, 0)
        print(f"  step {step:3d} → threshold {trainer._effective_threshold():.4f}")
    trainer.state.global_step = saved_step


def run_qat_loop(
    train_dataset: torch.utils.data.Dataset,
    eval_dataset: torch.utils.data.Dataset,
    args: argparse.Namespace,
) -> TernaryTrainer:
    """Fine-tune a toy classifier with TernaryTrainer and log the threshold curve."""

    model = TinyClassifier(input_dim=args.input_dim, num_classes=args.num_classes)
    training_args = TernaryTrainingArguments(
        output_dir="examples/ternary_qat_output",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="no",
        learning_rate=args.learning_rate,
        ternary_threshold=args.threshold,
        ternary_warmup_steps=args.warmup_steps,
    )
    trainer = TernaryTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    _report_threshold_schedule(trainer)
    return trainer


def _timeit(func: Callable[[], None], iterations: int = 32) -> float:
    start = time.perf_counter()
    for _ in range(iterations):
        func()
    return (time.perf_counter() - start) / iterations


def compare_inference(
    weight: Optional[torch.Tensor] = None,
    threshold: float = 0.45,
    iterations: int = DEFAULT_ITERATIONS,
) -> Tuple[float, float]:
    """Measure latency of torch.matmul vs. ternary GEMM on cached weights."""

    device = torch.device("cpu")
    if weight is None:
        output_dim = input_dim = 64
        weight = torch.randn(output_dim, input_dim, device=device)
    else:
        weight = weight.to(device)
        output_dim, input_dim = weight.shape
    batch_size = 16
    example_input = torch.randn(batch_size, input_dim, device=device)
    ternary = t81_torch.TernaryTensor.from_float(weight, threshold=threshold)

    def torch_matmul() -> None:
        torch.matmul(example_input, weight.T)

    def ternary_gemm() -> None:
        ternary.matmul_input(example_input)

    torch_latency = _timeit(torch_matmul, iterations)
    ternary_latency = _timeit(ternary_gemm, iterations)
    return torch_latency, ternary_latency


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a ternary QAT loop with validation + timing.")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES, help="Total synthetic examples.")
    parser.add_argument("--val-split", type=float, default=DEFAULT_VAL_SPLIT, help="Validation ratio.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH, help="Train batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=DEFAULT_BATCH, help="Eval batch size.")
    parser.add_argument("--input-dim", type=int, default=DEFAULT_INPUT_DIM, help="Input dimensionality.")
    parser.add_argument("--num-classes", type=int, default=DEFAULT_CLASSES, help="Number of classes.")
    parser.add_argument("--threshold", type=float, default=0.45, help="Ternary threshold used during training.")
    parser.add_argument("--learning-rate", type=float, default=2e-3, help="Optimizer learning rate.")
    parser.add_argument("--warmup-steps", type=int, default=20, help="Ternary warmup steps.")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS, help="Inference timing iterations.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset split.")
    args = parser.parse_args()

    train_dataset, eval_dataset = build_datasets(
        args.samples,
        args.val_split,
        args.input_dim,
        args.num_classes,
        args.seed,
    )
    logger.info("running ternary QAT loop (%d epochs, %d samples total)", args.epochs, args.samples)
    trainer = run_qat_loop(train_dataset, eval_dataset, args)
    logger.info("trainer completed %d steps", trainer.state.global_step)
    quantization_impact_report(trainer, eval_dataset, args.eval_batch_size)
    torch_latency, ternary_latency = compare_inference(
        trainer.model.linear.weight.detach(),
        threshold=trainer.model.linear.ternary_threshold,
        iterations=args.iterations,
    )
    logger.info(
        "inference latency (per call) torch=%0.3f ms, ternary GEMM=%0.3f ms",
        torch_latency * 1e3,
        ternary_latency * 1e3,
    )


if __name__ == "__main__":
    main()
