#!/usr/bin/env python3
"""Run a mini ternary QAT loop and compare inference timing (torch.matmul vs ternary GEMM)."""

from __future__ import annotations

import time
from collections.abc import Callable

import torch
from torch import nn
from transformers import TrainingArguments

import t81.torch as t81_torch
from t81.nn import Linear
from t81.trainer import TernaryTrainer, TernaryTrainingArguments


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


def run_qat_loop() -> TernaryTrainer:
    """Fine-tune a toy classifier with TernaryTrainer and log the threshold curve."""

    dataset = RandomClassificationDataset(samples=128, input_dim=32, num_classes=4)
    model = TinyClassifier(input_dim=32, num_classes=4)
    training_args = TernaryTrainingArguments(
        output_dir="examples/ternary_qat_output",
        per_device_train_batch_size=32,
        num_train_epochs=1,
        logging_strategy="no",
        evaluation_strategy="no",
        save_strategy="no",
        learning_rate=2e-3,
        ternary_threshold=0.45,
        ternary_warmup_steps=20,
    )
    trainer = TernaryTrainer(model=model, args=training_args, train_dataset=dataset)
    trainer.train()
    _report_threshold_schedule(trainer)
    return trainer


def _timeit(func: Callable[[], None], iterations: int = 32) -> float:
    start = time.perf_counter()
    for _ in range(iterations):
        func()
    return (time.perf_counter() - start) / iterations


def compare_inference() -> None:
    """Measure latency of torch.matmul vs. ternary GEMM on cached weights."""

    batch_size = 16
    input_dim = 64
    output_dim = 64
    device = torch.device("cpu")
    example_input = torch.randn(batch_size, input_dim, device=device)
    weight = torch.randn(output_dim, input_dim, device=device)
    ternary = t81_torch.TernaryTensor.from_float(weight, threshold=0.45)

    def torch_matmul() -> None:
        torch.matmul(example_input, weight.T)

    def ternary_gemm() -> None:
        ternary.matmul_input(example_input)

    torch_latency = _timeit(torch_matmul)
    ternary_latency = _timeit(ternary_gemm)
    print("\nInference timing (mean per call):")
    print(f"  torch.matmul : {torch_latency * 1e3:.3f} ms")
    print(f"  ternary GEMM : {ternary_latency * 1e3:.3f} ms")


def main() -> None:
    print("Running synthetic ternary QAT loop...")
    trainer = run_qat_loop()
    print(f"\nTrainer reported {trainer.state.global_step} steps and saved threshold history.")
    compare_inference()


if __name__ == "__main__":
    main()
