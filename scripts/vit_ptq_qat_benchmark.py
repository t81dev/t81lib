#!/usr/bin/env python3
"""Benchmark ViT PTQ/QAT on CIFAR-10 with quick, reproducible baselines."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification

from t81.convert import convert
from t81.nn import Linear as TernaryLinear
from t81.trainer import TernaryTrainer, TernaryTrainingArguments


@dataclass
class StageMetrics:
    size_gib: float
    accuracy: float
    loss: float
    images_per_s: float


@dataclass
class BenchmarkSummary:
    model_id: str
    dataset: str
    device: str
    threshold: float
    baseline: StageMetrics
    ptq: StageMetrics
    qat: StageMetrics | None


def bytes_to_gib(num_bytes: int) -> float:
    return num_bytes / (1024**3)


def model_param_bytes(model: torch.nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


def ternary_weight_bytes(model: torch.nn.Module) -> int:
    total = 0
    for module in model.modules():
        if isinstance(module, TernaryLinear):
            rows, cols = module.weight.shape
            k_limbs = (cols + 47) // 48
            total += rows * k_limbs * 16
            if module.bias is not None:
                total += module.bias.numel() * module.bias.element_size()
    return total


def collate_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.tensor([item["labels"] for item in batch])
    return {"pixel_values": pixel_values, "labels": labels}


@torch.inference_mode()
def evaluate_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int,
) -> dict[str, float]:
    criterion = torch.nn.CrossEntropyLoss()
    total = correct = 0
    loss_sum = 0.0
    model.eval()
    for idx, batch in enumerate(loader):
        if max_batches and idx >= max_batches:
            break
        inputs = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(pixel_values=inputs)
        logits = outputs.logits
        loss = criterion(logits, labels)
        loss_sum += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    accuracy = correct / max(total, 1)
    avg_loss = loss_sum / max(total, 1)
    return {"accuracy": accuracy, "loss": avg_loss}


@torch.inference_mode()
def measure_throughput(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int,
) -> float:
    model.eval()
    warmup = min(1, len(loader))
    for idx, batch in enumerate(loader):
        if idx >= warmup:
            break
        inputs = batch["pixel_values"].to(device)
        _ = model(pixel_values=inputs)
    start = time.perf_counter()
    images = 0
    for idx, batch in enumerate(loader):
        if max_batches and idx >= max_batches:
            break
        inputs = batch["pixel_values"].to(device)
        _ = model(pixel_values=inputs)
        images += inputs.size(0)
    elapsed = time.perf_counter() - start
    return images / max(elapsed, 1e-6)


def run(args: argparse.Namespace) -> BenchmarkSummary:
    device = torch.device(args.device)
    tqdm.write("Loading dataset + processor...")
    dataset = load_dataset("cifar10")
    train_split = dataset["train"].select(range(args.max_train_samples))
    eval_split = dataset["test"].select(range(args.max_eval_samples))

    processor = AutoImageProcessor.from_pretrained(args.model_id, use_fast=True)

    def preprocess(batch: dict[str, list]) -> dict[str, list]:
        processed = processor(images=batch["img"], return_tensors="pt")
        return {"pixel_values": processed["pixel_values"], "labels": batch["label"]}

    train_split = train_split.map(preprocess, batched=True, remove_columns=["img", "label"])
    eval_split = eval_split.map(preprocess, batched=True, remove_columns=["img", "label"])
    train_split.set_format(type="torch")
    eval_split.set_format(type="torch")

    train_loader = torch.utils.data.DataLoader(
        train_split, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_split, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch
    )

    tqdm.write("Loading baseline model...")
    model = AutoModelForImageClassification.from_pretrained(
        args.model_id,
        num_labels=10,
        ignore_mismatched_sizes=True,
    ).to(device)

    tqdm.write("Measuring baseline metrics...")
    baseline_bytes = model_param_bytes(model)
    eval_batches = args.eval_batches if args.max_eval_batches is None else args.max_eval_batches
    throughput_batches = max(1, eval_batches) if eval_batches >= 0 else 1
    if eval_batches == 0:
        baseline_eval = {"accuracy": 0.0, "loss": 0.0}
    else:
        baseline_eval = evaluate_model(model, eval_loader, device, eval_batches)
    baseline_throughput = (
        0.0
        if args.skip_throughput
        else measure_throughput(model, eval_loader, device, throughput_batches)
    )
    tqdm.write("Baseline metrics captured.")
    baseline = StageMetrics(
        size_gib=bytes_to_gib(baseline_bytes),
        accuracy=baseline_eval["accuracy"],
        loss=baseline_eval["loss"],
        images_per_s=baseline_throughput,
    )

    tqdm.write("Running PTQ conversion...")
    model = convert(model, threshold=args.threshold, inplace=True)
    model.eval()
    ptq_bytes = ternary_weight_bytes(model)
    if eval_batches == 0:
        ptq_eval = {"accuracy": 0.0, "loss": 0.0}
    else:
        ptq_eval = evaluate_model(model, eval_loader, device, eval_batches)
    tqdm.write("Measuring PTQ metrics...")
    ptq_throughput = (
        0.0
        if args.skip_throughput
        else measure_throughput(model, eval_loader, device, throughput_batches)
    )
    tqdm.write("PTQ metrics captured.")
    ptq = StageMetrics(
        size_gib=bytes_to_gib(ptq_bytes),
        accuracy=ptq_eval["accuracy"],
        loss=ptq_eval["loss"],
        images_per_s=ptq_throughput,
    )

    qat_metrics = None
    if args.run_qat:
        tqdm.write("Running short QAT loop...")
        qat_args = TernaryTrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            max_steps=args.qat_steps,
            logging_steps=max(1, args.qat_steps // 5),
            save_steps=0,
            save_strategy="no",
            learning_rate=args.learning_rate,
            ternary_threshold=args.threshold,
            ternary_warmup_steps=max(1, args.qat_steps // 2),
            report_to=[],
            remove_unused_columns=False,
            dataloader_pin_memory=False,
        )
        trainer = TernaryTrainer(
            model=model,
            args=qat_args,
            train_dataset=train_split,
            data_collator=collate_batch,
        )
        trainer.train()
        model = trainer.model
        model.eval()
        qat_bytes = ternary_weight_bytes(model)
        if eval_batches == 0:
            qat_eval = {"accuracy": 0.0, "loss": 0.0}
        else:
            qat_eval = evaluate_model(model, eval_loader, device, eval_batches)
        qat_throughput = (
            0.0
            if args.skip_throughput
            else measure_throughput(model, eval_loader, device, throughput_batches)
        )
        qat_metrics = StageMetrics(
            size_gib=bytes_to_gib(qat_bytes),
            accuracy=qat_eval["accuracy"],
            loss=qat_eval["loss"],
            images_per_s=qat_throughput,
        )

    return BenchmarkSummary(
        model_id=args.model_id,
        dataset="cifar10",
        device=str(device),
        threshold=args.threshold,
        baseline=baseline,
        ptq=ptq,
        qat=qat_metrics,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="ViT PTQ/QAT benchmark on CIFAR-10.")
    parser.add_argument("--model-id", default="google/vit-base-patch16-224")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-train-samples", type=int, default=2048)
    parser.add_argument("--max-eval-samples", type=int, default=512)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=None,
        help="Override eval batches; set 0 to skip accuracy/loss and run throughput only.",
    )
    parser.add_argument(
        "--skip-throughput",
        action="store_true",
        help="Skip images/s measurements (useful for quick size/accuracy baselines).",
    )
    parser.add_argument("--run-qat", action="store_true")
    parser.add_argument("--qat-steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--output-dir", default="vit-qat")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    summary = run(args)

    print("\nSummary")
    print(f"Model: {summary.model_id}")
    print(f"Baseline size GiB: {summary.baseline.size_gib:.2f}")
    print(f"Baseline acc/loss: {summary.baseline.accuracy:.4f}/{summary.baseline.loss:.4f}")
    print(f"Baseline images/s: {summary.baseline.images_per_s:.2f}")
    print(f"PTQ size GiB: {summary.ptq.size_gib:.2f}")
    print(f"PTQ acc/loss: {summary.ptq.accuracy:.4f}/{summary.ptq.loss:.4f}")
    print(f"PTQ images/s: {summary.ptq.images_per_s:.2f}")
    if summary.qat is not None:
        print(f"QAT size GiB: {summary.qat.size_gib:.2f}")
        print(f"QAT acc/loss: {summary.qat.accuracy:.4f}/{summary.qat.loss:.4f}")
        print(f"QAT images/s: {summary.qat.images_per_s:.2f}")

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(summary)
        with args.json_output.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
