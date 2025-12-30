#!/usr/bin/env python3
"""Benchmark Phi-3-mini PTQ/QAT with lightweight progress reporting."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from transformers import TrainerCallback, TrainerState, TrainerControl

import t81 as t8


@dataclass
class StageMetrics:
    size_gib: float
    ppl: float | None
    tok_s: float


@dataclass
class BenchResults:
    model_id: str
    dataset: str
    device: str
    dtype: str
    threshold: float
    max_eval_tokens: int
    eval_texts: int
    max_new_tokens: int
    skip_latency: bool
    skip_ptq_ppl: bool
    run_qat: bool
    qat_steps: int
    train_split: str
    learning_rate: float
    compression_ratio: float
    baseline: StageMetrics
    ptq: StageMetrics
    qat: StageMetrics | None = None


class QATProgressCallback(TrainerCallback):
    def __init__(self, total_steps: int):
        self._bar = tqdm(total=total_steps, desc="QAT steps", unit="step")

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self._bar.update(1)
        return control

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self._bar.close()
        return control


def bytes_to_gib(num_bytes: int) -> float:
    return num_bytes / (1024**3)


def model_param_bytes(model: torch.nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


def ternary_weight_bytes(model: torch.nn.Module) -> int:
    total = 0
    for module in model.modules():
        if isinstance(module, t8.Linear):
            rows, cols = module.weight.shape
            k_limbs = (cols + 47) // 48
            total += rows * k_limbs * 16
            if module.bias is not None:
                total += module.bias.numel() * module.bias.element_size()
    return total


def measure_generate_latency(model, tokenizer, prompt: str, max_new_tokens: int) -> float:
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    _ = model.generate(**inputs, max_new_tokens=1)
    start = time.perf_counter()
    _ = model.generate(**inputs, max_new_tokens=max_new_tokens)
    elapsed = time.perf_counter() - start
    return max_new_tokens / max(elapsed, 1e-6)


@torch.inference_mode()
def perplexity(model, tokenizer, dataset, max_tokens: int, eval_texts: int) -> float:
    model.eval()
    raw_texts = [text for text in dataset["text"] if isinstance(text, str) and text.strip()]
    if not raw_texts:
        raise ValueError("No non-empty text samples available for perplexity.")
    text = "\n\n".join(raw_texts[:eval_texts])
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens,
    )
    input_ids = enc["input_ids"][0][:max_tokens]
    input_ids = input_ids.unsqueeze(0).to(next(model.parameters()).device)
    labels = input_ids.clone()
    outputs = model(input_ids=input_ids, labels=labels)
    return torch.exp(outputs.loss).item()


def run(args: argparse.Namespace) -> BenchResults:
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    device_map = "cpu" if args.device == "cpu" else "auto"

    phases = tqdm(total=6, desc="Phi-3 benchmark phases", unit="phase")

    tqdm.write("Loading tokenizer/model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map=device_map,
    )
    model.eval()
    phases.update(1)

    tqdm.write("Measuring baseline size/latency...")
    baseline_bytes = model_param_bytes(model)
    prompt = "Summarize balanced ternary quantization in one paragraph."
    baseline_tok_s = 0.0
    if not args.skip_latency:
        baseline_tok_s = measure_generate_latency(model, tokenizer, prompt, args.max_new_tokens)
    phases.update(1)

    tqdm.write("Loading wikitext-2 + baseline perplexity...")
    wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    baseline_ppl = perplexity(model, tokenizer, wikitext, args.max_eval_tokens, args.eval_texts)
    phases.update(1)

    tqdm.write("Running PTQ conversion...")
    model = t8.convert(model, threshold=args.threshold, inplace=True)
    model.eval()
    ternary_bytes = ternary_weight_bytes(model)
    ptq_tok_s = 0.0
    if not args.skip_latency:
        ptq_tok_s = measure_generate_latency(model, tokenizer, prompt, args.max_new_tokens)
    if args.skip_ptq_ppl:
        ptq_ppl = None
    else:
        ptq_ppl = perplexity(model, tokenizer, wikitext, args.max_eval_tokens, args.eval_texts)
    phases.update(1)

    qat_ppl = None
    qat_tok_s = None
    if args.run_qat:
        tqdm.write("Running short QAT loop...")
        train_split = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split=args.train_split,
        )
        tokenized = train_split.map(lambda x: tokenizer(x["text"], truncation=True), batched=True)
        tokenized = tokenized.remove_columns(["text"])
        tokenized = tokenized.filter(lambda x: len(x["input_ids"]) > 1)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        qat_args = t8.TernaryTrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=1,
            max_steps=args.qat_steps,
            logging_steps=max(1, args.qat_steps // 5),
            save_steps=0,
            save_strategy="no",
            learning_rate=args.learning_rate,
            ternary_threshold=args.threshold,
            ternary_warmup_steps=max(1, args.qat_steps // 2),
            report_to=[],
            use_cpu=True,
            dataloader_pin_memory=False,
        )

        trainer = t8.TernaryTrainer(
            model=model,
            args=qat_args,
            train_dataset=tokenized,
            data_collator=data_collator,
            callbacks=[QATProgressCallback(args.qat_steps)],
        )
        trainer.train()
        model = trainer.model
        model.eval()

        qat_ppl = perplexity(model, tokenizer, wikitext, args.max_eval_tokens, args.eval_texts)
        if not args.skip_latency:
            qat_tok_s = measure_generate_latency(model, tokenizer, prompt, args.max_new_tokens)
    phases.update(1)
    phases.close()

    compression_ratio = baseline_bytes / max(ternary_bytes, 1)
    return BenchResults(
        model_id=args.model_id,
        dataset="wikitext-2-raw-v1",
        device=args.device,
        dtype=args.dtype,
        threshold=args.threshold,
        max_eval_tokens=args.max_eval_tokens,
        eval_texts=args.eval_texts,
        max_new_tokens=args.max_new_tokens,
        skip_latency=args.skip_latency,
        skip_ptq_ppl=args.skip_ptq_ppl,
        run_qat=args.run_qat,
        qat_steps=args.qat_steps,
        train_split=args.train_split,
        learning_rate=args.learning_rate,
        compression_ratio=compression_ratio,
        baseline=StageMetrics(
            size_gib=bytes_to_gib(baseline_bytes),
            ppl=baseline_ppl,
            tok_s=baseline_tok_s,
        ),
        ptq=StageMetrics(
            size_gib=bytes_to_gib(ternary_bytes),
            ppl=ptq_ppl,
            tok_s=ptq_tok_s,
        ),
        qat=(
            None
            if qat_ppl is None
            else StageMetrics(
                size_gib=bytes_to_gib(ternary_bytes),
                ppl=qat_ppl,
                tok_s=qat_tok_s or 0.0,
            )
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Phi-3-mini PTQ/QAT benchmark.")
    parser.add_argument("--model-id", default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--device", choices=["cpu", "auto"], default="cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--max-eval-tokens", type=int, default=1024)
    parser.add_argument("--eval-texts", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument(
        "--skip-ptq-ppl",
        action="store_true",
        help="Skip PTQ perplexity (useful when CPU runs are too slow).",
    )
    parser.add_argument("--run-qat", action="store_true")
    parser.add_argument("--qat-steps", type=int, default=50)
    parser.add_argument("--train-split", default="train[:1%]")
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--output-dir", default="phi3-qat")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    results = run(args)

    print("\nSummary")
    print(f"Baseline size GiB: {results.baseline.size_gib:.2f}")
    print(f"PTQ size GiB: {results.ptq.size_gib:.2f}")
    print(f"Compression ratio: {results.compression_ratio:.1f}x")
    print(f"Baseline PPL: {results.baseline.ppl:.2f}")
    if results.ptq.ppl is None:
        print("PTQ PPL: pending")
    else:
        print(f"PTQ PPL: {results.ptq.ppl:.2f}")
    print(f"Baseline tok/s: {results.baseline.tok_s:.2f}")
    print(f"PTQ tok/s: {results.ptq.tok_s:.2f}")
    if results.qat is not None:
        print(f"QAT PPL: {results.qat.ppl:.2f}")
        print(f"QAT tok/s: {results.qat.tok_s:.2f}")

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(results)
        with args.json_output.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
