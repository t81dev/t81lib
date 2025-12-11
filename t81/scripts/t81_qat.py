"""CLI that fine-tunes Hugging Face models with t81lib's ternary QAT stack."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

import torch

from t81.trainer import TernaryTrainer, TernaryTrainingArguments

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling


def _parse_dtype(value: str) -> torch.dtype:
    normalized = value.replace("-", "_").lower()
    if not hasattr(torch, normalized):
        raise argparse.ArgumentTypeError(f"{value} is not a torch.dtype")
    return getattr(torch, normalized)


def _find_text_column(full_dataset: "Dataset") -> str:
    for name, feature in full_dataset.features.items():
        if getattr(feature, "dtype", None) == "string":
            return name
    raise ValueError("Dataset does not expose a string column for text inputs")


def _group_texts(examples: dict[str, list[list[int]]], block_size: int) -> dict[str, list[list[int]]]:
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(next(iter(concatenated.values()), []))
    total_length = (total_length // block_size) * block_size
    result: dict[str, list[list[int]]] = {}
    for key, token_list in concatenated.items():
        chunks = [
            token_list[i : i + block_size]
            for i in range(0, total_length, block_size)
        ]
        result[key] = chunks
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="t81 ternary QAT runner")
    parser.add_argument("model_name_or_path", help="Hugging Face model identifier or local path")
    parser.add_argument("--dataset-name", required=True, help="HF dataset name for QAT")
    parser.add_argument("--dataset-config-name", help="Optional dataset configuration")
    parser.add_argument("--dataset-cache-dir", help="Cache directory for datasets")
    parser.add_argument("--output-dir", required=True, help="Where to save ternary checkpoints")
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--torch-dtype", type=_parse_dtype, help="Optional torch dtype like bfloat16")
    parser.add_argument("--ternary-threshold", type=float, default=0.44)
    parser.add_argument("--ternary-stochastic-rounding", action="store_true")
    parser.add_argument("--ternary-warmup-steps", type=int, default=100)
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--max-train-samples", type=int, help="Truncate training samples for quick experiments")
    args = parser.parse_args()

    try:
        import datasets
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
    except ImportError as exc:  # pragma: no cover - optional dependencies
        raise RuntimeError(
            "t81lib's QAT CLI requires 'datasets' and 'transformers'; "
            "install them via 'pipx inject t81lib torch transformers accelerate'"
        ) from exc

    dataset = datasets.load_dataset(
        args.dataset_name,
        args.dataset_config_name,
        cache_dir=args.dataset_cache_dir,
    )
    train_dataset = dataset.get("train") or dataset.get("validation")
    if train_dataset is None:
        raise RuntimeError("Dataset must expose a train or validation split")

    text_column = _find_text_column(train_dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(examples: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        texts = examples[text_column]
        if isinstance(texts, str):
            texts = [texts]
        return tokenizer(texts, return_attention_mask=False, add_special_tokens=True)

    tokenized = train_dataset.map(
        tokenize,
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    if args.max_train_samples is not None:
        tokenized = tokenized.select(range(min(args.max_train_samples, len(tokenized))))

    grouped = tokenized.map(
        lambda examples: _group_texts(examples, args.block_size),
        batched=True,
        remove_columns=tokenized.column_names,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=args.torch_dtype,
    )

    training_args = TernaryTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_steps=args.ternary_warmup_steps,
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        report_to="none",
        torch_dtype=args.torch_dtype,
        ternary_threshold=args.ternary_threshold,
        ternary_stochastic_rounding=args.ternary_stochastic_rounding,
        ternary_warmup_steps=args.ternary_warmup_steps,
    )

    trainer = TernaryTrainer(
        model=model,
        args=training_args,
        train_dataset=grouped,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
