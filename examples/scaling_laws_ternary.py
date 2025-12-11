"""Run a miniature scaling-law story with exact ternary attention."""

from __future__ import annotations

import argparse
import itertools
import logging
import math
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import t81.torch as t81_torch
from t81.nn import ExactRMSNorm, scaling_laws_experiment, ternary_rope, ternary_softmax

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VOCAB_SIZE = 64
DEFAULT_SEQ = 32
DEFAULT_BATCH = 8


class ToyTokenizer:
    def __init__(self, text: str) -> None:
        self.text = text

    def tokens(self, seq_len: int, batch_size: int) -> DataLoader:
        text = self.text
        min_len = seq_len + 2
        if len(text) < min_len:
            text = text * ((min_len // len(text)) + 1)
        tokens = torch.tensor([ord(ch) % VOCAB_SIZE for ch in text], dtype=torch.long)
        horizon = len(tokens) - seq_len - 1
        inputs = torch.stack([tokens[i : i + seq_len] for i in range(horizon)])
        targets = torch.stack([tokens[i + 1 : i + seq_len + 1] for i in range(horizon)])
        dataset = TensorDataset(inputs, targets)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


def build_loader(seq_len: int = DEFAULT_SEQ, batch_size: int = DEFAULT_BATCH) -> DataLoader:
    text = "The ternary scaling story refuses to fade. "
    tokenizer = ToyTokenizer(text)
    loader = tokenizer.tokens(seq_len, batch_size)
    return loader


class DemoBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = ExactRMSNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.scale = 1.0 / math.sqrt(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q = ternary_rope(q)
        k = ternary_rope(k)
        attn = ternary_softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
        x = torch.matmul(attn, v)
        return self.proj(x) + residual


class DemoModel(nn.Module):
    def __init__(self, dim: int, depth: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, dim)
        self.blocks = nn.ModuleList([DemoBlock(dim) for _ in range(depth)])
        self.output = nn.Linear(dim, VOCAB_SIZE)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x)
        return self.output(x)


def demo_training(dim: int, loader: DataLoader, steps: int = 8) -> tuple[int, float]:
    depth = max(1, dim // 32)
    model = DemoModel(dim, depth)
    model.to(dtype=t81_torch.trit)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_value = 0.0
    for step, batch in zip(range(steps), itertools.cycle(loader)):
        inp, tgt = batch
        optimizer.zero_grad()
        logits = model(inp)
        loss = nn.functional.cross_entropy(logits.view(-1, VOCAB_SIZE), tgt.view(-1))
        loss.backward()
        optimizer.step()
        loss_value = loss.item()
        if step % 2 == 0:
            logger.info("demo dim=%d step=%d loss=%.5f", dim, step, loss_value)
    total_params = sum(p.numel() for p in model.parameters())
    return total_params, loss_value


def plot_curve(params: Sequence[int], losses: Sequence[float], output: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.loglog(params, losses, marker="o", label="demo losses")
    plt.loglog(params, [losses[-1] * (p / params[-1]) ** (-0.35) for p in params], linestyle="--", label="Chinchilla fit")
    plt.xlabel("Parameters")
    plt.ylabel("Loss")
    plt.title("Exact ternary scaling law demo")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output)
    logger.info("demo plot saved to %s", output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a ternary scaling demo.")
    parser.add_argument("--plot", type=Path, default=Path("scaling_demo.png"))
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--scaling", action="store_true")
    args = parser.parse_args()

    loader = build_loader()
    dims = [32, 64, 96]
    records: list[tuple[int, float]] = []
    for dim in dims:
        params, loss = demo_training(dim, loader, steps=args.steps)
        records.append((params, loss))
    params, losses = zip(*records)
    plot_curve(params, losses, args.plot)

    if args.scaling:
        summary = scaling_laws_experiment({"model_dims": dims, "steps_per_config": args.steps, "plot_path": "scaling_laws.png"})
        logger.info("scaling experiment produced losses %s", summary["losses"])


if __name__ == "__main__":
    main()
