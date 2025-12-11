"""Lightweight preview of ternary sparse × dense multiplication."""

from typing import Tuple

import torch
import t81lib

def build_random_sparse(rows: int, cols: int, density: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    nnz = max(1, int(rows * cols * density))
    row_indices = torch.randint(0, rows, (nnz,), dtype=torch.int32)
    col_indices = torch.randint(0, cols, (nnz,), dtype=torch.int32)
    values = torch.randint(-3, 4, (nnz,), dtype=torch.int16)
    values[values == 0] = 1
    return values, row_indices, col_indices

def main():
    rows = 256
    cols = 256
    features = 128
    density = 0.03
    values, row_indices, col_indices = build_random_sparse(rows, cols, density)

    B = torch.randn(cols, features, dtype=torch.float32)
    C = torch.zeros(rows, features, dtype=torch.float32)

    t81lib.spmm_simple(
        values.tolist(),
        row_indices.tolist(),
        col_indices.tolist(),
        rows,
        cols,
        B.numpy().ravel(),
        C.numpy().ravel(),
        features,
    )

    dense = torch.zeros(rows, cols, dtype=torch.float32)
    for r, c, v in zip(row_indices.tolist(), col_indices.tolist(), values.tolist()):
        dense[r, c] += float(v)
    reference = dense @ B

    if not torch.allclose(C, reference, rtol=1e-3, atol=1e-3):
        raise RuntimeError("Ternary sparse result deviated from dense reference")

    print("Ternary sparse matmul works with preview sparse format")
    print(f"Non-zeros: {values.numel():,} / {rows * cols:,} ({density * 100:.2f}% density)")
    print(f"Result norm: {C.norm():.4f}")

if __name__ == "__main__":
    main()
