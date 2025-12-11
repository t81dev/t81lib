import torch
from torch import nn
import t81.torch as t81_torch

class TinyLlamaBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Linear(1024, 4096)
    def forward(self, x):
        return self.w(x)

model = TinyLlamaBlock()
activations = torch.randn(4, 1024)
model.to(dtype=t81_torch.trit)
print("Converted weights cache:", list(getattr(model, "_t81_ternary_cache", {}).keys()))
with torch.no_grad():
    output = model(activations)
print("Output shape:", output.shape)
print("Sample logits:", output[0, :4])
assert output.shape == (4, 4096)
