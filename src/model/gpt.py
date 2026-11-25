# needs transformers.py

import torch
import torch.nn as nn

from transformer import Transformer
from mlp import MLP

class GPT(nn.Module):
  def __init__(self, block_size, n_embd, n_head, dropout, n_block, eps=1e-5, bias=False, expand_factor=4):
    super().__init__()
    self.n_block = n_block
    self.transformers = nn.ModuleList([Transformer(block_size, n_embd, n_head, dropout, eps, bias, expand_factor) for _ in range(n_block)])

  def forward(self, x):
    output = x
    # blocks_output = [out = block(out) for block in self.transformer]
    # 1. this out=Block(out) doesn't work, with a one line loop (list comprihension)
    # 2. no need to store this as an list where it captures list comprihantion, we just need to return the final out
    for block in self.transformers:
      output = block(output)
    return output