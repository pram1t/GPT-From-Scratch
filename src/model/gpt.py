# needs transformers.py

import torch
import torch.nn as nn
from embeddings import GPTEmbedding
from transformer import Transformer


class GPT(nn.Module):
  def __init__(self, block_size, vocab_size, n_embd, n_head, dropout, n_block, eps=1e-5, bias=False, expand_factor=4):
    super().__init__()
    self.n_block = n_block
    self.transformers = nn.ModuleList([Transformer(block_size, n_embd, n_head, dropout, eps, bias, expand_factor) for _ in range(n_block)])
    self.ln_f = nn.LayerNorm(n_embd) # this will be the final layer norm that takes in the input of the last of the transformer block
    # as the output of this layer norm will be the input to the language model input!
    self.LM = nn.Linear(n_embd, vocab_size, bias)
    self.embedding = GPTEmbedding(block_size, vocab_size, n_embd)

  def forward(self, x):
    # blocks_output = [out = block(out) for block in self.transformer]
    # 1. this out=Block(out) doesn't work, with a one line loop (list comprehension)
    # 2. no need to store this as an list where it captures list comprehension, we just need to return the final out
    embeddings_output = self.embedding(x)
    output = embeddings_output
    for block in self.transformers:
      output = block(output)

    ln_f_output = self.ln_f(output)
    lm_output = self.LM(ln_f_output)
    return lm_output


if __name__ == "__main__":
    import torch

    # Model configuration
    block_size = 1024
    vocab_size = 50257
    n_embd = 384
    n_head = 8
    dropout = 0.1
    n_block = 6

    # Initialize GPT
    model = GPT(
        block_size=block_size,
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        dropout=dropout,
        n_block=n_block
    )

    # Create dummy input embeddings: (B, T, C)
    # x = torch.randn(2, 10, n_embd)
    x = torch.randint(0, vocab_size, (2, 10)) # syntax: torch.randint(low, high, size)

    # Forward pass
    logits = model(x)

    # Print shapes
    print("Input shape:", x.shape)
    print("Logits shape:", logits.shape)

    # Verify expected output
    assert logits.shape == (2, 10, vocab_size)

    print("GPT forward pass successful!")