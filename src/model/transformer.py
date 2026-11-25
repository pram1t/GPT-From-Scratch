# needs attention.py and layers.py

import torch
import torch.nn as nn
from layers import LayerNorms
from attention import MultiHead
from mlp import MLP

class Transformer(nn.Module):
    def __init__(self, block_size, n_embd, n_head, dropout, eps=1e-5, bias=False, expand_factor=4):
        super().__init__()
        # we do these below to store these as attribute of the class but it's not required
        # as we simply use these to construct child modules.
        # so they are unnecessary
        # self.n_embd = n_embd
        # self.n_head = n_head
        # self.dropout = dropout
        # self.eps = eps
        # self.block_size = block_size
        self.ln1 = LayerNorms(n_embd, eps)
        self.attention = MultiHead(block_size, n_embd, n_head, dropout)
        self.ln2 = LayerNorms(n_embd, eps)
        self.mlp = MLP(n_embd, expand_factor, dropout, bias)


    def forward(self, X):
        ln1_output = self.ln1(X)
        attention_output = self.attention(ln1_output)
        intr_input = X + attention_output
        ln2_output = self.ln2(intr_input)
        mlp_output = self.mlp(ln2_output)
        output = intr_input + mlp_output
        return output



if __name__ == "__main__":
    model = Transformer(
        block_size=1024,
        n_embd=384,
        n_head=8,
        dropout=0.1
    )

    x = torch.randn(2, 10, 384)
    output = model(x)

    print(output.shape)