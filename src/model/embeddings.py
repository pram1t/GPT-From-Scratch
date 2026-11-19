# writing text encoding and positional encoding

import torch
import torch.nn as nn

class GPTEmbedding(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd):
        super().__init__()
        # vocab_size = self.vocab_size
        # n_embd = self.n_embd
        # block_size = self.block_size
        self.txt_embd = nn.Embedding(vocab_size, n_embd)
        self.pos_embd = nn.Embedding(block_size, n_embd)

    def forward(self, idx):
        B,T = idx.size()
        text_embedding = self.txt_embd(idx)
        pos_embedding = self.pos_embd(torch.arange(T, device=idx.device))
        block = text_embedding + pos_embedding
        return block



if __name__ == "__main__":
    model = GPTEmbedding(50257, 1024, 384) # this is the initialization
    idx = torch.randint(0, 50257, (2, 10))
    output = model(idx) # this is the input of that initialized object
    print(output)
    print(output.size())