import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, n_embd, expand_factor, dropout, bias):
        super().__init__()
        self.n_embd = n_embd
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = self.n_embd * expand_factor
        self.linear1 = nn.Linear(self.n_embd, self.hidden_dim, bias=bias)
        self.non_linear = nn.GELU()
        self.linear2 = nn.Linear(self.hidden_dim, self.n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        ll1 = self.linear1(X)
        nll = self.non_linear(ll1)
        ll2 = self.linear2(nll)
        output = self.dropout(ll2)
        return output

if __name__ == "__main__":
    model = MLP(
        n_embd=384,
        expand_factor=4,
        dropout=0.1,
        bias=False
    )

    x = torch.randn(2, 10, 384)
    output = model(x)

    print(output.shape)