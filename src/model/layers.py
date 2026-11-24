import torch
import torch.nn as nn

class LayerNorms(nn.Module):
  def __init__(self, n_embd, eps=1e-5):
    super().__init__()
    self.gamma = nn.Parameter(torch.ones(n_embd))
    self.beta = nn.Parameter(torch.zeros(n_embd))
    self.eps = eps

  def forward(self, idx):
    mean_value = torch.mean(idx, dim=-1, keepdim=True) # last dim = 1, like (2,10,1)
    variance_value = torch.var(idx, dim=-1, keepdim=True, correction=0) # [(i-mean_value)**2 for i in C]/len(C)
    output = self.gamma * ((idx - mean_value)/torch.sqrt(variance_value + self.eps)) + self.beta
    return output



if __name__=='__main__':
  n_embd = 384

  custom_ln = LayerNorms(n_embd)
  pytorch_ln = nn.LayerNorm(n_embd)

  x = torch.randn(2, 10, n_embd)

  custom_output = custom_ln(x)
  pytorch_output = pytorch_ln(x)

  print(custom_output.shape)
  print(pytorch_output.shape)

  print(torch.allclose(custom_output, pytorch_output, atol=1e-6))
  print(torch.max(torch.abs(custom_output - pytorch_output)))

  loss = custom_output.sum()
  loss.backward()

  print(custom_ln.gamma.grad is not None)
  print(custom_ln.beta.grad is not None)