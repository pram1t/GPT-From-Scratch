import torch
import torch.nn as nn
import math

class LayerNorms(nn.Module):
  def __init__(self, gamma, delta):
    super().__init__()
    self.gamma = gamma
    self.delta = delta

  def forward(self, idx):
    B, T, C = idx.shape
    mean_value = torch.mean(idx)
    varience_value = torch.var(mean_value) #[(i-mean_valu)**2 efor i in C]/len(C)
    output = self.gamma * ((idx - mean_value)/math.sqrt(varience_value**2)) + self.delta
    return output