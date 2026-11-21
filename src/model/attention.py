import torch
import torch.nn as nn
import math

class Head(nn.Module):
  def __init__(self, block_size, n_embd, head_size, dropout): # T, C, size of the output head (last dim of the output) here on a single head attention block we do not need n_head, so head size is
    # now here in a single head, head size can be n_embd / 1 which is completely fine but we chose a divisible number instead
    super().__init__()
    self.block_size = block_size
    self.n_embd = n_embd
    self.head_size = head_size
    self.dropout = dropout
    self.query = nn.Linear(n_embd, head_size, bias=False) # this can be nn.Linear(n_embd, n_embd) ie both in feature and out feature be same as n_embd as for 1 head head size == n_embd
    # For single-head attention, head_size can be any positive integer. It doesn't need to divide n_embd.
    self.key = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))
    # Attention dropout
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    b, t, c = x.shape
    q = self.query(x)
    k = self.key(x)
    v = self.value(x)

    print("query shape: ", q.shape)
    print("key shape: ", k.shape)
    print("value shape: ", v.shape)

    score = q @ k.transpose(-2, -1)/math.sqrt(self.head_size)
    masked_score = score.masked_fill(self.mask[:t, :t]==0, float('-inf'))
    softmax_score = torch.softmax(masked_score, dim=-1)# dim=-1 means apply the softmax on the last dim!
    softmax_score = self.dropout(softmax_score)
    output = softmax_score @ v

    print("output shape: ",output.shape)
    return output


class MultiHead(nn.Module):
  def __init__(self, block_size, n_embd, n_head, dropout ):
    super().__init__()
    self.block_size = block_size
    self.n_embd = n_embd
    self.n_head = n_head
    assert n_embd % n_head == 0 # a check so that we can be sure of that the n_head can divide the n_embd
    self.head_size = n_embd // n_head
    self.heads = nn.ModuleList([Head(block_size, n_embd, self.head_size, dropout) for _ in range (n_head)])
    self.proj = nn.Linear(n_embd, n_embd) # after concat we got back the last dim as same size as the n_embd
    # and want the Multi-Head Attention module to return a tensor with the same embedding dimension as its input, that why the shape is like this
    self.resid_dropout = nn.Dropout(dropout)


  def forward(self, x):

    multihead_output = [single_head(x) for single_head in self.heads]
    concated_output = torch.cat(multihead_output, dim=-1)
    output = self.proj(concated_output) # we pass the concanated output of all those attention head into a linear layer to get:
    # "To learn how to mix information across all attention heads into one unified representation while preserving the original embedding dimension"
    # Residual dropout
    output = self.resid_dropout(output) # we apply a final dorpout after the multi head
    # this is useless for this multihead class but will be used as an masked input in the transformer block!
    print(output.shape)
    return output



if __name__ == "__main__":

    # Step 1: Initialize model
    model = MultiHead(
        block_size=1024,
        n_embd=384,
        n_head=8,
        dropout=0.5
    )

    # Step 2: Create input
    x = torch.rand(2, 10, 384)

    output = model(x)

    print(output.shape)

    # Step 3: Training mode
    # model.train()

    # out1 = model(x)
    # out2 = model(x)

    # print("Training outputs identical:",
    #       torch.allclose(out1, out2))

    # Step 4: Evaluation mode
    # model.eval()

    # out3 = model(x)
    # out4 = model(x)

    # print("Evaluation outputs identical:",
    #       torch.allclose(out3, out4))
