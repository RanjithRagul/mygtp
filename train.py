import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor

class LayerNorm(nn.Module):
  def __init__(self, ndim:int, bias:bool = True):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(ndim))
    self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    
  def forward(self, input: Tensor) -> Tensor:
    return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    """
    input -> tensor, weight shape, weight, bias, epsilon to tackel zero division error
    1. normalize the input tensor: 
    mean = sum(input)/len(input)
    #--------------------------------
    variance = 0
    for n in input:
      variance += (n - mean)**2
    variance /= len(input)
    #--------------------------------
    normalized = (x - mean) / sqrt(variance + tiny number to tackle divisible by zero error)
    NOTE: mean, variance = integer, tiny = small decimal value, x = Tensor
    #--------------------------------
    2. normalised @ weight + bias
    return the input
    1e-5 = 0.00001
    """
class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()
    assert config.n_embd % config.n_head == 0
    self.n_head = config.n_head
    self.n_embd = config.n_embd
    self.dropout = config.dropout
    self.bias = self.bias
    self.c_attn = nn.Linear(self.n_embd, self.n_embd * 3, bias=self.bias)
    self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=self.bias)
    '''
    nn.linear
    1. input @ w.T + bias
    2. weight, bias will be created by itself using given shape
    3. nn.Linear(in_features, out_features, bias=True)
    4. weight = torch.randn(out_features, in_features)
    5. bias = torch.randn(out_features)
    '''
    # remember dropout is always between 0 - 1
    # nn.Dropout(value, inplace=True/False) -> this in place is to create a new tensor or inplace
    self.attn_dropout = nn.Dropout(self.dropout)
    self.resid_dropout = nn.Dropout(self.dropout)
