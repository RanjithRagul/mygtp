import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
  def __init__(self, ndim: integer, bias: boolean):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(ndim))
    self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    
  def forward(self, input: Tensor) -> Tensor:
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
   2. input @ weight + bias
   return the input
   1e-5 = 0.00001
   """
    return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    
