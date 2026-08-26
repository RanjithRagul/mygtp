import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
import math

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
    self.bias = config.bias
    self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=self.bias)
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
    # nn.Dropout(value, inplace=True/False) -> inplace is an optional and this inplace is to create a new tensor or inplace
    self.attn_dropout = nn.Dropout(self.dropout)
    self.resid_dropout = nn.Dropout(self.dropout)
    #------------------------------------------------------------------------
    '''
    check our PyTorch >= 2.0, then scaled_dot_product_attention will be there
    scaled_dot_product_attention = softmax((Q.K_T)/sqrt(channel)).V
    '''
    self.flash = hasattr(F, 'scaled_dot_product_attention')
    if not self.flash: # older version
      print("WARNING: using slow attention. Flash attention requires PyTorch >= 2.0")
      '''
      1. this created tensor will be in the model, but don't train it
      2. this is a templace 
      [1,0,0]
      [1,1,0]
      [1,1,1]
      3. [1] * block_size -> reshape it to -> (1, 1, block_size, block_size)
      4. and to the model buffer
      '''
      self.register_buffer(
            "bias",
            torch.tril(
              torch.ones(config.block_size, config.block_size)
            ).view(1, 1, config.block_size, config.block_size)
      )
		
	def forward(self, x:Tensor)->Tensor:
		B, T, C = x.size()
		q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
		q = q.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
		k = k.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
		v = v.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
	
		if self.flash:
			y = F.scaled_dot_product_attention(
				q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_casual=True
			)
		else:
			att = q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))
			att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('inf'))
			att = F.softmax(att, dim=-1)
			att = self.attn_dropout(att)
			y = att @ v
		y = y.transpose(1, 2).contiguous().view(B, T, C)
		y = self.resid_dropout(self.c_proj(y))
		return y
class MLP(nn.module):
	def __init__(self, config):
		super().__init__()
		self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd, bias=config.bias) # in_feature = n, out_feature = 4*n
		self.gelu = nn.GELU()
		self.c_proj = nn.Linear(4*config.n_embd, config.n_embd, bias=config.bias) # in_feature = 4*n, out_feature = n
		self.dropout = nn.Dropout(config.dropout)
		'''
		GELU:
		simplifed aprox: (x/2) * (1 + erf(x/sqrt(2))) # x is a tensor
		tensor version : 0.5 * x * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0)))) # not tensor([2])
		erf (normal distribution)-> -1 to 1
		'''
		
	def forward(self, x:Tensor)->Tensor:
		# linear -> GELU -> Linear -> Dropout
		x = self.c_fc(x)
		x = self.gelu(x)
		x = self.c_proj(x)
		x = self.dropout(x)
		return x
