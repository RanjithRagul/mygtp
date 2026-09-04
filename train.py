import os
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_groups, destory_process_group

from model import GPTConfig, GPT
#------------------------------ dir --------------------------------------------
out_dir = 'out'
dataset = 'openwebtext'
#------------------------------ config-1 ---------------------------------------
batch_size = 12
block_size = 1024
n_layer = 12
n_head  = 12
n_embd  = 768
dropout = 0.0
bias    = False
#------------------------------ adamW optimizer ---------------------------------------
learning_rate = 6e-4 # 0.6000
max_iters = 600000 
weight_decay = 1e-1 # 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
#--------------------------------------------------------------------------
device = 'cuda'
gradient_Accumulation = 5 * 8
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_b16_supported() else 'float16'
#--------------------------------------------------------------------------
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp: # GPU
  init_process_groups(backend=backend)
  ddp_rank       = int(os.environ['RANK'])
  ddp_local_rank = int(os.environ['LOCAL_RANK'])
  ddp_world_size = int(os.environ['WORLD_SIZE'])
  
  device = f'cuda:{dpp_local_rank}'
  torch.cuda.set_device('device')
  master_process = ddp_rank ==  0
  seed_offset = ddp_rank

  assert = gradient_Accumulation % ddp_world_size == 0
  gradient_accumulation_stes //= ddp_world_size
else: # CPU
  master_process = True
  seed_offset = 0
  ddp_world_size = 1

token_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f'toeks per iteration will be: {token_per_iter:,}')

if master_process:
  os.makedir(out_dir, exist_ok=True)

torch.backend.cuda.matmul.allow_tf32 = True
torch.backend.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype     = {
                'float32'  : torch.float32,
                'float16'  : torch.float16,
                'bfloat16' : torch.bfloat16,
              }[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

data_dir = os.path.join('data', dataset)
def get_batch(split):
  if split == 'train':
    data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
  else:
    data = np.memmap(os.path.join(data_dir, 'val.bin'  ), dtype=np.uint16, mode='r')
  ix = torch.ranint(len(data) - block_size, (batch_size,))
  x = torch.stack([torch.from_numpy((data[i  :i+block_size ]).astype(np.int64))  for i in ix])
  y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in id])

  if device_type == 'cuda':
    x = x.pin_memory().to(device, non_blocking=True)
    y = y.pin_memory().to(device, non_blocking=True)
  else:
    x = x.to(device)
    y = y.to(device)
  return x, y

# init these up here, can override if init_from='resume' (i.e, from a checkpoint)
iter_num = 0
bext_val_loss = 1e9 # 1000,000,000 = 1Billion

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
  with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
  meta_vocab_size = meta['vocab_size']
  print(f'found vocab_size = {meta_vocab_size} (inside {meta_path})')
