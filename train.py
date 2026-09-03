import os
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_groups, destory_process_group

from model import GPTConfig, GPT
#--------------------------------------------------------------------------
out_dir = 'out'
gradient_Accumulation = 5 * 8
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
