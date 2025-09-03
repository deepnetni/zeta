import torch
import torch.nn as nn
import torch.nn.functional as F

import einops
from einops import rearrange
from einops.layers.torch import Rearrange


a = torch.randn(1, 2, 3, 4)
r, i = a.chunk(2, dim=1)
print(a)
print(r.shape)

i = torch.complex(r, i)
print(i, i.shape, i.dtype)
