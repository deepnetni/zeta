import torch
import torch.nn as nn
import torch.nn.functional as F

import einops
from einops import rearrange
from einops.layers.torch import Rearrange


inp = torch.randn(2, 10, 2, 4)
l = nn.BatchNorm2d(10)
out = l(inp)
print(out.shape)
for n, v in l.named_parameters():
    print(n, v.shape)
for n, v in l.named_buffers():
    print(n, v.shape)


print("##")

l = nn.InstanceNorm2d(10, affine=True, track_running_stats=True)
out = l(inp)
print(out.shape)
for n, v in l.named_parameters():
    print(n, v.shape)
for n, v in l.named_buffers():
    print(n, v.shape)
