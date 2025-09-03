import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
print(sys.path)

import torch
import torch.nn as nn
import torch.nn.functional as F

import einops
from einops import rearrange
from einops.layers.torch import Rearrange

from AFCModel import CRN
from utils.audiolib_pt import AcousticFeedbackSim


# net = nn.GRU(32, 32, batch_first=True)
net = CRN(64)

inp = torch.randn(10, 10, 64)
ref = torch.randn(10, 64)
tgt = torch.randn(10, 640)

FB = AcousticFeedbackSim(torch.randn(512), 64)

out = []
stat = None
fb = torch.randn(10, 64)
for i in range(10):
    x_ = inp[:, i, :]
    x = x_ + fb
    x, stat = net(x, ref, stat)

    ref = x.detach()
    fb = FB(x.detach())

    out.append(x)

out = torch.concat(out, dim=1)
print(out.shape)

loss = F.mse_loss(tgt, out)
print(loss)
loss.backward()
