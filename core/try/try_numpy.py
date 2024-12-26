#!/usr/bin/env python3
import numpy as np
import torch

a = torch.randn(4, 10).cpu().detach().numpy()
b = list(a)


for t in a:
    print(t.shape)
