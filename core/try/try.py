import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


a = np.array([1, 2, 3, 4, 5]) + 3

b = np.argwhere(a > 6).squeeze()
print(b)


b = np.flatnonzero(a > 6)
print(b)
