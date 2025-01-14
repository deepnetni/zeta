import torch


a = torch.arange(0, 10) / 10
print(a[None].shape)
