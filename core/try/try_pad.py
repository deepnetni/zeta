import torch
from torch.nn.utils.rnn import pad_sequence


a = torch.randn(10, 2)
b = torch.randn(10, 2)
c = torch.randn(8, 2)

d = pad_sequence([a, b, c], batch_first=True)
print(d.shape, d[2], c)
