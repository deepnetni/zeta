import numpy as np
import torch

n_fft = 11
GD_matrix = (
    torch.ones(n_fft // 2 + 1, n_fft // 2 + 1).triu(1)
    - torch.triu(torch.ones(n_fft // 2 + 1, n_fft // 2 + 1), diagonal=2)
    - torch.eye(n_fft // 2 + 1)
)
print(GD_matrix)


a = torch.randn(1, 2, 3, 4)
b, _ = a.chunk(2, dim=1)
c, _ = a.chunk(2, dim=3)
print(b.shape, c.shape)
