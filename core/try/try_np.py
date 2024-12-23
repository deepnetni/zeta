import numpy as np
import torch


# a = np.random.randint(0, 1)
# print(a)
# a = np.random.choice(np.arange(3), p=[0.1, 0.6, 0.3])
# print(a)
print(np.random.uniform(-1, -3))


a = torch.randn(4, 5)
b = torch.tensor([3, 5, 5, 4])

c = torch.min(torch.ones_like(b) * a.shape[-1], b)
print(c)
