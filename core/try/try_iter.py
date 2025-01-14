from itertools import product
import torch

p = [range(10)] * 2
print(p)
inds = list(product(*p))  # num_vars, num_vars
inds = torch.tensor(inds)
print(inds.shape, inds[0])


idx = torch.randint(0, 10, size=(3,))
print(idx.shape, idx)
