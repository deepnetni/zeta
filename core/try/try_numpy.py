#!/usr/bin/env python3
import numpy as np


i = np.arange(4)
a = np.arange(24).reshape(1, 2, 3, 4)
print("a:", a)
idx = np.where((i >= 2) & (i < 3))[0]
print(idx.shape, idx)
b = a[..., idx]
print(b.shape, b)
b = a[..., (1)]
print(b.shape, b)

print(b.size)


print("@@@")

a = np.random.randn(2, 5, 3)  # B,T,C
print("a:", a)
b = np.zeros((2, 3))

for i in range(5):
    b = a[:, i, :]
    print(b.shape)
