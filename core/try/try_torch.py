#!/usr/bin/env python3
import torch

tensor = torch.tensor([[0, 1, 2], [3, 0, 4], [0, 5, 6]])

indices = torch.nonzero(tensor)

indices_tuple = torch.nonzero(tensor, as_tuple=True)

print("Indices (non-tuple):", indices)
print("Indices (tuple):", indices_tuple)

import torch

bands_range = torch.tensor([0, 10, 20, 30, 40])
all_freqs = torch.tensor([5, 6, 15, 25, 35])

bands_idx = [
    (i, ((all_freqs >= low) & (all_freqs < high)).nonzero(as_tuple=True)[0])
    for i, (low, high) in enumerate(zip(bands_range[:-1], bands_range[1:]))
]

print(bands_idx)


def buffered_arange(max, device="cpu"):
    if not hasattr(buffered_arange, "buf"):
        buffered_arange.buf = torch.LongTensor().to(device)
    if max > buffered_arange.buf.numel():
        buffered_arange.buf.resize_(max)
        torch.arange(max, out=buffered_arange.buf)
    return buffered_arange.buf[:max]


a = buffered_arange(10).unsqueeze(-1)
print("a", a)
a = a.expand(-1, 20)
print("a", a.flatten())

import torch
import torch.nn.functional as F

# 假设我们有一个模型的 logits 输出
logits = torch.tensor([[1.0, 2.0, 3.0]])

# 使用 Gumbel-Softmax 进行采样
tau = 0.5  # 温度
hard = True  # 是否返回硬的 one-hot 向量

# 进行采样
gumbel_sample = F.gumbel_softmax(logits, tau=tau, hard=hard)

print("Gumbel-Softmax 采样结果:", gumbel_sample)
