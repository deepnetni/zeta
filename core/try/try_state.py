import torch
import torch.nn as nn


inp = torch.randn(1, 5, 10)
l = nn.LSTM(
    10,
    3,
    batch_first=True,
    # bidirectional=True,
)

# out, (h, c) = l(inp)
# print(h.shape, c.shape, out.shape)
# print(out[:, 4, :])
# print(h)

stat = None

out_, (h, c) = l(inp)

out = []
for i in range(5):
    d = inp[:, (i,), :]
    do, stat = l(d, stat)
    out.append(do)

out = torch.concat(out, dim=1)
print(out.shape, out_.shape)
print(out, out_)
