import numpy as np
import math
import torch
import torch.nn as nn


def positional_encoding(sequence_length, embedding_dim):
    position = torch.arange(0, sequence_length).float()
    div_term = torch.exp(
        torch.arange(0, embedding_dim, 2).float() * -(math.log(10000.0) / embedding_dim)
    )
    pos_encoding = torch.zeros(sequence_length, embedding_dim)
    pos_encoding[:, 0::2] = torch.sin(position * div_term)
    pos_encoding[:, 1::2] = torch.cos(position * div_term)
    pos_encoding = pos_encoding.unsqueeze(0)
    return pos_encoding


# a = positional_encoding(6, 6)
# print(a)


class SinPositionEncoding(nn.Module):
    def __init__(self, max_sequence_length, d_model, base=10000):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model
        self.base = base

    def forward(self):
        pe = torch.zeros(
            self.max_sequence_length, self.d_model, dtype=torch.float
        )  # size(max_sequence_length, d_model)
        exp_1 = torch.arange(self.d_model // 2, dtype=torch.float)
        exp_value = exp_1 / (self.d_model / 2)

        alpha = 1 / (self.base**exp_value)  # size(dmodel/2)
        out = (
            torch.arange(self.max_sequence_length, dtype=torch.float)[:, None] @ alpha[None, :]
        )  # size(max_sequence_length, d_model/2)
        embedding_sin = torch.sin(out)
        embedding_cos = torch.cos(out)

        pe[:, 0::2] = embedding_sin
        pe[:, 1::2] = embedding_cos
        return pe


a = SinPositionEncoding(d_model=10, max_sequence_length=6, base=10000).forward()
print(a)
