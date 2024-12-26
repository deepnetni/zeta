import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import GRU, Dropout, LayerNorm, Linear, MultiheadAttention


class FFN(nn.Module):
    def __init__(self, input_feature_size, bidirectional=True, dropout=0):
        super(FFN, self).__init__()
        self.gru = GRU(input_feature_size, input_feature_size * 2, 1, bidirectional=bidirectional)
        if bidirectional:
            self.linear = Linear(input_feature_size * 2 * 2, input_feature_size)
        else:
            self.linear = Linear(input_feature_size * 2, input_feature_size)
        self.dropout = Dropout(dropout)

    def forward(self, x):
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = self.linear(x)

        return x


class TransformerBlock(nn.Module):
    """
    input: B,T,d_jmodel
    """

    def __init__(self, input_feature_size, n_heads, bidirectional=True, dropout=0):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(input_feature_size)
        self.attention = MultiheadAttention(input_feature_size, n_heads, dropout=dropout)
        self.dropout1 = Dropout(dropout)

        self.norm2 = LayerNorm(input_feature_size)
        self.ffn = FFN(input_feature_size, bidirectional=bidirectional)
        self.dropout2 = Dropout(dropout)

        self.norm3 = LayerNorm(input_feature_size)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        xt = self.norm1(x)
        xt, _ = self.attention(xt, xt, xt, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = x + self.dropout1(xt)

        xt = self.norm2(x)
        xt = self.ffn(xt)
        x = x + self.dropout2(xt)

        x = self.norm3(x)

        return x


def main():
    x = torch.randn(4, 64, 401, 201)
    b, c, t, f = x.size()
    x = x.permute(0, 3, 2, 1).contiguous().view(b, f * t, c)
    transformer = TransformerBlock(input_feature_size=64, n_heads=4)
    x = transformer(x)
    x = x.view(b, f, t, c).permute(0, 3, 2, 1)
    print(x.size())


if __name__ == "__main__":
    main()
