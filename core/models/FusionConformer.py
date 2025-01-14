from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

import einops
from einops import rearrange
from einops.layers.torch import Rearrange
from .conformer import FeedForward, ConformerConvModule, Scale, PreNorm


class ConditionalConformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads=8,
        ff_mult=4,
        conv_expansion_factor=2,
        conv_kernel_size=31,
        conv_causal_mode=True,
        attn_dropout=0.2,
        ff_dropout=0.2,
        conv_dropout=0.0,
    ):
        super().__init__()
        self.ff1 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        # self.attn = Attention(
        #     dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout
        # )
        self.attn_pre_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, batch_first=True, dropout=attn_dropout
        )
        self.conv = ConformerConvModule(
            dim=dim,
            causal=conv_causal_mode,
            expansion_factor=conv_expansion_factor,
            kernel_size=conv_kernel_size,
            dropout=conv_dropout,
        )
        self.ff2 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)

        # self.attn = PreNorm(dim, self.attn)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))

        self.post_norm = nn.LayerNorm(dim)

    def get_mask(self, x, attn_wlen: Optional[int] = None):
        """should be call outside
        mask the componet where the mask is True.

        x with shape (B,T,H)

        return: B,T,T
        """
        nT = x.shape[-2]

        mask = torch.ones(nT, nT, device=x.device, dtype=torch.bool).triu_(1)  # TxT
        if attn_wlen is not None:
            # assert attn_wlen >= 1
            mask_prev = torch.ones(nT, nT, device=x.device, dtype=torch.bool).tril_(-attn_wlen)
            mask = mask + mask_prev
        return mask.bool()

    def forward(
        self,
        x,
        c,
        mask: Optional[torch.Tensor] = None,
        need_weights=True,
    ):
        x = self.ff1(x) + x
        # x = self.attn(x, mask=mask) + x

        x = self.attn_pre_norm(x)
        x_mhsa, attn = self.attn(
            x, x, x, need_weights=need_weights, average_attn_weights=False, attn_mask=mask
        )
        x = x + x_mhsa

        x = self.conv(x) + x
        x = self.ff2(x) + x
        x = self.post_norm(x)
        return x, attn


class FusionConformer(nn.Module):
    def __init__(
        self,
        dim,
        heads=4,
        ff_mult=4,
        conv_expansion_factor=2,
        conv_kernel_size=31,
        attn_dropout=0.2,
        ff_dropout=0.2,
        conv_dropout=0.0,
    ):
        super().__init__()
