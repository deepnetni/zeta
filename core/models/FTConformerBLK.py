#!/usr/bin/env python3
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

import einops
from einops import rearrange
from einops.layers.torch import Rearrange
from .conformer import ConformerBlock, Swish, Attention, DiTConformerBlock


class FTConformer(nn.Module):
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
        self.time_conformer = ConformerBlock(
            dim=dim,
            heads=heads,
            ff_mult=ff_mult,
            conv_expansion_factor=conv_expansion_factor,
            conv_kernel_size=conv_kernel_size,
            conv_causal_mode=True,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            conv_dropout=conv_dropout,
        )
        self.freq_conformer = ConformerBlock(
            dim=dim,
            heads=heads,
            ff_mult=ff_mult,
            conv_expansion_factor=conv_expansion_factor,
            conv_kernel_size=conv_kernel_size,
            conv_causal_mode=False,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            conv_dropout=conv_dropout,
        )

    def forward(self, x, causal=False, wlen=None):
        """
        x: b,c,t,f
        """
        nB = x.size(0)
        x_f = einops.rearrange(x, "b c t f->(b t) f c")
        x_, attn_f = self.freq_conformer(x_f)
        x_f = x_ + x_f

        if causal:
            mask = self.time_conformer.get_mask(x, wlen)
        else:
            mask = None

        x_t = einops.rearrange(x_f, "(b t) f c->(b f) t c", b=nB)
        x_, attn_t = self.time_conformer(x_t, mask=mask)
        x_t = x_t + x_
        x_ = einops.rearrange(x_t, "(b f) t c->b c t f", b=nB)

        return x_, (attn_f, attn_t)


# class ConditionalFTConformer_(nn.Module):
#     def __init__(
#         self,
#         dim,
#         heads=4,
#         ff_mult=4,
#         conv_expansion_factor=2,
#         conv_kernel_size=31,
#         attn_dropout=0.2,
#         ff_dropout=0.2,
#         conv_dropout=0.0,
#     ):
#         super().__init__()

#         self.pre_trans_1 = Attention(dim=dim, heads=heads, dropout=attn_dropout, max_pos_emb=32)
#         self.pre_trans_2 = Attention(dim=dim, heads=heads, dropout=attn_dropout, max_pos_emb=32)

#         self.time_conformer = ConformerBlock(
#             dim=dim,
#             heads=heads,
#             ff_mult=ff_mult,
#             conv_expansion_factor=conv_expansion_factor,
#             conv_kernel_size=conv_kernel_size,
#             conv_causal_mode=True,
#             attn_dropout=attn_dropout,
#             ff_dropout=ff_dropout,
#             conv_dropout=conv_dropout,
#         )
#         self.freq_conformer = ConformerBlock(
#             dim=dim,
#             heads=heads,
#             ff_mult=ff_mult,
#             conv_expansion_factor=conv_expansion_factor,
#             conv_kernel_size=conv_kernel_size,
#             conv_causal_mode=False,
#             attn_dropout=attn_dropout,
#             ff_dropout=ff_dropout,
#             conv_dropout=conv_dropout,
#         )
#         self.adaLN_modulation_f = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
#         self.norm1_f = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
#         self.norm2_f = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
#         self.mlp_f = nn.Sequential(
#             nn.Linear(dim, dim * ff_mult),
#             nn.GELU(approximate="tanh"),
#             nn.Linear(dim * ff_mult, dim),
#         )

#         self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
#         self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
#         self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, dim * ff_mult),
#             nn.GELU(approximate="tanh"),
#             nn.Linear(dim * ff_mult, dim),
#         )

#     @staticmethod
#     def modulate_unsqueeze(x, shift, scale):
#         """

#         :param x: b,t,c
#         :param shift: b,c
#         :param scale: b,c
#         :returns:

#         """
#         return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

#     @staticmethod
#     def modulate(x, shift, scale):
#         return x * (1 + scale) + shift

#     def forward(self, x, c, causal=False, wlen=None):
#         """
#         x: b,c,t,f
#         c: b,f,c
#         """

#         nB = x.size(0)
#         nT = x.size(-2)
#         ##################
#         # Freq Conformer #
#         ##################

#         if c.size(0) != nB * nT:
#             c_ = c.unsqueeze(1).repeat(1, nT, 1, 1).view(-1, c.size(-2), c.size(-1))  # btfc->(bt)fc
#         else:
#             c_ = c

#         x_ = rearrange(x, "b c t f-> (b t) f c")
#         f1 = self.pre_trans_1(c_, x_ + c_)  # BT,F,C
#         f2 = self.pre_trans_2(x_, x_ + c_)  # BT,F,C

#         c = (f1 + f2).tanh()

#         # conditions
#         (
#             shift_msa_f,
#             scale_msa_f,
#             gate_msa_f,
#             shift_mlp_f,
#             scale_mlp_f,
#             gate_mlp_f,
#         ) = self.adaLN_modulation_f(c).chunk(6, dim=-1)

#         x_f = einops.rearrange(x, "b c t f->(b t) f c")

#         x_ = self.modulate(self.norm1_f(x_f), shift_msa_f, scale_msa_f)
#         x_, attn_f = self.freq_conformer(x_)
#         x_f = x_ * gate_msa_f + x_f
#         x_ = gate_mlp_f * self.mlp_f(self.modulate(self.norm2_f(x_f), shift_mlp_f, scale_mlp_f))
#         x_f = x_f + x_

#         ####################
#         # # Time Conformer #
#         ####################
#         if causal:
#             mask = self.time_conformer.get_mask(x, wlen)
#         else:
#             mask = None

#         c_ = einops.rearrange(c, "(b t) f c->(b f) t c", b=nB)
#         shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
#             c_
#         ).chunk(6, dim=-1)

#         x_t = einops.rearrange(x_f, "(b t) f c->(b f) t c", b=nB)

#         x_ = self.modulate(self.norm1(x_t), shift_msa, scale_msa)
#         x_, attn_t = self.time_conformer(x_, mask=mask)
#         x_t = x_ * gate_msa + x_t
#         x_ = gate_mlp * self.mlp(self.modulate(self.norm2(x_t), shift_mlp, scale_mlp))
#         x_t = x_t + x_
#         x_ = einops.rearrange(x_t, "(b f) t c->b c t f", b=nB)

#         return x_, c, (attn_f, attn_t)


class ConditionalFTConformerIter(nn.Module):
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

        self.time_conformer = ConformerBlock(
            dim=dim,
            heads=heads,
            ff_mult=ff_mult,
            conv_expansion_factor=conv_expansion_factor,
            conv_kernel_size=conv_kernel_size,
            conv_causal_mode=True,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            conv_dropout=conv_dropout,
        )
        self.freq_conformer = ConformerBlock(
            dim=dim,
            heads=heads,
            ff_mult=ff_mult,
            conv_expansion_factor=conv_expansion_factor,
            conv_kernel_size=conv_kernel_size,
            conv_causal_mode=False,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            conv_dropout=conv_dropout,
        )
        self.adaLN_modulation_f = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        self.norm1_f = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2_f = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp_f = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * ff_mult, dim),
        )

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * ff_mult, dim),
        )
        self.mlp_cond = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * ff_mult, dim),
        )

    @staticmethod
    def modulate(x, shift, scale):
        """

        :param x: b,t,c
        :param shift: b,c
        :param scale: b,c
        :returns:

        """
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    @staticmethod
    def modulate_f(x, shift, scale):
        return x * (1 + scale) + shift

    def forward(self, x, c, causal=False, wlen=None):
        """
        x: b,c,t,f
        c: b,f,c
        """

        nB = x.size(0)
        nT = x.size(-2)
        ##################
        # Freq Conformer #
        ##################

        # conditions, b,f,c
        (
            shift_msa_f,
            scale_msa_f,
            gate_msa_f,
            shift_mlp_f,
            scale_mlp_f,
            gate_mlp_f,
        ) = self.adaLN_modulation_f(c).chunk(6, dim=-1)

        shift_msa_f = shift_msa_f.repeat_interleave(nT, dim=0)  # bt,f,c
        scale_msa_f = scale_msa_f.repeat_interleave(nT, dim=0)
        gate_msa_f = gate_msa_f.repeat_interleave(nT, dim=0)
        shift_mlp_f = shift_mlp_f.repeat_interleave(nT, dim=0)
        scale_mlp_f = scale_mlp_f.repeat_interleave(nT, dim=0)
        gate_mlp_f = gate_mlp_f.repeat_interleave(nT, dim=0)

        x_f = einops.rearrange(x, "b c t f->(b t) f c")

        x_ = self.modulate_f(self.norm1_f(x_f), shift_msa_f, scale_msa_f)
        x_, attn_f = self.freq_conformer(x_)
        x_f = x_ * gate_msa_f + x_f
        x_ = gate_mlp_f * self.mlp_f(self.modulate_f(self.norm2_f(x_f), shift_mlp_f, scale_mlp_f))
        x_f = x_f + x_

        ####################
        # # Time Conformer #
        ####################
        if causal:
            mask = self.time_conformer.get_mask(x, wlen)
        else:
            mask = None

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            c.view(-1, c.size(-1)).contiguous()
        ).chunk(6, dim=1)

        x_t = einops.rearrange(x_f, "(b t) f c->(b f) t c", b=nB)

        # bf,c->bf,1,c x bf,t,c
        x_ = self.modulate(self.norm1(x_t), shift_msa, scale_msa)
        x_, attn_t = self.time_conformer(x_, mask=mask)
        x_t = x_ * gate_msa.unsqueeze(1) + x_t
        x_ = gate_mlp.unsqueeze(1) * self.mlp(self.modulate(self.norm2(x_t), shift_mlp, scale_mlp))
        x_t = x_t + x_
        x_ = einops.rearrange(x_t, "(b f) t c->b c t f", b=nB)

        c_ = self.mlp_cond(c)
        return x_, c_, (attn_f, attn_t)


class ConditionalFTConformer(nn.Module):
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

        self.time_conformer = ConformerBlock(
            dim=dim,
            heads=heads,
            ff_mult=ff_mult,
            conv_expansion_factor=conv_expansion_factor,
            conv_kernel_size=conv_kernel_size,
            conv_causal_mode=True,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            conv_dropout=conv_dropout,
        )
        self.freq_conformer = ConformerBlock(
            dim=dim,
            heads=heads,
            ff_mult=ff_mult,
            conv_expansion_factor=conv_expansion_factor,
            conv_kernel_size=conv_kernel_size,
            conv_causal_mode=False,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            conv_dropout=conv_dropout,
        )
        self.adaLN_modulation_f = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        self.norm1_f = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2_f = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp_f = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * ff_mult, dim),
        )

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * ff_mult, dim),
        )

    @staticmethod
    def modulate(x, shift, scale):
        """

        :param x: b,t,c
        :param shift: b,c
        :param scale: b,c
        :returns:

        """
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    @staticmethod
    def modulate_f(x, shift, scale):
        return x * (1 + scale) + shift

    def forward(self, x, c, causal=False, wlen=None):
        """
        x: b,c,t,f
        c: b,f,c
        """

        nB = x.size(0)
        nT = x.size(-2)
        ##################
        # Freq Conformer #
        ##################

        # conditions, b,f,c
        (
            shift_msa_f,
            scale_msa_f,
            gate_msa_f,
            shift_mlp_f,
            scale_mlp_f,
            gate_mlp_f,
        ) = self.adaLN_modulation_f(c).chunk(6, dim=-1)

        shift_msa_f = shift_msa_f.repeat_interleave(nT, dim=0)  # bt,f,c
        scale_msa_f = scale_msa_f.repeat_interleave(nT, dim=0)
        gate_msa_f = gate_msa_f.repeat_interleave(nT, dim=0)
        shift_mlp_f = shift_mlp_f.repeat_interleave(nT, dim=0)
        scale_mlp_f = scale_mlp_f.repeat_interleave(nT, dim=0)
        gate_mlp_f = gate_mlp_f.repeat_interleave(nT, dim=0)

        x_f = einops.rearrange(x, "b c t f->(b t) f c")

        x_ = self.modulate_f(self.norm1_f(x_f), shift_msa_f, scale_msa_f)
        x_, attn_f = self.freq_conformer(x_)
        x_f = x_ * gate_msa_f + x_f
        x_ = gate_mlp_f * self.mlp_f(self.modulate_f(self.norm2_f(x_f), shift_mlp_f, scale_mlp_f))
        x_f = x_f + x_

        ####################
        # # Time Conformer #
        ####################
        if causal:
            mask = self.time_conformer.get_mask(x, wlen)
        else:
            mask = None

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            c.view(-1, c.size(-1)).contiguous()
        ).chunk(6, dim=1)

        x_t = einops.rearrange(x_f, "(b t) f c->(b f) t c", b=nB)

        # bf,c->bf,1,c x bf,t,c
        x_ = self.modulate(self.norm1(x_t), shift_msa, scale_msa)
        x_, attn_t = self.time_conformer(x_, mask=mask)
        x_t = x_ * gate_msa.unsqueeze(1) + x_t
        x_ = gate_mlp.unsqueeze(1) * self.mlp(self.modulate(self.norm2(x_t), shift_mlp, scale_mlp))
        x_t = x_t + x_
        x_ = einops.rearrange(x_t, "(b f) t c->b c t f", b=nB)

        return x_, (attn_f, attn_t)


class FTDiTConformer(nn.Module):
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

        self.time_conformer = DiTConformerBlock(
            dim=dim,
            heads=heads,
            ff_mult=ff_mult,
            conv_expansion_factor=conv_expansion_factor,
            conv_kernel_size=conv_kernel_size,
            conv_causal_mode=True,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            conv_dropout=conv_dropout,
        )
        self.freq_conformer = DiTConformerBlock(
            dim=dim,
            heads=heads,
            ff_mult=ff_mult,
            conv_expansion_factor=conv_expansion_factor,
            conv_kernel_size=conv_kernel_size,
            conv_causal_mode=False,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            conv_dropout=conv_dropout,
        )

    def forward(self, x, c, causal=False, wlen=None):
        """
        x: b,c,t,f
        c: b,f,c
        """

        nB = x.size(0)
        nT = x.size(-2)
        ##################
        # Freq Conformer #
        ##################

        # conditions, b,f,c

        x_f = einops.rearrange(x, "b c t f->(b t) f c")
        c_ = c.repeat_interleave(nT, dim=0)

        x_, attn_f = self.freq_conformer(x_f, c_)
        x_f = x_f + x_

        ####################
        # # Time Conformer #
        ####################
        if causal:
            mask = self.time_conformer.get_mask(x, wlen)
        else:
            mask = None

        x_t = einops.rearrange(x_f, "(b t) f c->(b f) t c", b=nB)
        c_ = einops.rearrange(c, "b f c->(b f) () c")

        # bf,c->bf,1,c x bf,t,c
        x_, attn_t = self.time_conformer(x_t, c_, mask=mask)
        x_t = x_t + x_
        x_ = einops.rearrange(x_t, "(b f) t c->b c t f", b=nB)

        return x_, (attn_f, attn_t)


if __name__ == "__main__":
    inp = torch.randn(2, 20, 10, 65)
    c = torch.randn(2, 65, 20)
    # net = ConditionalFTConformer(20)
    net = ConditionalFTConformerIter(20)
    out, c_, _ = net(inp, c)
    print(out.shape, c_.shape)
