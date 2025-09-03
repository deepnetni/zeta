import torch
import torch.nn as nn
import torch.nn.functional as F

import einops
from einops import rearrange
from einops.layers.torch import Rearrange
from scipy.signal import get_window
import numpy as np

from typing import Optional, List
from comps.stackedConv2d import StackedConv2d, StackedTransposedConv2d
from comps.conv_stft import STFT


def zca_whiting(inp):
    """
    inp: B,T
    """
    pass


class FrameConvert(nn.Module):
    def __init__(self, nblk) -> None:
        """Frame2Frame Time <--> Frequency convertion."""
        super().__init__()

        self.register_buffer("win", torch.tensor(get_window("hann", 2 * nblk)).sqrt().float())
        self.nblk = nblk

    def reset_buff(self, inp):
        nB = inp.size(0)

        self.buff_frame = torch.zeros(nB, 2 * self.nblk, dtype=inp.dtype).to(inp.device)
        self.buff_ola = torch.zeros(nB, self.nblk, dtype=inp.dtype).to(inp.device)

    def inverse(self, xk):
        """
        xk: B,2,1,F
        """
        assert xk.shape[2] == 1

        xk = rearrange(xk, "b c 1 f -> b f c").contiguous()
        x = torch.fft.irfft(torch.view_as_complex(xk), n=self.nblk * 2, dim=-1)
        x = x * self.win

        out = self.buff_ola + x[..., : self.nblk]

        self.buff_ola = x[..., -self.nblk :]

        return out

    def transform(self, inp):
        """
        return: B,2,1,F
        """
        self.buff_frame[:, : self.nblk] = self.buff_frame[:, self.nblk :]
        self.buff_frame[:, self.nblk :] = inp

        # B,F,2
        xk = torch.fft.rfft(self.buff_frame * self.win, n=self.nblk * 2, dim=-1)
        # B,2,1,F
        xk = torch.view_as_real(xk).permute(0, 2, 1).unsqueeze(2)

        return xk


class AFCNet(nn.Module):
    def __init__(self, nblk) -> None:
        super().__init__()
        self._is_init_buff = False
        self.convert = FrameConvert(nblk)
        self.convert_ref = FrameConvert(nblk)

    def reset_buff(self, inp):
        self.convert.reset_buff(inp)
        self.convert_ref.reset_buff(inp)

    def forward(self, inp, ref):
        if not self._is_init_buff:
            self.reset_buff(inp)
            self._is_init_buff = True


# class CRN(nn.Module):
#     def __init__(self, nblk) -> None:
#         super().__init__()

#         self.encoder = StackedConv2d([4, 16, 32], (1, 3), (1, 2))
#         self.decoder = StackedTransposedConv2d([4, 16, 32][::-1], (1, 3), (1, 2))
#         self.post = nn.Sequential(nn.Conv2d(4, 1, (1, 1)), nn.Sigmoid())

#         self.gru = nn.GRU(32, 64, batch_first=True)
#         self.linear = nn.Sequential(
#             nn.Linear(64, 32),
#             Rearrange("(b f) t c->(b t) f c", f=nblk // 4 + 1),
#             nn.BatchNorm1d(nblk // 4 + 1),
#         )

#         self.nblk = nblk

#         self._is_init_buff = False
#         self.convert = FrameConvert(nblk)
#         self.convert_ref = FrameConvert(nblk)

#     def reset_buff(self, inp):
#         self.convert.reset_buff(inp)
#         self.convert_ref.reset_buff(inp)

#     def init_gru(self, inp):
#         nB = inp.size(0)
#         nF = self.nblk // 4 + 1
#         h0 = torch.zeros(1, nB * nF, 64).to(inp.device)
#         return h0

#     def forward(self, inp, ref, h=None, **kwargs):
#         if not self._is_init_buff:
#             self.reset_buff(inp)
#             self._is_init_buff = True

#         if h is None:
#             h = self.init_gru(inp)

#         xk = self.convert.transform(inp)  # b,2,1,f
#         # b,1,t,f
#         xk_mag = xk.pow(2).sum(1, keepdim=True).sqrt()
#         # b,1,t,f
#         xk_pha = torch.atan2(xk[:, (1,), ...], xk[:, (0,), ...])
#         xkr = self.convert_ref.transform(ref)

#         x = torch.concat([xk, xkr], dim=1)

#         x, stat = self.encoder(x)

#         x = rearrange(x, "b c t f->(b f) t c")
#         x, stat_g = self.gru(x, h)
#         x = self.linear(x)
#         x = rearrange(x, "(b t) f c->b c t f", b=inp.size(0))

#         x = self.decoder(x, stat[::-1])
#         m = self.post(x)

#         xk_mag_ = xk_mag * m
#         xk_i = xk_mag_ * torch.sin(xk_pha)
#         xk_r = xk_mag_ * torch.cos(xk_pha)
#         xk = torch.concat([xk_r, xk_i], dim=1)

#         out = self.convert.inverse(xk)

#         return out, stat_g


class CRN(nn.Module):
    def __init__(self, nblk) -> None:
        super().__init__()

        self.encoder = StackedConv2d([4, 16, 32], (1, 3), (1, 2))
        self.decoder = StackedTransposedConv2d([4, 16, 32][::-1], (1, 3), (1, 2))
        self.post = nn.Sequential(nn.Conv2d(4, 1, (1, 1)), nn.Sigmoid())

        self.gru = nn.GRU(32, 64, batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(64, 32),
            Rearrange("(b f) t c->(b t) f c", f=nblk // 4 + 1),
            nn.BatchNorm1d(nblk // 4 + 1),
        )

        self.nblk = nblk

        self.convert_ola = FrameConvert(nblk)
        self.convert_ola_ref = FrameConvert(nblk)
        self.convert_stft = STFT(128, 64, win="hann sqrt")

    def reset_buff(self, inp):
        self.convert_ola.reset_buff(inp)
        self.convert_ola_ref.reset_buff(inp)

    def init_gru(self, inp):
        nB = inp.size(0)
        nF = self.nblk // 4 + 1
        h0 = torch.zeros(1, nB * nF, 64).to(inp.device)
        return h0

    def forward(self, inp, ref, h=None, online=True):
        """
        call `reset_buff` first.

        inp: B,T
        ref: B,T
        """
        if h is None:
            h = self.init_gru(inp)

        if not online:
            xk = self.convert_stft.transform(inp)  # b,2,t,f
            xkr = self.convert_stft.transform(ref)  # b,2,t,f
        else:
            xk = self.convert_ola.transform(inp)  # b,2,1,f
            xkr = self.convert_ola_ref.transform(ref)

        # b,1,t,f
        xk_mag = xk.pow(2).sum(1, keepdim=True).sqrt()
        # b,1,t,f
        xk_pha = torch.atan2(xk[:, (1,), ...], xk[:, (0,), ...])

        x = torch.concat([xk, xkr], dim=1)

        x, stat = self.encoder(x)

        x = rearrange(x, "b c t f->(b f) t c")
        x, stat_g = self.gru(x, h)
        x = self.linear(x)
        x = rearrange(x, "(b t) f c->b c t f", b=inp.size(0))

        x = self.decoder(x, stat[::-1])
        m = self.post(x)

        xk_mag_ = xk_mag * m
        xk_i = xk_mag_ * torch.sin(xk_pha)
        xk_r = xk_mag_ * torch.cos(xk_pha)
        xk = torch.concat([xk_r, xk_i], dim=1)

        if not online:
            out = self.convert_stft.inverse(xk)
        else:
            out = self.convert_ola.inverse(xk)

        return out, stat_g


if __name__ == "__main__":
    inp = torch.randn(1, 512).float()
    ref = torch.randn(1, 64).float()
    net = CRN(64)

    o = []
    for i in range(512 // 64):
        d = inp[:, i * 64 : i * 64 + 64]
        out, _ = net(d, ref)
        o.append(out)

    o = torch.concat(o, dim=-1)
    print(o.shape, o.dtype, inp.dtype)
    o = o[:, 64:]
    inp = inp[:, :-64]
    print(torch.allclose(o[:, :256], inp[:, :256], 1e-7, 1e-6))

    # inp = torch.randn(1, 16000)
    # ref = torch.randn(1, 16000)
    # net = CRN_E2E(64)
    # out, _ = net(inp, ref)
    # print(out.shape)
