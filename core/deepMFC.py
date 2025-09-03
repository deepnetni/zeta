import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

import einops
from einops import rearrange
from einops.layers.torch import Rearrange


from comps.stackedConv2d import StackedConv2d
from comps.conv_stft import STFT
from comps.overlapAdd import OverlapAdd


class EncoderBlk(nn.Module):
    def __init__(self, cin, cout, kernel, stride) -> None:
        super().__init__()

        nt, nf = (*kernel,)
        padf = (nf - 1) // 2
        padt = nt - 1
        self.conv1 = nn.Sequential(
            nn.ConstantPad2d((padf, padf, padt, 0), value=0.0),
            nn.Conv2d(cin, cout, kernel_size=kernel, stride=stride),
        )
        self.conv2 = nn.Sequential(
            nn.ConstantPad2d((padf, padf, padt, 0), value=0.0),
            nn.Conv2d(cin, cout, kernel_size=kernel, stride=stride),
            nn.Sigmoid(),
        )

        self.post = nn.Sequential(nn.BatchNorm2d(cout), nn.ELU())

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x = x1 * x2
        x = self.post(x)

        return x


class DecoderBlk(nn.Module):
    def __init__(self, cin, cout, kernel, stride) -> None:
        super().__init__()

        nt, nf = (*kernel,)
        padf = (nf - 1) // 2
        padt = nt - 1
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(cin, cout, kernel, stride, padding=(padt, padf)),
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(cin, cout, kernel, stride, padding=(padt, padf)),
            nn.Sigmoid(),
        )

        self.post = nn.Sequential(nn.BatchNorm2d(cout), nn.ELU())

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x = x1 * x2
        x = self.post(x)

        return x


class GLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sz,
        groups: int = 4,
        bidirectional=False,
        batch_first=True,
        **kw,
    ):
        super().__init__()
        assert input_size % groups == 0 and hidden_sz % groups == 0

        input_sz = input_size // groups
        hidden_sz = hidden_sz // groups
        batch_first = batch_first
        self.lstms = nn.ModuleList(
            [
                nn.LSTM(
                    input_size=input_sz,
                    hidden_size=hidden_sz,
                    batch_first=batch_first,
                    bidirectional=bidirectional,
                    **kw,
                )
                for _ in range(groups)
            ]
        )

        self.input_sz = input_sz

    def _split(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        # x: (B, T, C) or (T, B, C)
        return torch.split(x, self.input_sz, dim=-1)

    def _concat(self, parts: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(parts, dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        h0: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ):
        xs = self._split(x)

        y_parts = []
        hn_list = []

        for g, (xg, lstm) in enumerate(zip(xs, self.lstms)):
            if h0 is not None:
                h0_g, c0_g = h0[g]
                yg, (hn_g, cn_g) = lstm(xg, (h0_g, c0_g))
            else:
                yg, (hn_g, cn_g) = lstm(xg)

            y_parts.append(yg)
            hn_list.append((hn_g, cn_g))

        y = self._concat(y_parts)
        return y, hn_list


class DeepMFC(nn.Module):
    def __init__(self, nblk, chs=[2, 16, 32, 64, 128, 256]) -> None:
        super().__init__()
        self.encoder_l = nn.ModuleList()
        for cin, cout in zip(chs[:-1], chs[1:]):
            self.encoder_l.append(EncoderBlk(cin, cout, (1, 3), (1, 2)))

        self.decoder_l_r = nn.ModuleList()
        self.decoder_l_i = nn.ModuleList()
        chs = chs[1:][::-1] + [1]
        for cin, cout in zip(chs[:-1], chs[1:]):
            self.decoder_l_r.append(DecoderBlk(cin * 2, cout, (1, 3), (1, 2)))
            self.decoder_l_i.append(DecoderBlk(cin * 2, cout, (1, 3), (1, 2)))

        self.fc = nn.ModuleList([nn.Linear(nblk + 1, nblk + 1) for _ in range(2)])
        self.glstm = GLSTM(768, 768, 16)

        self.stft = STFT(nblk * 2, nblk, win="hann sqrt")
        self.ola = OverlapAdd(nblk)

    def reset_buff(self, inp):
        self.ola.reset_buff(inp)

    def forward(self, x, h_list=None, online=False):
        if not online:
            xk = self.stft.transform(x)  # b,2,t,f
        else:
            xk = self.ola.transform(x)  # b,2,1,f

        stat = []
        for l in self.encoder_l:
            xk = l(xk)
            stat.append(xk)

        nC = xk.shape[1]
        xk = einops.rearrange(xk, "b c t f -> b t (c f)")
        xk, h_list = self.glstm(xk, h_list)
        xk = einops.rearrange(xk, "b t (c f)->b c t f", c=nC)

        xkr, xki = xk, xk
        for lr, li, xp in zip(self.decoder_l_r, self.decoder_l_i, stat[::-1]):
            xkr = torch.concat([xp, xkr], dim=1)
            xkr = lr(xkr)

            xki = torch.concat([xp, xki], dim=1)
            xki = li(xki)

        xkr = self.fc[0](xkr)
        xki = self.fc[1](xki)

        xk = torch.concat([xkr, xki], dim=1)

        if not online:
            x = self.stft.inverse(xk)
        else:
            x = self.ola.inverse(xk)

        return x, h_list


def check():
    inp = torch.randn(3, 640)
    inp_p = F.pad(inp, (0, 64))
    net = DeepMFC(64)
    net.eval()

    l = []
    h = None

    out, h = net(inp, None, False)
    # print(out[0, 0, 0, :10])
    # print(out[0, :10])

    net.reset_buff(inp)
    for i in range(11):
        st = i * 64
        ed = i * 64 + 64
        out_, h = net(inp_p[:, st:ed], h, True)
        l.append(out_)

    # print(l[0][0, 0, 0, :10])
    # print(l[1][0, :10])
    out_ = torch.concat(l[1:], dim=-1)

    print(torch.allclose(out_, out, 1e-4, 1e-3))


if __name__ == "__main__":
    check()

    # out__ = torch.concat(l[1:], dim=-1)
    # print(out.shape, out__.shape)
    # print(out[0, :20])
    # print(out__[0, :20])

    # diff = torch.abs(out__ - out).sum()
    # print(diff)
