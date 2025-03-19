import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

from JointNSHModel import expand_HT
from models.conv_stft import STFT
from models.s4d import S4D
from models.stackedConv2d import StackedConv2d, StackedTransposedConv2d
from utils.check_flops import check_flops


class S4DBLK(nn.Module):
    def __init__(self, ndim) -> None:
        super().__init__()

        self.layer = nn.Sequential(
            S4D(ndim),  # B,C,T
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Conv1d(ndim, 2 * ndim, 1),
            # Rearrange("b c t->b t c"),
            nn.GLU(dim=1),
            nn.Dropout(0.2),
            # Rearrange("b t c->b c t"),
        )
        self.post = nn.Sequential(
            Rearrange("b c t->b t c"),
            nn.LayerNorm(ndim),
            Rearrange("b t c->b c t"),
        )

    def forward(self, inp):
        x = self.layer(inp)
        x = self.post(inp + x)
        return x


class Interaction(nn.Module):
    def __init__(self, ndim) -> None:
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(ndim * 2, ndim, (1, 1)), nn.BatchNorm2d(ndim), nn.Sigmoid()
        )

    def forward(self, x, y):
        """
        x: b,c,t,f
        """
        d = torch.concat([x, y], dim=1)
        w = self.layer(d)
        return y * w + x


class DualS4D(nn.Module):
    def __init__(self, nframe=512, nhop=256, channel=[16, 32, 64, 128, 256, 256]) -> None:
        super().__init__()

        self.stft = STFT(nframe, nhop, nframe)

        chs = [2] + channel
        chs_ = [1] + channel

        self.spec_encoder = StackedConv2d(chs, (2, 5), (1, 2))
        self.spec_decoder = StackedTransposedConv2d(chs[::-1], (2, 5), (1, 2), skip=False)
        self.spec_inter = nn.ModuleList([Interaction(256) for _ in range(2)])
        self.spec_s4d_blks = nn.Sequential(*[S4DBLK(256) for _ in range(4)])

        self.mag_encoder = StackedConv2d(chs_, (2, 5), (1, 2))
        self.mag_decoder = StackedTransposedConv2d(chs_[::-1], (2, 5), (1, 2), skip=False)
        self.mag_inter = nn.ModuleList([Interaction(256) for _ in range(2)])
        self.mag_s4d_blks = nn.Sequential(*[S4DBLK(256) for _ in range(4)])

    def forward(self, inp):
        nB = inp.size(0)
        x = self.stft.transform(inp)
        mag = x.pow(2).sum(1, keepdim=True).sqrt()
        pha = torch.atan2(x[:, (1,), ...], x[:, (0,), ...])

        xk, _ = self.spec_encoder(x)
        xm, _ = self.mag_encoder(mag)
        xk = self.spec_inter[0](xk, xm)
        xm = self.mag_inter[0](xm, xk)

        xk = rearrange(xk, "b c t f-> (b f) c t")
        xm = rearrange(xm, "b c t f-> (b f) c t")
        # xk = rearrange(xk, "b c t f-> b (c f) t")
        # xm = rearrange(xm, "b c t f-> b (c f) t")

        xk = self.spec_s4d_blks(xk)
        xm = self.mag_s4d_blks(xk)

        xk = rearrange(xk, "(b f) c t -> b c t f", b=nB)
        xm = rearrange(xm, "(b f) c t -> b c t f", b=nB)
        # xk = rearrange(xk, "b (c f) t -> b c t f", c=nB)
        # xm = rearrange(xm, "b (c f) t -> b c t f", c=nB)
        xk = self.spec_inter[1](xk, xm)
        xm = self.mag_inter[1](xm, xk)

        xk = self.spec_decoder(xk)
        xm = self.mag_decoder(xm)

        enh_r = xm * torch.cos(pha) + xk[:, (0,), ...]
        enh_i = xm * torch.sin(pha) + xk[:, (1,), ...]

        enh_spec = torch.concat([enh_r, enh_i], dim=1)
        enh = self.stft.inverse(enh_spec)

        return enh


class DualS4DFIG6(nn.Module):
    def __init__(self, nframe=512, nhop=256, channel=[16, 32, 64, 128, 256, 256]) -> None:
        super().__init__()

        self.stft = STFT(nframe, nhop, nframe)

        chs = [3] + channel
        chs_ = [2] + channel

        self.spec_encoder = StackedConv2d(chs, (2, 5), (1, 2))
        self.spec_decoder = StackedTransposedConv2d(chs[::-1], (2, 5), (1, 2), skip=False)
        self.spec_inter = nn.ModuleList([Interaction(256) for _ in range(2)])
        self.spec_s4d_blks = nn.Sequential(*[S4DBLK(256) for _ in range(4)])

        self.mag_encoder = StackedConv2d(chs_, (2, 5), (1, 2))
        self.mag_decoder = StackedTransposedConv2d(chs_[::-1], (2, 5), (1, 2), skip=False)
        self.mag_inter = nn.ModuleList([Interaction(256) for _ in range(2)])
        self.mag_s4d_blks = nn.Sequential(*[S4DBLK(256) for _ in range(4)])

        self.reso = 16000 / nframe

    def forward(self, inp, HL):
        nB = inp.size(0)
        x = self.stft.transform(inp)

        hl = expand_HT(HL, x.shape[-2], self.reso)  # B,C(1),T,F

        mag = x.pow(2).sum(1, keepdim=True).sqrt()
        mag = torch.concat([mag, hl], dim=1)
        pha = torch.atan2(x[:, (1,), ...], x[:, (0,), ...])

        x = torch.concat([x, hl], dim=1)
        xk, _ = self.spec_encoder(x)
        xm, _ = self.mag_encoder(mag)
        xk = self.spec_inter[0](xk, xm)
        xm = self.mag_inter[0](xm, xk)

        xk = rearrange(xk, "b c t f-> (b f) c t")
        xm = rearrange(xm, "b c t f-> (b f) c t")
        # xk = rearrange(xk, "b c t f-> b (c f) t")
        # xm = rearrange(xm, "b c t f-> b (c f) t")

        xk = self.spec_s4d_blks(xk)
        xm = self.mag_s4d_blks(xk)

        xk = rearrange(xk, "(b f) c t -> b c t f", b=nB)
        xm = rearrange(xm, "(b f) c t -> b c t f", b=nB)
        # xk = rearrange(xk, "b (c f) t -> b c t f", c=nB)
        # xm = rearrange(xm, "b (c f) t -> b c t f", c=nB)
        xk = self.spec_inter[1](xk, xm)
        xm = self.mag_inter[1](xm, xk)

        xk = self.spec_decoder(xk)
        xm = self.mag_decoder(xm)

        enh_r = xm * torch.cos(pha) + xk[:, (0,), ...]
        enh_i = xm * torch.sin(pha) + xk[:, (1,), ...]

        enh_spec = torch.concat([enh_r, enh_i], dim=1)
        enh = self.stft.inverse(enh_spec)

        return enh


if __name__ == "__main__":
    # inp = torch.randn(1, 16000)
    # net = DualS4D()
    # out = net(inp)
    # print(out.shape)

    # check_flops(net, inp)

    inp = torch.randn(1, 16000)
    hl = torch.randn(1, 6)
    net = DualS4DFIG6()
    out = net(inp, hl)
    print(out.shape)

    check_flops(net, inp, hl)
