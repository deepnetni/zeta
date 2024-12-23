from numba.cuda.simulator import kernel
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import einops
from einops import rearrange
from einops.layers.torch import Rearrange
from utils.register import tables
from utils.check_flops import check_flops
from models.conv_stft import STFT


class FTGRU_RESNET_Train(nn.Module):
    """Input and output has the same dimension. Operation along the C dim.
    Input:  B,C,T,F
    Return: B,C,T,F

    Args:
        input_size: should be equal to C of input shape B,C,T,F
        hidden_size: input_size -> hidden_size
        batch_first: input shape is B,C,T,F if true
        use_fc: add fc layer after lstm
    """

    def __init__(self, input_size, hidden_size, batch_first=True, use_fc=True):
        super().__init__()

        if not isinstance(hidden_size, tuple):
            hidden_size = (hidden_size, hidden_size)

        assert (
            not use_fc and input_size == hidden_size[0]
        ) or use_fc, f"hidden_size {hidden_size[-1]} should equals to input_size {input_size} when use_fc is True"

        self.f_unit = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size[0],  # bi-directional LSTM output is 2xhidden_size
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        if use_fc:
            self.f_post = nn.Sequential(
                nn.Linear(2 * hidden_size[0], input_size),
                nn.LayerNorm(input_size),
            )
        else:
            self.f_post = nn.Identity()

        self.t_unit = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size[1],
            batch_first=batch_first,
        )

        if use_fc:
            self.t_post = nn.Sequential(
                nn.Linear(hidden_size[1], input_size),
                nn.LayerNorm(input_size),
            )
        else:
            self.t_post = nn.Identity()

    def forward(self, inp: torch.Tensor):
        """
        Args:
            x: input shape should be B,C,T,F
        """
        nB = inp.shape[0]

        # step1. F-LSTM
        x = einops.rearrange(inp, "b c t f-> (b t) f c")  # BxT,F,C
        # x = inp.permute(0, 2, 3, 1)  # B, T, F, C
        # x = x.reshape(-1, nF, nC)  # BxT,F,C
        x, _ = self.f_unit(x)  # BxT,F,C
        x = self.f_post(x)
        # BxT,F,C => B,C,T,F
        x = einops.rearrange(x, "(b t) f c-> b c t f", b=nB)
        # x = x.reshape(nB, nT, nF, nC)
        # x = x.permute(0, 3, 1, 2)  # B,C,T,F
        inp = inp + x

        # step2. T-LSTM
        x = einops.rearrange(inp, "b c t f->(b f) t c")  # BxF,T,C
        x, _ = self.t_unit(x)
        x = self.t_post(x)
        x = einops.rearrange(x, "(b f) t c -> b c t f", b=nB)
        inp = inp + x

        return inp


class Encoder(nn.Module):
    """
    input: b,c,f,t
    """

    def __init__(
        self,
        inp_channel: int = 3,
        channels=[16, 16, 16, 32, 32, 32, 32, 32],
        kernels=[(5, 1), (1, 5), (6, 5), (4, 3), (6, 5), (5, 3), (3, 5), (3, 3)],
        strides=[(1, 1), (1, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (1, 1)],
    ) -> None:
        super().__init__()

        assert len(kernels) == len(strides) == len(channels)
        channels = [inp_channel] + channels

        self.layers = nn.ModuleList()
        for cin, cout, k, s in zip(channels[:-1], channels[1:], kernels, strides):
            kf, kt = (*k,)
            p0 = (kf - 1) // 2 if kf % 2 == 1 else (kf // 2) - 1
            p1 = (kf - 1) // 2 if kf % 2 == 1 else kf // 2
            p2 = kt - 1
            self.layers.append(
                nn.Sequential(
                    nn.ConstantPad2d((p2, 0, p0, p1), value=0.0),
                    nn.Conv2d(cin, cout, k, s),
                    nn.BatchNorm2d(cout),
                    nn.PReLU(),
                ),
            )

    def forward(self, x):
        out = []

        for l in self.layers:
            x = l(x)
            out.append(x)

        return x, out


class GatedTrConv2D(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel, stride) -> None:
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(inp_channels * 2, inp_channels, (1, 1), (1, 1)),
            nn.Tanh(),
        )
        kf, kt = (*kernel,)
        p0 = (kf - 1) // 2 if kf % 2 == 1 else (kf // 2) - 1
        p1 = (kf - 1) // 2 if kf % 2 == 1 else kf // 2
        p2 = kt - 1
        self.up_sample = nn.Sequential(
            nn.ConvTranspose2d(inp_channels, out_channels, kernel, stride),
            nn.ConstantPad2d((-p2, 0, -p0, -p1), value=0.0),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
        )

    def forward(self, x, y):
        """
        x: b,c,f,t
        """

        c = torch.concat([x, y], dim=1)
        x = x * self.layer(c)
        x = self.up_sample(x)

        return x


class Decoder(nn.Module):
    """
    input: b,c,f,t
    """

    def __init__(
        self,
        inp_channel: int = 3,
        channels=[16, 16, 16, 32, 32, 32, 32, 32],
        kernels=[(5, 1), (1, 5), (6, 5), (4, 3), (6, 5), (5, 3), (3, 5), (3, 3)],
        strides=[(1, 1), (1, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (1, 1)],
    ) -> None:
        super().__init__()

        assert len(kernels) == len(strides) == len(channels)
        channels = [inp_channel] + channels

        self.layers = nn.ModuleList()
        for cin, cout, k, s in zip(
            channels[::-1][:-1], channels[::-1][1:], kernels[::-1], strides[::-1]
        ):
            self.layers.append(GatedTrConv2D(cin, cout, k, s))

    def forward(self, x, y: list):
        for l, y_ in zip(self.layers, y[::-1]):
            x = l(x, y_)

        return x


class VAD_BLK(nn.Module):
    """
    input: b,c,t,f
    """

    def __init__(self, inp_channels=32, out_channels=16) -> None:
        super().__init__()

        self.pre_l = nn.Sequential(
            nn.Conv2d(inp_channels, out_channels, (1, 1), (1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
        )  # b,c,t,f

        # bt,f,c
        self.f_gru = nn.GRU(
            input_size=out_channels,
            hidden_size=8,  # bi-directional LSTM output is 2xhidden_size
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.post = nn.Sequential(
            nn.Conv1d(16, 16, 1, 1),
            nn.BatchNorm1d(16),
            nn.PReLU(),
            nn.Conv1d(16, 2, 1, 1),
        )

    def forward(self, x):
        nB = x.shape[0]
        x = self.pre_l(x)
        x = einops.rearrange(x, "b c t f-> (b t) f c")  # BxT,F,C
        x, h = self.f_gru(x)  # bt,f,2xh; hidden: 2x1,bxt,hidden(8)
        h = einops.rearrange(h, "d (b t) h->b (h d) t", b=nB)  # B,16,T
        x = self.post(h)  # b,2,T

        return x.permute(0, 2, 1)  # B,T,2


@tables.register("models", "TPNN")
class TPNN(nn.Module):
    def __init__(self, nframe=320, nhop=160, nf=3, nt=3, nl=1) -> None:
        super().__init__()

        self.stft = STFT(nframe, nhop, nfft=nframe)
        nbin = nframe // 2 + 1

        self.enc = Encoder(2)
        self.rnn = nn.Sequential(
            FTGRU_RESNET_Train(input_size=32, hidden_size=(32, 64)),
            FTGRU_RESNET_Train(input_size=32, hidden_size=(32, 32)),
        )
        self.dec = Decoder(2)
        self.mask = nn.Sequential(nn.Linear(nbin, nbin), nn.Sigmoid())
        self.vad_blk = VAD_BLK(32, 16)

        self.enc_ = Encoder(3, channels=[16, 16, 32, 32, 64, 64, 64, 64])
        self.rnn_ = nn.Sequential(
            FTGRU_RESNET_Train(input_size=64, hidden_size=(64, 128)),
            FTGRU_RESNET_Train(input_size=64, hidden_size=(64, 64)),
        )
        self.nf_ = 2 * nf + 1
        self.nt = nt
        self.nl = nl
        self.nt_ = nt + nl + 1
        self.dec_ = Decoder(
            2 * self.nf_ * self.nt_, channels=[16, 16, 32, 32, 64, 64, 64, 64]
        )  # b,2,f,t
        self.post = nn.Sequential(nn.Linear(nbin, nbin), nn.Sigmoid())

    def speech_unfold(self, x: torch.Tensor):
        """
        x: b,c,t,f
        """

        pf = (self.nf_ - 1) // 2
        x = F.pad(x, (pf, pf, self.nt, self.nl))
        # x = F.pad(x, (pf, pf, self.nt_ - 1, 0))
        x = x.unfold(2, self.nt + self.nl + 1, 1).permute(0, 1, 2, 4, 3)  # b,c,t,f,nt -> b,c,t,nt,f
        x = x.unfold(4, self.nf_, 1).permute(0, 1, 2, 4, 3, 5)  # b,c,t,nt,f,nf -> b,c,t,f,nt,nf
        return x

    def forward(self, mic, ref):
        xk_mic = self.stft.transform(mic)
        xk_ref = self.stft.transform(ref)

        mag_mic = xk_mic.pow(2).sum(1).sqrt().pow(0.3)
        mag_ref = xk_ref.pow(2).sum(1).sqrt().pow(0.3)

        inp_c = torch.stack([mag_mic, mag_ref], dim=1)
        inp_c = inp_c.permute(0, 1, 3, 2)  # bctf->bcft
        x, x_steps = self.enc(inp_c)
        x = x.permute(0, 1, 3, 2)  # bcft->bctf

        x = self.rnn(x)

        vad_p = self.vad_blk(x)

        x = x.permute(0, 1, 3, 2)  # bctf->bcft
        x = self.dec(x, x_steps)  # b,2,f,t
        x = x.permute(0, 1, 3, 2)  # bcft->bctf
        mask = self.mask(x)

        xk_mic_1 = mask * xk_mic
        mag_mic_1 = xk_mic_1.pow(2).sum(1).sqrt().pow(0.3)
        inp_1 = torch.stack([mag_mic, mag_ref, mag_mic_1], dim=1)
        inp_1 = inp_1.permute(0, 1, 3, 2)  # bctf->bcft
        x, x_steps = self.enc_(inp_1)
        x = x.permute(0, 1, 3, 2)  # bcft->bctf

        x = self.rnn_(x)

        x = x.permute(0, 1, 3, 2)  # bctf->bcft
        x = self.dec_(x, x_steps)  # b,2,f,t
        x = x.permute(0, 1, 3, 2)  # bcft->bctf
        x = self.post(x)
        mask = einops.rearrange(x, "b (c nt nf) t f->b c t f nt nf", c=2, nt=self.nt_, nf=self.nf_)

        spec = self.speech_unfold(xk_mic_1)
        spec = torch.einsum("...tf,...tf->...", spec, mask)

        out_c = self.stft.inverse(xk_mic_1)
        out_f = self.stft.inverse(spec)
        return out_f, out_c, vad_p


if __name__ == "__main__":
    inp = torch.randn(1, 16000)
    net = TPNN()
    out, out_c, _ = net(inp, inp)
    print(out.shape)

    check_flops(net, inp, inp)

    for k, v in reversed(list(net.named_parameters())):
        print(k, v)
    # inp = torch.randn(1, 2, 3, 4)
    # out = net.speech_unfold(inp)
    # print(inp)
