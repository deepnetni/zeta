import sys
from typing import List, Union

import einops
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch import Size, Tensor

sys.path.append(__file__.rsplit("/", 1)[0])

from models.conformer import GLU, FeedForward, PreNorm, Scale
from models.conv_stft import STFT
from models.FTConformerBLK import (
    ConditionalFTConformer,
    ConditionalFTConformerIter,
    FTConformer,
    FTDiTConformer,
)
from models.Fusion.ms_cam import AFF
from models.gumbel_vector_quantizer import GumbelVectorQuantizer
from utils.register import tables


def expand_HT(ht: torch.Tensor, T: int, reso) -> torch.Tensor:
    """

    :param ht: B,6
    :param T:
    :param reso:

    return b,1,t,f
    """
    # batch_size = ht.shape[0]
    # Freq_size = self.nbin

    m = int(250 / reso)
    bandarray = torch.tensor([0] + [(2**i) * m for i in range(ht.shape[1])]).to(ht.device)

    repeat_n = bandarray[1:] - bandarray[:-1]
    repeat_n[0] += 1

    expand_ht = ht.repeat_interleave(repeat_n, dim=-1).unsqueeze(1).unsqueeze(1)  # B,1,1,nbin
    expand_ht = expand_ht.repeat(1, 1, T, 1) / 100.0

    return expand_ht


def compute_subbands_energy(xk, index):
    """
    xk: B,2,T,F
    subbands: index of (bands,)
    """
    if torch.is_complex(xk):
        # b,t,f->b,2,t,f
        xk = torch.stack([xk.real, xk.imag], dim=1)

    pow_l = []
    for i, (low, high) in enumerate(zip(index[:-1], index[1:])):
        pow = torch.sum(xk[..., low:high] ** 2, dim=(1, 3))  # B,T
        pow = pow / (high - low)
        pow_l.append(pow)

    pow_c = torch.stack(pow_l, dim=-1)  # B,T,C
    return 10 * torch.log10(pow_c + 1e-7)


class InstanceNorm(nn.Module):
    """Normalization along the last two dimensions, and the output shape is equal to that of the input.
    Input: B,C,T,F

    Args:
        feats: CxF with input B,C,T,F
    """

    def __init__(self, feats=1):
        super().__init__()
        self.eps = torch.finfo(torch.float32).eps
        self.gamma = nn.Parameter(torch.ones(feats))
        self.beta = nn.Parameter(torch.zeros(feats))

    def forward(self, inputs: torch.Tensor):
        """
        inputs shape is (B, C, T, F)
        """
        nB, nC, nT, nF = inputs.shape
        inputs = inputs.permute(0, 2, 1, 3).flatten(-2)  # B, T, CxF

        mean = torch.mean(inputs, dim=-1, keepdim=True)
        var = torch.mean(torch.square(inputs - mean), dim=-1, keepdim=True)

        std = torch.sqrt(var + self.eps)

        outputs = (inputs - mean) / std
        outputs = outputs * self.gamma + self.beta

        outputs = outputs.reshape(nB, nT, nC, nF)
        outputs = outputs.permute(0, 2, 1, 3)  # B,C,T,F

        return outputs


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: Union[int, List[int], Size]) -> None:
        super().__init__()

        self.norm = nn.Sequential(
            Rearrange("b c t f-> b t f c"),
            nn.LayerNorm(normalized_shape=normalized_shape),
            Rearrange("b t f c-> b c t f"),
        )

    def forward(self, x: Tensor):
        """
        x: b,c,t,f
        """
        return self.norm(x)


class DilatedDenseNet(nn.Module):
    def __init__(
        self,
        depth=4,
        in_channels=64,
        kernel_size=(2, 3),
    ):
        super().__init__()
        self.depth = depth
        self.in_channels = in_channels
        twidth = kernel_size[0]
        for i in range(self.depth):
            dil = 2**i
            pad_length = twidth + (dil - 1) * (twidth - 1) - 1

            setattr(
                self,
                "conv{}".format(i + 1),
                nn.Sequential(
                    nn.ConstantPad2d((1, 1, pad_length, 0), value=0.0),  # lrtb
                    nn.Conv2d(
                        in_channels * (i + 1),
                        in_channels,
                        kernel_size=kernel_size,
                        dilation=(dil, 1),
                    ),
                    # nn.BatchNorm2d(in_channels),
                    LayerNorm(in_channels),
                    nn.PReLU(in_channels),
                ),
            )

        self.post = nn.Sequential(nn.Conv2d(in_channels, in_channels, (1, 1), (1, 1)))

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, "conv{}".format(i + 1))(skip)
            skip = torch.cat([out, skip], dim=1)

        return out


class DilatedDenseNet_(nn.Module):
    def __init__(
        self,
        depth=4,
        in_channels=64,
        kernel_size=(2, 3),
    ):
        super().__init__()
        self.depth = depth
        self.in_channels = in_channels
        twidth = kernel_size[0]
        for i in range(self.depth):
            dil = 2**i
            pad_length = twidth + (dil - 1) * (twidth - 1) - 1

            setattr(
                self,
                "conv{}".format(i + 1),
                nn.Sequential(
                    nn.ConstantPad2d((1, 1, pad_length, 0), value=0.0),  # lrtb
                    nn.Conv2d(
                        in_channels * (i + 1),
                        in_channels,
                        kernel_size=kernel_size,
                        dilation=(dil, 1),
                    ),
                    LayerNorm(in_channels),
                    nn.PReLU(in_channels),
                ),
            )

            setattr(
                self,
                "conv_{}".format(i + 1),
                nn.Sequential(
                    nn.Conv2d(in_channels * (i + 1), in_channels, (1, 3), (1, 1), (0, 1)),
                    LayerNorm(in_channels),
                    nn.PReLU(in_channels),
                ),
            )
            setattr(self, "fu{}".format(i + 1), AFF(in_channels, 65, r=1))

        self.post = nn.Sequential(nn.Conv2d(in_channels, in_channels, (1, 1), (1, 1)))

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out_1 = getattr(self, "conv{}".format(i + 1))(skip)
            out_2 = getattr(self, "conv_{}".format(i + 1))(skip)
            out = getattr(self, "fu{}".format(i + 1))(out_1, out_2)
            skip = torch.cat([out, skip], dim=1)

        return out


class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        super(SPConvTranspose2d, self).__init__()
        self.pad1 = nn.ConstantPad2d((1, 1, 0, 0), value=0.0)
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1))
        self.r = r

    def forward(self, x):
        x = self.pad1(x)
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)  # b,nc/r,h,w,r
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out


class DenseEncoder(nn.Module):
    def __init__(self, in_channel, channels=64):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, channels, (1, 3), (1, 2), padding=(0, 1)),
            # nn.InstanceNorm2d(channels, affine=True),
            LayerNorm(channels),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, (1, 3), (1, 2), padding=(0, 1)),
            # nn.InstanceNorm2d(channels, affine=True),
            LayerNorm(channels),
            nn.PReLU(channels),
        )
        self.dilated_dense = DilatedDenseNet(depth=4, in_channels=channels)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.dilated_dense(x)
        return x


class DenseEncoder8(nn.Module):
    def __init__(self, in_channel, channels=64):
        super(DenseEncoder8, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, channels, (1, 3), (1, 2), padding=(0, 1)),
            LayerNorm(channels),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, (1, 3), (1, 2), padding=(0, 1)),
            LayerNorm(channels),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, (1, 3), (1, 2), padding=(0, 1)),
            LayerNorm(channels),
            nn.PReLU(channels),
        )
        self.dilated_dense = DilatedDenseNet(depth=4, in_channels=channels)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.dilated_dense(x)
        return x


class MaskDecoder(nn.Module):
    def __init__(self, num_features, num_channel=64, out_channel=1):
        super().__init__()
        self.dense_block = DilatedDenseNet(depth=4, in_channels=num_channel)
        self.sub_pixel = nn.Sequential(
            SPConvTranspose2d(num_channel, num_channel, (1, 3), 2),
            nn.Conv2d(num_channel, num_channel, (1, 2)),
            # nn.InstanceNorm2d(num_channel),
            LayerNorm(num_channel),
            nn.PReLU(num_channel),
            SPConvTranspose2d(num_channel, num_channel, (1, 3), 2),
            nn.Conv2d(num_channel, out_channel, (1, 2)),
            # nn.InstanceNorm2d(out_channel, affine=True),
            nn.BatchNorm2d(out_channel),
            nn.PReLU(out_channel),
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, (1, 1)),
            Rearrange("b c t f->b f t c"),
            nn.PReLU(num_features),
            Rearrange("b f t c->b c t f"),
        )

    def forward(self, x):
        x = self.dense_block(x)
        x = self.sub_pixel(x)
        x = self.final_conv(x)
        return x


class MaskDecoder8(nn.Module):
    def __init__(self, num_features, num_channel=64, out_channel=1):
        super().__init__()
        self.dense_block = DilatedDenseNet(depth=4, in_channels=num_channel)
        self.sub_pixel = nn.Sequential(
            SPConvTranspose2d(num_channel, num_channel, (1, 3), 2),
            nn.Conv2d(num_channel, num_channel, (1, 2)),
            LayerNorm(num_channel),
            nn.PReLU(num_channel),
            SPConvTranspose2d(num_channel, num_channel, (1, 3), 2),
            nn.Conv2d(num_channel, num_channel, (1, 2)),
            LayerNorm(num_channel),
            nn.PReLU(num_channel),
            SPConvTranspose2d(num_channel, num_channel, (1, 3), 2),
            nn.Conv2d(num_channel, out_channel, (1, 2)),
            # nn.InstanceNorm2d(out_channel, affine=True),
            nn.BatchNorm2d(out_channel),
            nn.PReLU(out_channel),
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, (1, 1)),
            Rearrange("b c t f->b f t c"),
            nn.PReLU(num_features),
            Rearrange("b f t c->b c t f"),
        )

    def forward(self, x):
        x = self.dense_block(x)
        x = self.sub_pixel(x)
        x = self.final_conv(x)
        return x


class ComplexDecoder(nn.Module):
    def __init__(self, num_channel=64):
        super(ComplexDecoder, self).__init__()
        self.dense_block = DilatedDenseNet(depth=4, in_channels=num_channel)
        self.sub_pixel = nn.Sequential(
            SPConvTranspose2d(num_channel, num_channel, (1, 3), 2),
            nn.Conv2d(num_channel, num_channel, (1, 2)),
            # nn.InstanceNorm2d(num_channel, affine=True),
            LayerNorm(num_channel),
            nn.PReLU(num_channel),
            SPConvTranspose2d(num_channel, num_channel, (1, 3), 2),
            # nn.InstanceNorm2d(num_channel, affine=True),
            LayerNorm(num_channel),
            nn.PReLU(num_channel),
        )
        self.conv = nn.Conv2d(num_channel, 2, (1, 2))

    def forward(self, x):
        x = self.dense_block(x)
        x = self.sub_pixel(x)
        x = self.conv(x)
        return x


class ComplexDecoder8(nn.Module):
    def __init__(self, num_channel=64):
        super().__init__()
        self.dense_block = DilatedDenseNet(depth=4, in_channels=num_channel)
        self.sub_pixel = nn.Sequential(
            SPConvTranspose2d(num_channel, num_channel, (1, 3), 2),
            nn.Conv2d(num_channel, num_channel, (1, 2)),
            LayerNorm(num_channel),
            nn.PReLU(num_channel),
            SPConvTranspose2d(num_channel, num_channel, (1, 3), 2),
            nn.Conv2d(num_channel, num_channel, (1, 2)),
            LayerNorm(num_channel),
            nn.PReLU(num_channel),
            SPConvTranspose2d(num_channel, num_channel, (1, 3), 2),
            LayerNorm(num_channel),
            nn.PReLU(num_channel),
        )
        self.conv = nn.Conv2d(num_channel, 2, (1, 2))

    def forward(self, x):
        x = self.dense_block(x)
        x = self.sub_pixel(x)
        x = self.conv(x)
        return x


@tables.register("models", "baseline_fig6")
class Baseline(nn.Module):
    r"""In-context method"""

    def __init__(
        self, nframe: int, nhop: int, mid_channel: int = 64, conformer_num=4, fs=16000
    ) -> None:
        super().__init__()

        self.stft = STFT(nframe, nhop, nframe)
        self.reso = fs / nframe
        nbin = nframe // 2 + 1
        assert 250 % self.reso == 0

        self.encoder = DenseEncoder(in_channel=3, channels=mid_channel)
        self.ht_freq = torch.ceil(
            torch.tensor([0, 250, 500, 1000, 2000, 4000, 8001]) / self.reso
        ).int()

        self.conformer = nn.ModuleList([FTConformer(dim=mid_channel) for _ in range(conformer_num)])

        self.mask_decoder = MaskDecoder(num_features=nbin, num_channel=mid_channel, out_channel=1)
        self.complex_decoder = ComplexDecoder(num_channel=mid_channel)

    def forward(self, x, HL):
        """
        x: B,T
        HL: B,6
        """
        xk = self.stft.transform(x)
        xk_mag = xk.pow(2).sum(1, keepdim=True).sqrt()  # B,1,T,F
        # xk_bands_pow = compute_subbands_energy(xk, self.ht_freq)
        hl = expand_HT(HL, xk.shape[-2], self.reso)  # B,C(1),T,F
        xk = torch.cat([xk, hl], dim=1)

        xk = self.encoder(xk)

        for l in self.conformer:
            xk, _ = l(xk, causal=True)

        mask = self.mask_decoder(xk)
        spec = self.complex_decoder(xk)
        r, i = spec.chunk(2, dim=1)  # b,1,t,f
        phase = torch.atan2(i, r)

        xk_mag_est = xk_mag * mask
        spec_r = xk_mag_est * torch.cos(phase)
        spec_i = xk_mag_est * torch.sin(phase)

        xk_est = torch.concat([spec_r, spec_i], dim=1)

        x = self.stft.inverse(xk_est)

        # return x, spec
        return x


@tables.register("models", "baseVADSE")
class BaselineVADSE(nn.Module):
    r"""In-context method"""

    def __init__(
        self, nframe: int, nhop: int, mid_channel: int = 64, conformer_num=4, fs=16000
    ) -> None:
        super().__init__()

        self.stft = STFT(nframe, nhop, nframe)
        self.reso = fs / nframe
        nbin = nframe // 2 + 1
        assert 250 % self.reso == 0

        self.encoder = DenseEncoder(in_channel=2, channels=mid_channel)
        self.conformer = nn.ModuleList([FTConformer(dim=mid_channel) for _ in range(conformer_num)])

        self.mask_decoder = MaskDecoder(num_features=nbin, num_channel=mid_channel, out_channel=1)
        self.complex_decoder = ComplexDecoder(num_channel=mid_channel)

        self.vad_predictor = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1, 65), stride=(1, 65)),  # B,C,T,1
            nn.Conv2d(
                in_channels=mid_channel,
                out_channels=mid_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ),  # b,c,t,1
            Rearrange("b c t ()->b t c"),
            nn.LayerNorm(mid_channel),
            nn.PReLU(),
            nn.GRU(
                input_size=mid_channel,
                hidden_size=128,
                num_layers=2,
                batch_first=True,
            ),
        )

        self.vad_post = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        x: B,T
        HL: B,6
        """
        xk = self.stft.transform(x)
        xk_mag = xk.pow(2).sum(1, keepdim=True).sqrt()  # B,1,T,F

        xk = self.encoder(xk)

        for l in self.conformer:
            xk, _ = l(xk, causal=True)  # b,c,t,f

        vad_pred, _ = self.vad_predictor(xk)
        vad = self.vad_post(vad_pred)

        mask = self.mask_decoder(xk)
        spec = self.complex_decoder(xk)
        r, i = spec.chunk(2, dim=1)  # b,1,t,f
        phase = torch.atan2(i, r)

        xk_mag_est = xk_mag * mask
        spec_r = xk_mag_est * torch.cos(phase)
        spec_i = xk_mag_est * torch.sin(phase)

        xk_est = torch.concat([spec_r, spec_i], dim=1)

        x = self.stft.inverse(xk_est)

        # return x, spec
        return x, vad


@tables.register("models", "baseline_fig6_vad")
class BaselineVAD(nn.Module):
    r"""In-context method"""

    def __init__(
        self, nframe: int, nhop: int, mid_channel: int = 64, conformer_num=4, fs=16000
    ) -> None:
        super().__init__()

        self.stft = STFT(nframe, nhop, nframe)
        self.reso = fs / nframe
        nbin = nframe // 2 + 1
        assert 250 % self.reso == 0

        self.encoder = DenseEncoder(in_channel=3, channels=mid_channel)
        self.ht_freq = torch.ceil(
            torch.tensor([0, 250, 500, 1000, 2000, 4000, 8001]) / self.reso
        ).int()

        self.conformer = nn.ModuleList([FTConformer(dim=mid_channel) for _ in range(conformer_num)])

        self.mask_decoder = MaskDecoder(num_features=nbin, num_channel=mid_channel, out_channel=1)
        self.complex_decoder = ComplexDecoder(num_channel=mid_channel)

        self.vad_predictor = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1, 65), stride=(1, 65)),  # B,C,T,1
            nn.Conv2d(
                in_channels=mid_channel,
                out_channels=mid_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ),  # b,c,t,1
            Rearrange("b c t ()->b t c"),
            nn.LayerNorm(mid_channel),
            nn.PReLU(),
            nn.GRU(
                input_size=mid_channel,
                hidden_size=128,
                num_layers=2,
                batch_first=True,
            ),
        )

        self.vad_post = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, HL):
        """
        x: B,T
        HL: B,6
        """
        xk = self.stft.transform(x)
        xk_mag = xk.pow(2).sum(1, keepdim=True).sqrt()  # B,1,T,F
        # xk_bands_pow = compute_subbands_energy(xk, self.ht_freq)
        hl = expand_HT(HL, xk.shape[-2], self.reso)  # B,C(1),T,F
        xk = torch.cat([xk, hl], dim=1)

        xk = self.encoder(xk)

        for l in self.conformer:
            xk, _ = l(xk, causal=True)  # b,c,t,f

        vad_pred, _ = self.vad_predictor(xk)
        vad = self.vad_post(vad_pred)

        mask = self.mask_decoder(xk)
        spec = self.complex_decoder(xk)
        r, i = spec.chunk(2, dim=1)  # b,1,t,f
        phase = torch.atan2(i, r)

        xk_mag_est = xk_mag * mask
        spec_r = xk_mag_est * torch.cos(phase)
        spec_i = xk_mag_est * torch.sin(phase)

        xk_est = torch.concat([spec_r, spec_i], dim=1)

        x = self.stft.inverse(xk_est)

        # return x, spec
        return x, vad


@tables.register("models", "baseline_fig6_linear")
class BaselineLinear(nn.Module):
    r"""In-context method"""

    def __init__(
        self, nframe: int, nhop: int, mid_channel: int = 64, conformer_num=4, fs=16000
    ) -> None:
        super().__init__()

        self.stft = STFT(nframe, nhop, nframe)
        self.reso = fs / nframe
        nbin = nframe // 2 + 1
        assert 250 % self.reso == 0
        self.freqs = torch.linspace(0, fs // 2, nbin)
        self.hl_process = HLModule(nbin=nbin, HL_freq_extend=self.freqs)

        self.encoder = DenseEncoder(in_channel=3, channels=mid_channel)
        self.ht_freq = torch.ceil(
            torch.tensor([0, 250, 500, 1000, 2000, 4000, 8001]) / self.reso
        ).int()

        self.conformer = nn.ModuleList([FTConformer(dim=mid_channel) for _ in range(conformer_num)])

        self.mask_decoder = MaskDecoder(num_features=nbin, num_channel=mid_channel, out_channel=1)
        self.complex_decoder = ComplexDecoder(num_channel=mid_channel)

    def forward(self, x, HL):
        """
        x: B,T
        HL: B,6
        """
        xk = self.stft.transform(x)
        xk_mag = xk.pow(2).sum(1, keepdim=True).sqrt()  # B,1,T,F
        # xk_bands_pow = compute_subbands_energy(xk, self.ht_freq)
        # hl = expand_HT(HL, xk.shape[-2], self.reso)  # B,C(1),T,F
        hl = self.hl_process.extend_with_linear(HL, T=xk.shape[-2])
        xk = torch.cat([xk, hl], dim=1)

        xk = self.encoder(xk)

        for l in self.conformer:
            xk, _ = l(xk, causal=True)

        mask = self.mask_decoder(xk)
        spec = self.complex_decoder(xk)
        r, i = spec.chunk(2, dim=1)  # b,1,t,f
        phase = torch.atan2(i, r)

        xk_mag_est = xk_mag * mask
        spec_r = xk_mag_est * torch.cos(phase)
        spec_i = xk_mag_est * torch.sin(phase)

        xk_est = torch.concat([spec_r, spec_i], dim=1)

        x = self.stft.inverse(xk_est)

        # return x, spec
        return x


class HLModule(nn.Module):
    def __init__(
        self,
        nbin=257,
        fs=16000,
        HL_freq=[250, 500, 1000, 2000, 4000, 8000],
        # fmt: off
        HL_freq_extend=torch.tensor([250, 375, 500, 625, 750, 1000, 1125, 1375,
                        1750, 2125, 2625, 3125, 3875, 4625, 5500, 6625]),
        freq_bands_range=[0, 250, 375, 500, 625, 750, 1000, 1250, 1625,
                          2000, 2375, 2875, 3500, 4250, 5125, 6125, 8001]
        # fmt: on
    ) -> None:
        super().__init__()

        self.freqs = torch.linspace(0, fs // 2, nbin)  # nbin, (fs//2) / (nbin)
        self.reso = fs // 2 / (nbin - 1)

        # sub-bands
        bands_filter = self._rectangular_filters(self.freqs, freq_bands_range)
        # hl_freq_ext = self.freqs if full else torch.tensor(HL_freq_extend)

        # HL curve index
        HL_curve, delta_x = self._HL_curve_idx(HL_freq=[0, *HL_freq], HL_freq_extend=HL_freq_extend)
        self.register_buffer("hl_freq", torch.tensor([0, *HL_freq]).float())

        self.register_buffer("bands_filter", bands_filter)
        self.register_buffer("HL_curve", HL_curve)
        self.register_buffer("delta_x", delta_x)

    @staticmethod
    # def _HL_curve(HL_freq, HL_freq_extend):
    #     """extend the Hearing Loss threshold to other freqency point.

    #     :param HL_freq: 7
    #     :param HL_freq_extend: 16
    #     :returns: 16; 16

    #     """
    #     hl_freq = torch.tensor(HL_freq).float()

    #     curve_idx = [(i, (x >= hl_freq).sum() - 1) for i, x in enumerate(HL_freq_extend)]
    #     HL_freq_extend_curve = torch.zeros((len(HL_freq_extend), len(HL_freq) - 1))
    #     for idx in curve_idx:
    #         HL_freq_extend_curve[idx] = 1.0
    #     HL_freq_extend_curve = HL_freq_extend_curve.permute(1, 0)  # hl,nbands

    #     delta_x = torch.tensor([x - hl_freq[idx] for x, (_, idx) in zip(HL_freq_extend, curve_idx)])
    #     return HL_freq_extend_curve, delta_x

    @staticmethod
    def _HL_curve_idx(HL_freq, HL_freq_extend):
        """extend the Hearing Loss threshold to other freqency point.

        :param HL_freq: 7,
        :param HL_freq_extend: N,
        :returns: N,; N,

        """
        hl_freq = torch.tensor(HL_freq).float()

        curve_idx = torch.tensor([(x >= hl_freq).sum() - 1 for x in HL_freq_extend])
        delta_x = torch.tensor([x - hl_freq[idx] for x, idx in zip(HL_freq_extend, curve_idx)])
        return curve_idx, delta_x

    def _rectangular_filters(self, all_freqs, bands_range):
        nbands = len(bands_range) - 1
        bands_filter = torch.zeros((nbands, len(all_freqs)))
        bands_idx = [
            (i, ((all_freqs >= low) & (all_freqs < high)).nonzero(as_tuple=True)[0])
            for i, (low, high) in enumerate(zip(bands_range[:-1], bands_range[1:]))
        ]
        for idx in bands_idx:
            bands_filter[idx] = 1

        return bands_filter.permute(1, 0)  # nbin, nbands

    def _HL_LinearFitting(self, HL):
        diff_x = torch.diff(self.hl_freq)
        diff_y = torch.diff(HL)
        # print(HL.shape, diff_x.shape, "@", diff_y.shape)

        k = diff_y / diff_x  # B,nbands
        k = torch.concat([k, k.new_zeros(k.size(0), 1)], dim=-1)
        b = HL

        return k, b

    def extend_with_value(self, hl, T=None):
        """extend with self value

        :param hl: B,8
        :returns: B,1,T,nbin

        """
        m = int(250 / self.reso)
        bandarray = torch.tensor([0] + [(2**i) * m for i in range(hl.shape[1])]).to(hl.device)
        T = T or 1

        repeat_n = bandarray[1:] - bandarray[:-1]
        repeat_n[0] += 1

        expand_ht = hl.repeat_interleave(repeat_n, dim=-1).unsqueeze(1).unsqueeze(1)  # B,1,1,nbin
        expand_ht = expand_ht.repeat(1, 1, T, 1) / 100.0

        return expand_ht

    def extend_with_linear(self, hl, T=None) -> Tensor:
        """extend with self value

        :param hl: B,6
        :returns: B,1,T,nbin

        """
        T = T or 1
        hl = torch.concat([hl[:, (0,)], hl], dim=-1)

        k, b = self._HL_LinearFitting(hl)  # b,nbands(16)
        hl_ext = k[:, self.HL_curve] * self.delta_x + b[:, self.HL_curve]
        # print(k.shape, b.shape, hl_ext.shape, self.HL_curve.shape)

        expand_ht = hl_ext.unsqueeze(1).unsqueeze(1).repeat(1, 1, T, 1)  # b,1,nbands

        return expand_ht / 100

    # def forward(self, xk: Tensor, hl):
    #     """
    #     x: B,2,T,F
    #     return: B,T,mid_channel
    #     """
    #     nT = xk.size(-2)
    #     hl_ext = self._HL_LinearFitting(hl) / 100  # b,nbands(16)
    #     x_hl = hl_ext.unsqueeze(1).repeat(1, nT, 1)  # b,1,nbands
    #     xk_pow = xk.pow(2).sum(1)  # b,t,f
    #     xk_bands = (xk_pow @ self.bands_filter).pow(0.3)  # b,t,nbands

    #     return xk_bands, x_hl


class CodeBook(nn.Module):
    def __init__(
        self, num_cb=256, dim_cb=65, group=2, mid_channel: int = 8, dim_inp: int = 16
    ) -> None:
        super().__init__()
        self.group = group
        self.num_cb = num_cb
        self.dim_cb = dim_cb

        self.codebook = nn.Embedding(num_cb * group, dim_cb)
        self.codebook.weight.data.uniform_(-1.0 / (num_cb * group), 1.0 / (num_cb * group))

        self.encode = nn.Sequential(
            nn.Conv2d(2, mid_channel, (1, 1), (1, 1)),
            nn.BatchNorm2d(mid_channel),
            nn.PReLU(mid_channel),
            nn.Conv2d(mid_channel, mid_channel * 2, (1, 1), (1, 1)),
            GLU(1),
            LayerNorm(mid_channel),
            nn.PReLU(),
        )

    def forward(self, xk, hl):
        """
        x, hl: b,t,n; n for subbands
        """
        x = torch.stack([xk, hl], dim=1)
        x = self.encode(x)  # b,t,D
        dist = self.compute_distance(x)

        idx = torch.argmin(dist, -1)  # b,t
        x_cb = self.codebook(idx)

        # stop gradient
        x_out = x + (x_cb - x).detach()

        return x_out, x, x_cb

    def compute_distance(self, x):
        """compute L2 distance between the vector x and embeddings.

        :param x: b,t,n
        :returns:

        """

        x_cb = self.codebook.weight  # GN,D
        x_cb = x_cb[None, None, ...]
        x = x.unsqueeze(2)  # b,t,1,D

        # b,t,1,D - 1,1,N,D
        dist = torch.sum((x - x_cb) ** 2, -1)
        return dist


class FactorizedAttn_(nn.Module):
    """Factorized attention
    Input:
        - xk, contenxt embeddings, with shape b,c,t,f.
        - x_cb, used to calculate the weights, with shape b,group,t,f.
    Argus:
        - ndim, `C` of inputs;

    Output: B,C,T,F
    """

    def __init__(self, ndim: int, ndim_w: int, nhead: int = 4) -> None:
        super().__init__()

        self.ctxToHeads = nn.Sequential(
            Rearrange("b c t f->(b t) f c"),
            nn.Linear(ndim, ndim * nhead),
            nn.PReLU(),
            Rearrange("b f (c n)->b f c n", n=nhead),
        )
        self.wToHeads = nn.Sequential(
            Rearrange("b c t f->(b t) f c"),
            nn.Linear(ndim_w, nhead),
            nn.PReLU(),
        )

    def forward(self, xk, cb):
        nB = xk.size(0)
        ctx = self.ctxToHeads(xk)
        w = self.wToHeads(cb)
        refined = torch.einsum("bfcn,bfn->bfc", ctx, w)
        refined = einops.rearrange(refined, "(b t) f c->b c t f", b=nB)
        return refined.tanh()


class FactorizedAttn(nn.Module):
    """Factorized attention
    Input:
        - xk, contenxt embeddings, with shape b,c,t,f.
        - x_cb, used to calculate the weights, with shape b,group,t,f.
    Argus:
        - ndim, `C` of inputs;

    Output: B,C,T,F
    """

    def __init__(self, ndim: int, group: int, nhead: int = 4) -> None:
        super().__init__()

        self.ctxToHeads = nn.Sequential(
            nn.Conv2d(ndim, ndim, (1, 1), (1, 1)),
            Rearrange("b c t f->(b f) t c"),
            nn.LayerNorm(ndim),
            nn.PReLU(),
            nn.Linear(ndim, ndim * nhead),
            Rearrange("b t (c n)->b t c n", n=nhead),
        )
        self.wToHeads = nn.Sequential(
            nn.Conv2d(group, group, (1, 1), (1, 1)),
            Rearrange("b c t f->(b f) t c"),
            nn.LayerNorm(group),
            nn.PReLU(),
            nn.Linear(group, nhead),
            nn.Softmax(-1),  # b,t,n
        )

    def forward(self, xk, cb):
        nB = xk.size(0)
        ctx = self.ctxToHeads(xk)
        w = self.wToHeads(cb)
        refined = torch.einsum("btcn,btn->btc", ctx, w)
        refined = einops.rearrange(refined, "(b f) t c->b c t f", b=nB)
        return refined.tanh()


@tables.register("models", "baseline_wHLCodec")
class BaselineHLCodec(nn.Module):
    def __init__(
        self, nframe: int, nhop: int, mid_channel: int = 64, conformer_num=4, fs=16000
    ) -> None:
        super().__init__()

        self.stft = STFT(nframe, nhop, nframe)
        self.reso = fs / nframe
        nbin = nframe // 2 + 1
        assert 250 % self.reso == 0

        self.preprocess = HLModule(nbin)

        self.encoder = DenseEncoder(in_channel=2, channels=mid_channel)

        self.conformer = nn.ModuleList([FTConformer(dim=mid_channel) for _ in range(conformer_num)])

        self.mask_decoder = MaskDecoder(
            num_features=nbin, num_channel=mid_channel + 2, out_channel=1
        )
        self.complex_decoder = ComplexDecoder(num_channel=mid_channel)

        self.cbook = nn.ModuleList(
            [CodeBook(num_cb=64, dim_cb=65, mid_channel=4, dim_inp=16) for _ in range(2)]
        )
        # self.factAttn = FactorizedAttn(mid_channel, 65, 8)

    def forward(self, x, HL):
        """
        x: B,T
        HL: B,6
        """
        xk = self.stft.transform(x)
        xk_mag = xk.pow(2).sum(1, keepdim=True).sqrt()  # B,1,T,F
        # xk_bands_pow = compute_subbands_energy(xk, self.ht_freq)

        # xk_b: b,t,16; hl_b: b,t,16
        xk_b, hl_b = self.preprocess(xk, HL)

        x_cbout_l, x_cbi_l, x_cb_l = [], [], []
        for l in self.cbook:
            x_, x_cbi_, x_cb_ = l(xk_b, hl_b)  # b,t,f
            x_cbout_l.append(x_)
            x_cbi_l.append(x_cbi_)
            x_cb_l.append(x_cb_)

        x_cbout = torch.stack(x_cbout_l, dim=1)
        x_cbi = torch.stack(x_cbi_l, dim=1)
        x_cb = torch.stack(x_cb_l, dim=1)

        xk = self.encoder(xk)

        for l in self.conformer:
            xk, _ = l(xk, causal=True)

        # x_fact = self.factAttn(xk, x_hl)
        # xk: b,c,t,f; x_hl: b,c,t,1
        x_dec = torch.concat([xk, x_cbout], dim=1)
        mask = self.mask_decoder(x_dec)
        spec = self.complex_decoder(xk)
        r, i = spec.chunk(2, dim=1)  # b,1,t,f
        phase = torch.atan2(i, r)

        xk_mag_est = xk_mag * mask
        spec_r = xk_mag_est * torch.cos(phase)
        spec_i = xk_mag_est * torch.sin(phase)

        xk_est = torch.concat([spec_r, spec_i], dim=1)

        x = self.stft.inverse(xk_est)

        return x, x_cbi, x_cb


@tables.register("models", "baseline_wGumbelCodebook")
class BaselineGumbelCodebook(nn.Module):
    def __init__(
        self, nframe: int, nhop: int, mid_channel: int = 64, conformer_num=4, fs=16000
    ) -> None:
        super().__init__()

        self.stft = STFT(nframe, nhop, nframe)
        self.reso = fs / nframe
        nbin = nframe // 2 + 1
        assert 250 % self.reso == 0
        self.freqs = torch.linspace(0, fs // 2, nbin)

        self.preprocess = HLModule(nbin, HL_freq_extend=self.freqs)

        self.encoder = DenseEncoder(in_channel=2, channels=mid_channel)

        # self.conformer = nn.ModuleList([FTConformer(dim=mid_channel) for _ in range(conformer_num)])

        self.mask_decoder = MaskDecoder(num_features=nbin, num_channel=mid_channel, out_channel=1)
        self.complex_decoder = ComplexDecoder(num_channel=mid_channel)

        # self.cbook = nn.ModuleList(
        #     [CodeBook(num_cb=64, dim_cb=65, mid_channel=4, dim_inp=16) for _ in range(2)]
        # )
        nch_cb = 8
        self.cbook_pre = nn.Sequential(
            nn.Conv2d(1, nch_cb, (1, 1), (1, 1), groups=1),
            nn.BatchNorm2d(nch_cb),
            nn.PReLU(),
            # nn.ConstantPad2d((1, 1, 0, 0), 0.0),
            nn.Conv2d(nch_cb, nch_cb * 2, (1, 1), (1, 1)),
            # nn.BatchNorm2d(nch_cb * 2),
            # nn.PReLU(),
            GLU(1),
            Rearrange("b c t n->b t (c n)"),
        )
        self.group = 16
        self.cb_vdim = 65
        self.cbook = GumbelVectorQuantizer(nch_cb * 16, 128, self.group, self.cb_vdim * self.group)
        self.factAttn = FactorizedAttn(mid_channel, self.group, 16)

        self.conformer = nn.ModuleList(
            [
                ConditionalFTConformer(dim=mid_channel, dim_cond=self.group)
                for _ in range(conformer_num)
            ]
        )
        # self.fact_post = nn.Sequential(
        #     nn.Conv2d(mid_channel, mid_channel, (1, 1), (1, 1)),
        #     nn.BatchNorm2d(mid_channel),
        #     nn.PReLU(),
        #     nn.Conv2d(mid_channel, mid_channel * 2, (1, 1), (1, 1)),
        #     GLU(1),
        # )

    def setup_num(self, num):
        """setup the params of GumbelBook

        :param num: epoch num
        :returns:

        """
        self.cbook.set_num_updates(num)

    def diversity_loss(self, output):
        l = self.cbook.diversity_loss(output)
        return l

    def forward(self, x, HL):
        """
        x: B,T
        HL: B,6
        """
        xk = self.stft.transform(x)
        xk_mag = xk.pow(2).sum(1, keepdim=True).sqrt()  # B,1,T,F
        # xk_bands_pow = compute_subbands_energy(xk, self.ht_freq)

        # xk_b: b,t,16; hl_b: b,t,16
        xk_b, hl_b = self.preprocess(xk, HL)
        # xk_b = torch.stack([xk_b, hl_b], dim=1)
        hl_b = hl_b[:, None, ...]

        xk_b = self.cbook_pre(hl_b)
        x_cb_dict = self.cbook(xk_b)
        x_cb = einops.rearrange(x_cb_dict["x"], "b t (c n)->b c t n", c=self.group)

        xk = self.encoder(xk)

        for l in self.conformer:
            xk, _ = l(xk, x_cb, causal=True)

        # x_fact = self.factAttn(xk, x_hl)
        # xk: b,c,t,f; x_hl: b,c,t,1
        # xk_ = self.factAttn(xk, x_cb)

        mask = self.mask_decoder(xk)
        spec = self.complex_decoder(xk)
        r, i = spec.chunk(2, dim=1)  # b,1,t,f
        phase = torch.atan2(i, r)

        xk_mag_est = xk_mag * mask
        spec_r = xk_mag_est * torch.cos(phase)
        spec_i = xk_mag_est * torch.sin(phase)

        xk_est = torch.concat([spec_r, spec_i], dim=1)

        x = self.stft.inverse(xk_est)

        return x, x_cb_dict


class HLCrossAttn(nn.Module):
    def __init__(self, ff_mult=4, heads=2) -> None:
        super().__init__()

        mch = 16
        ff1_hl = FeedForward(dim=mch, mult=ff_mult, dropout=0.2)
        ff1_xk = FeedForward(dim=mch, mult=ff_mult, dropout=0.2)
        self.ff1_hl = Scale(0.5, PreNorm(mch, ff1_hl))
        self.ff1_xk = Scale(0.5, PreNorm(mch, ff1_xk))
        self.attn_pre_norm_hl = nn.LayerNorm(mch)
        self.attn_pre_norm_xk = nn.LayerNorm(mch)

        # BT,16,mch; BT,257,mch
        self.attn_hl = nn.MultiheadAttention(
            embed_dim=mch, num_heads=heads, batch_first=True, dropout=0.2
        )

        # self.attn_xk = nn.MultiheadAttention(
        #     embed_dim=mch, num_heads=heads, kdim=257, vdim=257, batch_first=True, dropout=0.2
        # )

        ff2 = FeedForward(dim=16, mult=ff_mult, dropout=0.2)
        self.ff2 = Scale(0.5, PreNorm(16, ff2))
        self.post_norm = nn.Sequential(nn.LayerNorm(16))

    def forward(self, xk, hl):
        """
        xk: b,2,t,f
        hl: b,t,16
        """
        nB = xk.size(0)
        xk = self.pre_xk(xk)
        hl = self.pre_hl(hl)

        xk = self.attn_pre_norm_xk(self.ff1_xk(xk) + xk)  # BT,F,mch
        hl = self.attn_pre_norm_hl(self.ff1_hl(hl) + hl)  # BT,16,mch

        # BT,F,16
        x_mhsa, attn_w = self.attn_hl(xk, hl, hl, need_weights=True, average_attn_weights=False)
        # xk = xk.permute(0, 2, 1)
        # print(xk.shape, hl.shape)
        # x_mhsa, attn_w = self.attn_xk(hl, xk, xk, need_weights=True, average_attn_weights=False)
        # print(x_mhsa.shape, attn_w.shape, "@")
        #
        x = self.post_norm(self.ff2(x_mhsa) + x_mhsa)
        x = einops.rearrange(x, "(b t) f n->b n t f", b=nB)

        return x, attn_w


@tables.register("models", "condConformer")
class BaselineConditionalConformer(nn.Module):
    def __init__(
        self, nframe: int, nhop: int, mid_channel: int = 64, conformer_num=4, fs=16000
    ) -> None:
        super().__init__()

        self.stft = STFT(nframe, nhop, nframe)
        self.reso = fs / nframe
        nbin = nframe // 2 + 1
        assert 250 % self.reso == 0
        self.freqs = torch.linspace(0, fs // 2, nbin)  # []

        # self.preprocess = HLModule(nbin, HL_freq_extend=self.freqs)
        self.preprocess = HLModule(nbin, HL_freq_extend=self.freqs)
        self.group = mid_channel
        # self.cbook = GumbelVectorQuantizer(nbin, 128, self.group, 65 * self.group)

        self.mlp = nn.Sequential(
            nn.Linear(nbin, nbin * 4),
            Rearrange("b c t (f n)-> b (c n) t f", n=4),
            nn.GELU(approximate="tanh"),
            nn.Conv2d(4, 16, (1, 3), (1, 2), (0, 1)),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.Conv2d(16, 64, (1, 3), (1, 2), (0, 1)),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            Rearrange("b c t f-> b f t c"),
            nn.Linear(64, mid_channel),
        )
        # self.hl_attn = HLAttn()

        self.encoder = DenseEncoder(in_channel=2, channels=mid_channel)

        self.conformer = nn.ModuleList(
            [ConditionalFTConformer(dim=mid_channel) for _ in range(conformer_num)]
        )

        self.mask_decoder = MaskDecoder(num_features=nbin, num_channel=mid_channel, out_channel=1)
        self.complex_decoder = ComplexDecoder(num_channel=mid_channel)

        # self.cbook = nn.ModuleList(
        #     [CodeBook(num_cb=64, dim_cb=65, mid_channel=4, dim_inp=16) for _ in range(2)]
        # )
        # self.factAttn = FactorizedAttn(1, 16, 8)

    def forward(self, x, HL):
        """
        x: B,T
        HL: B,6
        """
        xk = self.stft.transform(x)
        xk_mag = xk.pow(2).sum(1, keepdim=True).sqrt()  # B,1,T,F
        # xk_bands_pow = compute_subbands_energy(xk, self.ht_freq)

        # xk_b: b,t,16; hl_b: b,t,16
        hl_b = self.preprocess.extend_with_linear(HL)  # b,1,1,f
        hl_b = self.mlp(hl_b)  # b,c,1,f
        # hl_b = hl_b.view(-1, hl_b.size(-1))  # b,mch
        hl_b = hl_b.squeeze(2)  # b,t,mch

        # b,16,t,f
        # xk_hl, _ = self.hl_attn(xk, hl_b)

        xk = self.encoder(xk)

        for l in self.conformer:
            xk, _ = l(xk, hl_b, causal=True)

        # x_fact = self.factAttn(xk, x_hl)
        # xk: b,c,t,f; x_hl: b,c,t,1
        mask = self.mask_decoder(xk)
        spec = self.complex_decoder(xk)
        r, i = spec.chunk(2, dim=1)  # b,1,t,f
        phase = torch.atan2(i, r)

        # mask = self.factAttn(mask, xk_hl)
        xk_mag_est = xk_mag * mask
        spec_r = xk_mag_est * torch.cos(phase)
        spec_i = xk_mag_est * torch.sin(phase)

        xk_est = torch.concat([spec_r, spec_i], dim=1)

        x = self.stft.inverse(xk_est)

        return x


@tables.register("models", "IterCondConformer")
class BaselineConditionalConformerIter(nn.Module):
    def __init__(
        self, nframe: int, nhop: int, mid_channel: int = 64, conformer_num=4, fs=16000
    ) -> None:
        super().__init__()

        self.stft = STFT(nframe, nhop, nframe)
        self.reso = fs / nframe
        nbin = nframe // 2 + 1
        assert 250 % self.reso == 0
        self.freqs = torch.linspace(0, fs // 2, nbin)  # []

        # self.preprocess = HLModule(nbin, HL_freq_extend=self.freqs)
        self.preprocess = HLModule(nbin, HL_freq_extend=self.freqs)
        self.group = mid_channel
        # self.cbook = GumbelVectorQuantizer(nbin, 128, self.group, 65 * self.group)

        self.mlp = nn.Sequential(
            nn.Linear(nbin, nbin * 4),
            Rearrange("b c t (f n)-> b (c n) t f", n=4),
            nn.GELU(approximate="tanh"),
            nn.Conv2d(4, 16, (1, 3), (1, 2), (0, 1)),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.Conv2d(16, 64, (1, 3), (1, 2), (0, 1)),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            Rearrange("b c t f-> b f t c"),
            nn.Linear(64, mid_channel),
        )
        # self.hl_attn = HLAttn()

        self.encoder = DenseEncoder(in_channel=2, channels=mid_channel)

        self.conformer = nn.ModuleList(
            [ConditionalFTConformerIter(dim=mid_channel) for _ in range(conformer_num)]
        )

        self.mask_decoder = MaskDecoder(num_features=nbin, num_channel=mid_channel, out_channel=1)
        self.complex_decoder = ComplexDecoder(num_channel=mid_channel)

        # self.cbook = nn.ModuleList(
        #     [CodeBook(num_cb=64, dim_cb=65, mid_channel=4, dim_inp=16) for _ in range(2)]
        # )
        # self.factAttn = FactorizedAttn(1, 16, 8)

    def forward(self, x, HL):
        """
        x: B,T
        HL: B,6
        """
        xk = self.stft.transform(x)
        xk_mag = xk.pow(2).sum(1, keepdim=True).sqrt()  # B,1,T,F
        # xk_bands_pow = compute_subbands_energy(xk, self.ht_freq)

        # xk_b: b,t,16; hl_b: b,t,16
        hl_b = self.preprocess.extend_with_linear(HL)  # b,1,1,f
        hl_b = self.mlp(hl_b)  # b,c,1,f
        # hl_b = hl_b.view(-1, hl_b.size(-1))  # b,mch
        hl_b = hl_b.squeeze(2)  # b,t,mch

        # b,16,t,f
        # xk_hl, _ = self.hl_attn(xk, hl_b)

        xk = self.encoder(xk)

        cond = hl_b
        for l in self.conformer:
            xk, cond, _ = l(xk, cond, causal=True)

        # x_fact = self.factAttn(xk, x_hl)
        # xk: b,c,t,f; x_hl: b,c,t,1
        mask = self.mask_decoder(xk)
        spec = self.complex_decoder(xk)
        r, i = spec.chunk(2, dim=1)  # b,1,t,f
        phase = torch.atan2(i, r)

        # mask = self.factAttn(mask, xk_hl)
        xk_mag_est = xk_mag * mask
        spec_r = xk_mag_est * torch.cos(phase)
        spec_i = xk_mag_est * torch.sin(phase)

        xk_est = torch.concat([spec_r, spec_i], dim=1)

        x = self.stft.inverse(xk_est)

        return x


@tables.register("models", "condConformerVAD")
class BaselineConditionalConformerVAD4(nn.Module):
    def __init__(
        self, nframe: int, nhop: int, mid_channel: int = 64, conformer_num=4, fs=16000
    ) -> None:
        super().__init__()

        self.stft = STFT(nframe, nhop, nframe)
        self.reso = fs / nframe
        nbin = nframe // 2 + 1
        assert 250 % self.reso == 0
        self.freqs = torch.linspace(0, fs // 2, nbin)  # []

        # self.preprocess = HLModule(nbin, HL_freq_extend=self.freqs)
        self.preprocess = HLModule(nbin, HL_freq_extend=self.freqs)
        self.group = mid_channel
        # self.cbook = GumbelVectorQuantizer(nbin, 128, self.group, 65 * self.group)

        self.mlp = nn.Sequential(
            nn.Linear(nbin, nbin * 4),
            Rearrange("b c t (f n)-> b (c n) t f", n=4),
            nn.GELU(approximate="tanh"),
            nn.Conv2d(4, 16, (1, 3), (1, 2), (0, 1)),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.Conv2d(16, 64, (1, 3), (1, 2), (0, 1)),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            Rearrange("b c t f-> b f t c"),
            nn.Linear(64, mid_channel),
        )
        # self.hl_attn = HLAttn()

        self.encoder = DenseEncoder(in_channel=2, channels=mid_channel)

        self.conformer = nn.ModuleList(
            [ConditionalFTConformer(dim=mid_channel) for _ in range(conformer_num)]
        )

        self.mask_decoder = MaskDecoder(num_features=nbin, num_channel=mid_channel, out_channel=1)
        self.complex_decoder = ComplexDecoder(num_channel=mid_channel)

        # self.cbook = nn.ModuleList(
        #     [CodeBook(num_cb=64, dim_cb=65, mid_channel=4, dim_inp=16) for _ in range(2)]
        # )
        # self.factAttn = FactorizedAttn(1, 16, 8)
        self.vad_predictor = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1, 65), stride=(1, 65)),  # B,C,T,1
            nn.Conv2d(
                in_channels=mid_channel,
                out_channels=mid_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ),  # b,c,t,1
            Rearrange("b c t ()->b t c"),
            nn.LayerNorm(mid_channel),
            nn.PReLU(),
            nn.GRU(
                input_size=mid_channel,
                hidden_size=128,
                num_layers=2,
                batch_first=True,
            ),
        )

        self.vad_post = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, HL):
        """
        x: B,T
        HL: B,6
        """
        xk = self.stft.transform(x)
        xk_mag = xk.pow(2).sum(1, keepdim=True).sqrt()  # B,1,T,F
        # xk_bands_pow = compute_subbands_energy(xk, self.ht_freq)

        # xk_b: b,t,16; hl_b: b,t,16
        hl_b = self.preprocess.extend_with_linear(HL)  # b,1,1,f
        hl_b = self.mlp(hl_b)  # b,1,1,f
        # hl_b = hl_b.view(-1, hl_b.size(-1))  # b,mch
        hl_b = hl_b.squeeze(2)

        # b,16,t,f
        # xk_hl, _ = self.hl_attn(xk, hl_b)

        xk = self.encoder(xk)

        for l in self.conformer:
            xk, _ = l(xk, hl_b, causal=True)

        vad_pred, _ = self.vad_predictor(xk)
        vad = self.vad_post(vad_pred)

        # x_fact = self.factAttn(xk, x_hl)
        # xk: b,c,t,f; x_hl: b,c,t,1
        mask = self.mask_decoder(xk)
        spec = self.complex_decoder(xk)
        r, i = spec.chunk(2, dim=1)  # b,1,t,f
        phase = torch.atan2(i, r)

        # mask = self.factAttn(mask, xk_hl)
        xk_mag_est = xk_mag * mask
        spec_r = xk_mag_est * torch.cos(phase)
        spec_i = xk_mag_est * torch.sin(phase)

        xk_est = torch.concat([spec_r, spec_i], dim=1)

        x = self.stft.inverse(xk_est)

        return x, vad


@tables.register("models", "condConformerVAD8")
class BaselineConditionalConformerVAD8(nn.Module):
    def __init__(
        self, nframe: int, nhop: int, mid_channel: int = 64, conformer_num=4, fs=16000
    ) -> None:
        super().__init__()

        self.stft = STFT(nframe, nhop, nframe)
        self.reso = fs / nframe
        nbin = nframe // 2 + 1
        assert 250 % self.reso == 0
        self.freqs = torch.linspace(0, fs // 2, nbin)  # []

        # self.preprocess = HLModule(nbin, HL_freq_extend=self.freqs)
        self.preprocess = HLModule(nbin, HL_freq_extend=self.freqs)
        self.group = mid_channel
        # self.cbook = GumbelVectorQuantizer(nbin, 128, self.group, 65 * self.group)

        self.mlp = nn.Sequential(
            nn.Linear(nbin, nbin * 4),
            Rearrange("b c t (f n)-> b (c n) t f", n=4),
            nn.GELU(approximate="tanh"),
            nn.Conv2d(4, 16, (1, 3), (1, 2), (0, 1)),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.Conv2d(16, 32, (1, 3), (1, 2), (0, 1)),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(32, 64, (1, 3), (1, 2), (0, 1)),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            Rearrange("b c t f-> b f t c"),
            nn.Linear(64, mid_channel),
        )

        self.encoder = DenseEncoder8(in_channel=2, channels=mid_channel)

        self.conformer = nn.ModuleList(
            [ConditionalFTConformer(dim=mid_channel) for _ in range(conformer_num)]
        )

        self.mask_decoder = MaskDecoder8(num_features=nbin, num_channel=mid_channel, out_channel=1)
        self.complex_decoder = ComplexDecoder8(num_channel=mid_channel)

        # self.cbook = nn.ModuleList(
        #     [CodeBook(num_cb=64, dim_cb=65, mid_channel=4, dim_inp=16) for _ in range(2)]
        # )
        # self.factAttn = FactorizedAttn(1, 16, 8)
        self.vad_predictor = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1, 33), stride=(1, 33)),  # B,C,T,1
            nn.Conv2d(
                in_channels=mid_channel,
                out_channels=mid_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ),  # b,c,t,1
            Rearrange("b c t ()->b t c"),
            nn.LayerNorm(mid_channel),
            nn.PReLU(),
            nn.GRU(
                input_size=mid_channel,
                hidden_size=128,
                num_layers=2,
                batch_first=True,
            ),
        )

        self.vad_post = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, HL):
        """
        x: B,T
        HL: B,6
        """
        xk = self.stft.transform(x)
        xk_mag = xk.pow(2).sum(1, keepdim=True).sqrt()  # B,1,T,F
        # xk_bands_pow = compute_subbands_energy(xk, self.ht_freq)

        # xk_b: b,t,16; hl_b: b,t,16
        hl_b = self.preprocess.extend_with_linear(HL)  # b,1,1,f
        hl_b = self.mlp(hl_b)  # b,1,1,f
        # hl_b = hl_b.view(-1, hl_b.size(-1))  # b,mch
        hl_b = hl_b.squeeze(2)

        # b,16,t,f
        # xk_hl, _ = self.hl_attn(xk, hl_b)

        xk = self.encoder(xk)

        for l in self.conformer:
            xk, _ = l(xk, hl_b, causal=True)

        vad_pred, _ = self.vad_predictor(xk)
        vad = self.vad_post(vad_pred)

        # x_fact = self.factAttn(xk, x_hl)
        # xk: b,c,t,f; x_hl: b,c,t,1
        mask = self.mask_decoder(xk)
        spec = self.complex_decoder(xk)
        r, i = spec.chunk(2, dim=1)  # b,1,t,f
        phase = torch.atan2(i, r)

        # mask = self.factAttn(mask, xk_hl)
        xk_mag_est = xk_mag * mask
        spec_r = xk_mag_est * torch.cos(phase)
        spec_i = xk_mag_est * torch.sin(phase)

        xk_est = torch.concat([spec_r, spec_i], dim=1)

        x = self.stft.inverse(xk_est)

        return x, vad


@tables.register("models", "xkcConformer")
class BaselineXkConditionalConformer(nn.Module):
    """condition is xk and hl"""

    def __init__(
        self, nframe: int, nhop: int, mid_channel: int = 64, conformer_num=4, fs=16000
    ) -> None:
        super().__init__()

        self.stft = STFT(nframe, nhop, nframe)
        self.reso = fs / nframe
        nbin = nframe // 2 + 1
        assert 250 % self.reso == 0
        self.freqs = torch.linspace(0, fs // 2, nbin)  # []

        # self.preprocess = HLModule(nbin, HL_freq_extend=self.freqs)
        self.preprocess = HLModule(nbin, HL_freq_extend=self.freqs)
        self.group = mid_channel
        # self.cbook = GumbelVectorQuantizer(nbin, 128, self.group, 65 * self.group)

        self.mlp = nn.Sequential(
            nn.Linear(nbin, nbin * 4),
            Rearrange("b c t (f n)-> b (c n) t f", n=4),
            nn.GELU(approximate="tanh"),
            nn.Conv2d(4, 16, (1, 3), (1, 2), (0, 1)),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.Conv2d(16, 64, (1, 3), (1, 2), (0, 1)),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            Rearrange("b c t f-> b f t c"),
            nn.Linear(64, mid_channel),
        )
        # self.hl_attn = HLAttn()
        self.attention = nn.MultiheadAttention(mid_channel, 8, dropout=0.2)

        self.encoder = DenseEncoder(in_channel=2, channels=mid_channel)

        self.conformer = nn.ModuleList(
            [ConditionalFTConformer_(dim=mid_channel) for _ in range(conformer_num)]
        )

        self.mask_decoder = MaskDecoder(num_features=nbin, num_channel=mid_channel, out_channel=1)
        self.complex_decoder = ComplexDecoder(num_channel=mid_channel)

        # self.cbook = nn.ModuleList(
        #     [CodeBook(num_cb=64, dim_cb=65, mid_channel=4, dim_inp=16) for _ in range(2)]
        # )
        # self.factAttn = FactorizedAttn(1, 16, 8)

    def forward(self, x, HL):
        """
        x: B,T
        HL: B,6
        """
        xk = self.stft.transform(x)
        xk_mag = xk.pow(2).sum(1, keepdim=True).sqrt()  # B,1,T,F
        # xk_bands_pow = compute_subbands_energy(xk, self.ht_freq)

        # xk_b: b,t,16; hl_b: b,t,16
        hl_b = self.preprocess.extend_with_linear(HL)  # b,1,1,f
        hl_b = self.mlp(hl_b)  # b,1,1,f
        # hl_b = hl_b.view(-1, hl_b.size(-1))  # b,mch
        hl_b = hl_b.squeeze(2)  # b,t,c

        # b,16,t,f
        # xk_hl, _ = self.hl_attn(xk, hl_b)

        xk = self.encoder(xk)

        c = hl_b
        for l in self.conformer:
            xk, c, _ = l(xk, c, causal=True)

        # x_fact = self.factAttn(xk, x_hl)
        # xk: b,c,t,f; x_hl: b,c,t,1
        mask = self.mask_decoder(xk)
        spec = self.complex_decoder(xk)
        r, i = spec.chunk(2, dim=1)  # b,1,t,f
        phase = torch.atan2(i, r)

        # mask = self.factAttn(mask, xk_hl)
        xk_mag_est = xk_mag * mask
        spec_r = xk_mag_est * torch.cos(phase)
        spec_i = xk_mag_est * torch.sin(phase)

        xk_est = torch.concat([spec_r, spec_i], dim=1)

        x = self.stft.inverse(xk_est)

        return x


@tables.register("models", "DiTConformer")
class DiTConformer(nn.Module):
    def __init__(
        self, nframe: int, nhop: int, mid_channel: int = 64, conformer_num=4, fs=16000
    ) -> None:
        super().__init__()

        self.stft = STFT(nframe, nhop, nframe)
        self.reso = fs / nframe
        nbin = nframe // 2 + 1
        assert 250 % self.reso == 0
        self.freqs = torch.linspace(0, fs // 2, nbin)  # []

        # self.preprocess = HLModule(nbin, HL_freq_extend=self.freqs)
        self.preprocess = HLModule(nbin, HL_freq_extend=self.freqs)
        self.group = mid_channel
        # self.cbook = GumbelVectorQuantizer(nbin, 128, self.group, 65 * self.group)

        self.mlp = nn.Sequential(
            nn.Linear(nbin, nbin * 4),
            Rearrange("b c t (f n)-> b (c n) t f", n=4),
            nn.GELU(approximate="tanh"),
            nn.Conv2d(4, 16, (1, 3), (1, 2), (0, 1)),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.Conv2d(16, 64, (1, 3), (1, 2), (0, 1)),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            Rearrange("b c t f-> b f t c"),
            nn.Linear(64, mid_channel),
        )
        # self.hl_attn = HLAttn()

        self.encoder = DenseEncoder(in_channel=2, channels=mid_channel)

        self.conformer = nn.ModuleList(
            [FTDiTConformer(dim=mid_channel) for _ in range(conformer_num)]
        )

        self.mask_decoder = MaskDecoder(num_features=nbin, num_channel=mid_channel, out_channel=1)
        self.complex_decoder = ComplexDecoder(num_channel=mid_channel)

        # self.cbook = nn.ModuleList(
        #     [CodeBook(num_cb=64, dim_cb=65, mid_channel=4, dim_inp=16) for _ in range(2)]
        # )
        # self.factAttn = FactorizedAttn(1, 16, 8)

    def forward(self, x, HL):
        """
        x: B,T
        HL: B,6
        """
        xk = self.stft.transform(x)
        xk_mag = xk.pow(2).sum(1, keepdim=True).sqrt()  # B,1,T,F
        # xk_bands_pow = compute_subbands_energy(xk, self.ht_freq)

        # xk_b: b,t,16; hl_b: b,t,16
        hl_b = self.preprocess.extend_with_linear(HL)  # b,1,1,f
        hl_b = self.mlp(hl_b)  # b,1,1,f
        # hl_b = hl_b.view(-1, hl_b.size(-1))  # b,mch
        hl_b = hl_b.squeeze(2)

        # b,16,t,f
        # xk_hl, _ = self.hl_attn(xk, hl_b)

        xk = self.encoder(xk)

        for l in self.conformer:
            xk, _ = l(xk, hl_b, causal=True)

        # x_fact = self.factAttn(xk, x_hl)
        # xk: b,c,t,f; x_hl: b,c,t,1
        mask = self.mask_decoder(xk)
        spec = self.complex_decoder(xk)
        r, i = spec.chunk(2, dim=1)  # b,1,t,f
        phase = torch.atan2(i, r)

        # mask = self.factAttn(mask, xk_hl)
        xk_mag_est = xk_mag * mask
        spec_r = xk_mag_est * torch.cos(phase)
        spec_i = xk_mag_est * torch.sin(phase)

        xk_est = torch.concat([spec_r, spec_i], dim=1)

        x = self.stft.inverse(xk_est)

        return x


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    from utils.check_flops import check_flops

    inp = torch.randn(1, 16000)
    # # hl = torch.randn(1, 6)
    freqs = torch.linspace(0, 8000, 257)
    # hln = HLModule(129, HL_freq_extend=freqs)
    # hln_2 = HLModule(129)
    # hl = torch.FloatTensor([[50, 60, 70, 75, 85, 95], [50, 60, 70, 75, 85, 95]])
    # hl = torch.FloatTensor([50, 60, 70, 75, 85, 95])[None]
    hl = torch.randn(1, 6)
    # out = hln.extend_with_value(hl)
    # out: Tensor = hln.extend_with_linear(hl)
    # out.squeeze_()
    # out_2: Tensor = hln_2.extend_with_linear(hl)
    # out_2.squeeze_()
    # plt.subplot(221)
    # plt.scatter([250, 500, 1000, 2000, 4000, 8000], hl.squeeze().numpy())
    # plt.scatter(
    #     [250, 375, 500, 625, 750, 1000, 1125, 1375, 1750, 2125, 2625, 3125, 3875, 4625, 5500, 6625],
    #     out_2.numpy() * 100,
    # )
    # plt.plot(freqs.numpy(), out.numpy() * 100)
    # plt.savefig("test.svg")

    # net = Baseline(512, 256, 48, 2)
    # net = BaselineLinear(512, 256, 48, 2)
    # net = BaselineVAD(512, 256, 48, 2)
    net = BaselineConditionalConformerVAD8(512, 256, 48, 2)
    # net = BaselineConditionalConformerIter(512, 256, 48, 2)
    # net = BaselineConditionalConformer(512, 256, 48, 2)
    # inp = torch.randn(2, 2, 10, 257)
    # l = HLModule()
    # l(inp, hl)

    # net = BaselineHLCodec(512, 256, 48, 2)
    # net = BaselineGumbelCodebook(512, 256, 48, 2)
    # net = BaselineXkConditionalConformer(512, 256, 48, 2)
    # net = BaselineConditionalConformer(512, 256, 48, 2)
    # net = DiTConformer(512, 256, 48, 2)
    # net = BaselineVAD(512, 256, 48, 2)
    out = net(inp, hl)
    # print(out.shape)

    check_flops(net, inp, hl)
