import glob
import math
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed
from pesq import pesq
from torchvision.ops.deform_conv import DeformConv2d

# from .dysample import DySample
from primeKNet.GPFCA import GPFCA
from utils.check_flops import check_flops

from models.conv_stft import STFT
from JointNSHModel import expand_HT


@dataclass
class h_dict:
    n_fft: int = 512
    hop_size: int = 256
    win_size: int = 512
    beta: float = 2.0
    dense_channel: int = 64


def phase_losses(phase_r, phase_g, h):
    dim_freq = h.n_fft // 2 + 1
    dim_time = phase_r.size(-1)

    gd_matrix = (
        torch.triu(torch.ones(dim_freq, dim_freq), diagonal=1)
        - torch.triu(torch.ones(dim_freq, dim_freq), diagonal=2)
        - torch.eye(dim_freq)
    ).to(phase_g.device)
    gd_r = torch.matmul(phase_r.permute(0, 2, 1), gd_matrix)
    gd_g = torch.matmul(phase_g.permute(0, 2, 1), gd_matrix)

    iaf_matrix = (
        torch.triu(torch.ones(dim_time, dim_time), diagonal=1)
        - torch.triu(torch.ones(dim_time, dim_time), diagonal=2)
        - torch.eye(dim_time)
    ).to(phase_g.device)
    iaf_r = torch.matmul(phase_r, iaf_matrix)
    iaf_g = torch.matmul(phase_g, iaf_matrix)

    ip_loss = torch.mean(anti_wrapping_function(phase_r - phase_g))
    gd_loss = torch.mean(anti_wrapping_function(gd_r - gd_g))
    iaf_loss = torch.mean(anti_wrapping_function(iaf_r - iaf_g))

    return ip_loss, gd_loss, iaf_loss


def anti_wrapping_function(x):
    return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)


def pesq_score(utts_r, utts_g, h):
    pesq_score = Parallel(n_jobs=30)(
        delayed(eval_pesq)(
            utts_r[i].squeeze().cpu().numpy(), utts_g[i].squeeze().cpu().numpy(), h.sampling_rate
        )
        for i in range(len(utts_r))
    )
    pesq_score = np.mean(pesq_score)

    return pesq_score


def eval_pesq(clean_utt, esti_utt, sr):
    try:
        pesq_score = pesq(sr, clean_utt, esti_utt)
    except:
        # error can happen due to silent period
        pesq_score = -1

    return pesq_score


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def get_padding_2d(kernel_size, dilation=(1, 1)):
    return (
        int((kernel_size[0] * dilation[0] - dilation[0]) / 2),
        int((kernel_size[1] * dilation[1] - dilation[1]) / 2),
    )


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + "????????")
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]


class LearnableSigmoid_1d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


class LearnableSigmoid_2d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


class DWConv2d_BN(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=1,
        stride=1,
        norm_layer=nn.BatchNorm2d,
        act_layer=nn.Hardswish,
        bn_weight_init=1,
        offset_clamp=(-1, 1),
    ):
        super().__init__()

        self.offset_clamp = offset_clamp
        self.offset_generator = nn.Sequential(
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=in_ch,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                groups=in_ch,
            ),
            nn.Conv2d(
                in_channels=in_ch, out_channels=18, kernel_size=1, stride=1, padding=0, bias=False
            ),
        )
        self.dcn = DeformConv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=in_ch,
        )
        self.pwconv = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.act = act_layer() if act_layer is not None else nn.Identity()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        offset = self.offset_generator(x)

        if self.offset_clamp:
            offset = torch.clamp(offset, min=self.offset_clamp[0], max=self.offset_clamp[1])
        x = self.dcn(x, offset)

        x = self.pwconv(x)
        x = self.act(x)
        return x


class Deform_Embedding(nn.Module):
    def __init__(
        self,
        in_chans=64,
        embed_dim=64,
        patch_size=3,
        stride=1,
        act_layer=nn.Hardswish,
        offset_clamp=(-1, 1),
    ):
        super().__init__()

        self.patch_conv = DWConv2d_BN(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            act_layer=act_layer,
            offset_clamp=offset_clamp,
        )

    def forward(self, x):
        """foward function"""
        x = self.patch_conv(x)

        return x


class DS_DDB(nn.Module):
    def __init__(self, h, kernel_size=(3, 3), depth=4):
        super(DS_DDB, self).__init__()
        self.h = h
        self.depth = depth
        # self.Deform_Embedding = Deform_Embedding(in_chans=h.dense_channel, embed_dim=h.dense_channel)
        self.dense_block = nn.ModuleList([])
        for i in range(depth):
            dil = 2**i
            dense_conv = nn.Sequential(
                nn.Conv2d(
                    h.dense_channel * (i + 1),
                    h.dense_channel * (i + 1),
                    kernel_size,
                    dilation=(dil, 1),
                    padding=get_padding_2d(kernel_size, dilation=(dil, 1)),
                    groups=h.dense_channel * (i + 1),
                    bias=True,
                ),
                nn.Conv2d(
                    in_channels=h.dense_channel * (i + 1),
                    out_channels=h.dense_channel,
                    kernel_size=1,
                    padding=0,
                    stride=1,
                    groups=1,
                    bias=True,
                ),
                nn.InstanceNorm2d(h.dense_channel, affine=True),
                nn.PReLU(h.dense_channel),
            )
            self.dense_block.append(dense_conv)

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            # if i == 0:
            #     x = self.Deform_Embedding(x)
            #     x = self.dense_block[i](x)
            # else:
            x = self.dense_block[i](skip)
            skip = torch.cat([x, skip], dim=1)
        return x


class DenseEncoder(nn.Module):
    def __init__(self, h, in_channel):
        super(DenseEncoder, self).__init__()
        self.h = h
        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, h.dense_channel, (1, 1)),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel),
        )

        self.dense_block = DS_DDB(h, depth=4)  # [b, h.dense_channel, ndim_time, h.n_fft//2+1]

        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2)),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel),
        )

    def forward(self, x):
        x = self.dense_conv_1(x)  # [b, 64, T, F]
        x = self.dense_block(x)  # [b, 64, T, F]
        x = self.dense_conv_2(x)  # [b, 64, T, F//2]
        return x


class MaskDecoder(nn.Module):
    def __init__(self, h, out_channel=1):
        super(MaskDecoder, self).__init__()
        self.dense_block = DS_DDB(h, depth=4)
        self.mask_conv = nn.Sequential(
            nn.ConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2)),
            nn.Conv2d(h.dense_channel, out_channel, (1, 1)),
            nn.InstanceNorm2d(out_channel, affine=True),
            nn.PReLU(out_channel),
            nn.Conv2d(out_channel, out_channel, (1, 1)),
        )
        self.lsigmoid = LearnableSigmoid_2d(h.n_fft // 2 + 1, beta=h.beta)

    def forward(self, x):
        x = self.dense_block(x)
        # x = self.dysample(x)
        x = self.mask_conv(x)
        x = x.permute(0, 3, 2, 1).squeeze(-1)
        x = self.lsigmoid(x).permute(0, 2, 1).unsqueeze(1)
        return x


class PhaseDecoder(nn.Module):
    def __init__(self, h, out_channel=1):
        super(PhaseDecoder, self).__init__()
        self.dense_block = DS_DDB(h, depth=4)
        # self.dysample = DySample(h.dense_channel)
        self.phase_conv = nn.Sequential(
            nn.ConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2)),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel),
        )
        self.phase_conv_r = nn.Conv2d(h.dense_channel, out_channel, (1, 1))
        self.phase_conv_i = nn.Conv2d(h.dense_channel, out_channel, (1, 1))

    def forward(self, x):
        x = self.dense_block(x)
        # x = self.dysample(x)
        x = self.phase_conv(x)
        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)
        x = torch.atan2(x_i, x_r)
        return x


class TS_BLOCK(nn.Module):
    def __init__(self, h):
        super(TS_BLOCK, self).__init__()
        self.h = h
        self.time = GPFCA(h.dense_channel)
        self.freq = GPFCA(h.dense_channel)
        self.beta = nn.Parameter(torch.zeros((1, 1, 1, h.dense_channel)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, 1, 1, h.dense_channel)), requires_grad=True)

    def forward(self, x):
        b, c, t, f = x.size()
        x = x.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)

        x = self.time(x) + x * self.beta
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b * t, f, c)

        x = self.freq(x) + x * self.gamma
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)
        return x


class LKFCA_Net(nn.Module):
    def __init__(self, h=h_dict(), num_tsblock=4):
        super(LKFCA_Net, self).__init__()
        self.h = h
        self.num_tsblock = num_tsblock
        self.dense_encoder = DenseEncoder(h, in_channel=2)
        self.LKFCAnet = nn.ModuleList([])
        for i in range(num_tsblock):
            self.LKFCAnet.append(TS_BLOCK(h))
        self.mask_decoder = MaskDecoder(h, out_channel=1)
        self.phase_decoder = PhaseDecoder(h, out_channel=1)
        self.stft = STFT(h.win_size, h.hop_size)

    def forward(self, inp):  # [B, F, T]
        # noisy_mag = noisy_mag.unsqueeze(-1).permute(0, 3, 2, 1)  # [B, 1, T, F]
        # noisy_pha = noisy_pha.unsqueeze(-1).permute(0, 3, 2, 1)  # [B, 1, T, F]
        xk = self.stft.transform(inp)
        noisy_mag = torch.norm(xk, dim=1).unsqueeze(1)
        noisy_pha = torch.atan2(xk[:, 1, ...], xk[:, 0, ...]).unsqueeze(1)  # b,t,f

        x = torch.cat((noisy_mag, noisy_pha), dim=1)  # [B, 2, T, F]
        x = self.dense_encoder(x)

        for i in range(self.num_tsblock):
            x = self.LKFCAnet[i](x)

        denoised_mag = (noisy_mag * self.mask_decoder(x)).permute(0, 3, 2, 1).squeeze(-1)
        denoised_pha = self.phase_decoder(x).permute(0, 3, 2, 1).squeeze(-1)
        # denoised_com = torch.stack(
        #     (denoised_mag * torch.cos(denoised_pha), denoised_mag * torch.sin(denoised_pha)), dim=-1
        # )
        denoised_com = torch.stack(
            (denoised_mag * torch.cos(denoised_pha), denoised_mag * torch.sin(denoised_pha)), dim=1
        ).permute(0, 1, 3, 2)

        out = self.stft.inverse(denoised_com)

        return out


class LKFCA_NetFIG6(nn.Module):
    def __init__(self, h=h_dict(), num_tsblock=4):
        super(LKFCA_NetFIG6, self).__init__()
        self.h = h
        self.num_tsblock = num_tsblock
        self.dense_encoder = DenseEncoder(h, in_channel=3)
        self.LKFCAnet = nn.ModuleList([])
        for i in range(num_tsblock):
            self.LKFCAnet.append(TS_BLOCK(h))
        self.mask_decoder = MaskDecoder(h, out_channel=1)
        self.phase_decoder = PhaseDecoder(h, out_channel=1)
        self.stft = STFT(h.win_size, h.hop_size)
        self.reso = 16000 / h.win_size

    def forward(self, inp, HL):  # [B, F, T]
        eps = torch.finfo(inp.dtype).eps
        # noisy_mag = noisy_mag.unsqueeze(-1).permute(0, 3, 2, 1)  # [B, 1, T, F]
        # noisy_pha = noisy_pha.unsqueeze(-1).permute(0, 3, 2, 1)  # [B, 1, T, F]
        xk = self.stft.transform(inp)
        hl = expand_HT(HL, xk.shape[-2], self.reso)  # B,C(1),T,F
        noisy_mag = torch.norm(xk, dim=1).unsqueeze(1)
        noisy_pha = torch.atan2(xk[:, 1, ...], xk[:, 0, ...]).unsqueeze(1)  # b,t,f

        x = torch.cat((noisy_mag, noisy_pha, hl), dim=1)  # [B, 2, T, F]
        x = self.dense_encoder(x)

        for i in range(self.num_tsblock):
            x = self.LKFCAnet[i](x)

        denoised_mag = (noisy_mag * self.mask_decoder(x)).permute(0, 3, 2, 1).squeeze(-1)
        denoised_pha = self.phase_decoder(x).permute(0, 3, 2, 1).squeeze(-1)
        # denoised_com = torch.stack(
        #     (denoised_mag * torch.cos(denoised_pha), denoised_mag * torch.sin(denoised_pha)), dim=-1
        # )
        denoised_com = torch.stack(
            (denoised_mag * torch.cos(denoised_pha), denoised_mag * torch.sin(denoised_pha)), dim=1
        ).permute(0, 1, 3, 2)

        out = self.stft.inverse(denoised_com)

        return out


if __name__ == "__main__":
    # inp = torch.randn(1, 257, 63)
    inp = torch.randn(1, 16000)
    hl = torch.randn(1, 6)
    net = LKFCA_NetFIG6()
    out = net(inp, hl)

    check_flops(net, inp, hl)
