from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed
from pesq import pesq

from JointNSHModel import expand_HT
from models.conv_stft import STFT
from models.transformer import TransformerBlock
from utils.check_flops import check_flops
from utils.register import tables


@dataclass
class h_dict:
    n_fft: int = 512
    hop_size: int = 256
    win_size: int = 512
    beta: float = 2.0
    dense_channel: int = 48


def phase_losses(phase_r, phase_g):
    """
    B,F,T
    """
    ip_loss = torch.mean(anti_wrapping_function(phase_r - phase_g))
    gd_loss = torch.mean(
        anti_wrapping_function(torch.diff(phase_r, dim=1) - torch.diff(phase_g, dim=1))
    )
    iaf_loss = torch.mean(
        anti_wrapping_function(torch.diff(phase_r, dim=2) - torch.diff(phase_g, dim=2))
    )

    phase_loss = ip_loss + gd_loss + iaf_loss
    return phase_loss, dict(
        ip_loss=ip_loss.detach(), gd_loss=gd_loss.detach(), iaf_loss=iaf_loss.detach()
    )


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
        pesq_score = -1

    return pesq_score


class LearnableSigmoid2d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1))

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


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
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out


class DenseBlock(nn.Module):
    def __init__(self, h, kernel_size=(2, 3), depth=4):
        super(DenseBlock, self).__init__()
        self.h = h
        self.depth = depth
        self.dense_block = nn.ModuleList([])
        for i in range(depth):
            dilation = 2**i
            pad_length = dilation
            dense_conv = nn.Sequential(
                nn.ConstantPad2d((1, 1, pad_length, 0), value=0.0),
                nn.Conv2d(
                    h.dense_channel * (i + 1), h.dense_channel, kernel_size, dilation=(dilation, 1)
                ),
                nn.InstanceNorm2d(h.dense_channel, affine=True),
                nn.PReLU(h.dense_channel),
            )
            self.dense_block.append(dense_conv)

    def forward(self, x):
        skip = x
        for i in range(self.depth):
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

        self.dense_block = DenseBlock(h, depth=4)

        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2), padding=(0, 1)),
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
        self.dense_block = DenseBlock(h, depth=4)
        self.mask_conv = nn.Sequential(
            SPConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), 2),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel),
            nn.Conv2d(h.dense_channel, out_channel, (1, 2)),
        )
        self.lsigmoid = LearnableSigmoid2d(h.n_fft // 2 + 1, beta=h.beta)

    def forward(self, x):
        x = self.dense_block(x)
        x = self.mask_conv(x)
        x = x.permute(0, 3, 2, 1).squeeze(-1)  # [B, F, T]
        x = self.lsigmoid(x)
        return x


class PhaseDecoder(nn.Module):
    def __init__(self, h, out_channel=1):
        super(PhaseDecoder, self).__init__()
        self.dense_block = DenseBlock(h, depth=4)
        self.phase_conv = nn.Sequential(
            SPConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), 2),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel),
        )
        self.phase_conv_r = nn.Conv2d(h.dense_channel, out_channel, (1, 2))
        self.phase_conv_i = nn.Conv2d(h.dense_channel, out_channel, (1, 2))

    def forward(self, x):
        x = self.dense_block(x)
        x = self.phase_conv(x)
        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)
        x = torch.atan2(x_i, x_r)
        x = x.permute(0, 3, 2, 1).squeeze(-1)  # [B, F, T]
        return x


class TSTransformerBlock(nn.Module):
    def __init__(self, h):
        super(TSTransformerBlock, self).__init__()
        self.h = h
        self.time_transformer = TransformerBlock(input_feature_size=h.dense_channel, n_heads=4)
        self.freq_transformer = TransformerBlock(input_feature_size=h.dense_channel, n_heads=4)

    def forward(self, x):
        b, c, t, f = x.size()
        x = x.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        x = self.time_transformer(x) + x
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b * t, f, c)
        x = self.freq_transformer(x) + x
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)
        return x


class MPNet(nn.Module):
    def __init__(self, h, num_tsblocks=4):
        super(MPNet, self).__init__()
        self.h = h
        self.num_tscblocks = num_tsblocks
        self.dense_encoder = DenseEncoder(h, in_channel=2)

        self.TSTransformer = nn.ModuleList([])
        for i in range(num_tsblocks):
            self.TSTransformer.append(TSTransformerBlock(h))

        self.mask_decoder = MaskDecoder(h, out_channel=1)
        self.phase_decoder = PhaseDecoder(h, out_channel=1)

    def forward(self, noisy_amp, noisy_pha):  # [B, F, T]
        x = torch.stack((noisy_amp, noisy_pha), dim=-1).permute(0, 3, 2, 1)  # [B, 2, T, F]
        x = self.dense_encoder(x)

        for i in range(self.num_tscblocks):
            x = self.TSTransformer[i](x)

        denoised_amp = noisy_amp * self.mask_decoder(x)
        denoised_pha = self.phase_decoder(x)
        denoised_com = torch.stack(
            (denoised_amp * torch.cos(denoised_pha), denoised_amp * torch.sin(denoised_pha)), dim=-1
        )

        return denoised_amp, denoised_pha, denoised_com


@tables.register("models", "MP_SENet")
class MPNetT(nn.Module):
    def __init__(self, h, num_tsblocks=4):
        super().__init__()
        self.h = h
        self.num_tscblocks = num_tsblocks
        self.dense_encoder = DenseEncoder(h, in_channel=2)

        self.TSTransformer = nn.ModuleList([])
        for i in range(num_tsblocks):
            self.TSTransformer.append(TSTransformerBlock(h))

        self.mask_decoder = MaskDecoder(h, out_channel=1)
        self.phase_decoder = PhaseDecoder(h, out_channel=1)

        self.stft = STFT(h.win_size, h.hop_size)

    def forward(self, inp):  # [B, F, T]
        x = self.stft.transform(inp)
        # x = torch.stack((noisy_amp, noisy_pha), dim=-1).permute(0, 3, 2, 1)  # [B, 2, T, F]
        noisy_amp = x.pow(2).sum(1).sqrt().permute(0, 2, 1)  # B,F,T
        x = self.dense_encoder(x)

        for i in range(self.num_tscblocks):
            x = self.TSTransformer[i](x)

        denoised_amp = noisy_amp * self.mask_decoder(x)
        denoised_pha = self.phase_decoder(x)
        denoised_com = torch.stack(
            (denoised_amp * torch.cos(denoised_pha), denoised_amp * torch.sin(denoised_pha)), dim=-1
        )  # b,f,t,2
        out = self.stft.inverse(denoised_com.permute(0, 3, 2, 1))

        return out, (denoised_amp, denoised_pha, denoised_com)


@tables.register("models", "MP_SENetFIG6")
class MPNetTFIG6(nn.Module):
    def __init__(self, h=h_dict, num_tsblocks=4):
        super().__init__()
        self.h = h
        self.num_tscblocks = num_tsblocks
        self.dense_encoder = DenseEncoder(h, in_channel=3)

        self.TSTransformer = nn.ModuleList([])
        for i in range(num_tsblocks):
            self.TSTransformer.append(TSTransformerBlock(h))

        self.mask_decoder = MaskDecoder(h, out_channel=1)
        self.phase_decoder = PhaseDecoder(h, out_channel=1)

        self.stft = STFT(h.win_size, h.hop_size)
        self.reso = 16000 / h.win_size

    def forward(self, inp, HL):  # [B, F, T]
        x = self.stft.transform(inp)  # b,2,t,f
        hl = expand_HT(HL, x.shape[-2], self.reso)  # B,C(1),T,F
        # x = torch.stack((noisy_amp, noisy_pha), dim=-1).permute(0, 3, 2, 1)  # [B, 2, T, F]
        noisy_amp = x.pow(2).sum(1).sqrt().permute(0, 2, 1)  # B,F,T
        x = torch.concat([x, hl], dim=1)
        x = self.dense_encoder(x)

        for i in range(self.num_tscblocks):
            x = self.TSTransformer[i](x)

        denoised_amp = noisy_amp * self.mask_decoder(x)
        denoised_pha = self.phase_decoder(x)
        denoised_com = torch.stack(
            (denoised_amp * torch.cos(denoised_pha), denoised_amp * torch.sin(denoised_pha)), dim=-1
        )  # b,f,t,2
        out = self.stft.inverse(denoised_com.permute(0, 3, 2, 1))

        return out

    def loss(self, sph, enh):
        """
        B,T
        """
        sph = sph[:, : enh.size(-1)]
        xk_sph = self.stft.transform(sph)
        xk_enh = self.stft.transform(enh)

        phase_sph = torch.atan2(xk_sph[:, 1, ...], xk_sph[:, 0, ...]).transpose(-1, -2)
        phase_enh = torch.atan2(xk_enh[:, 1, ...], xk_enh[:, 0, ...]).transpose(-1, -2)
        mag_sph = xk_sph.pow(2).sum(1).sqrt()
        mag_enh = xk_enh.pow(2).sum(1).sqrt()

        time_lv = 0.2 * F.l1_loss(sph, enh)
        phase_lv, meta = phase_losses(phase_sph, phase_enh)
        mag_lv = 0.9 * F.mse_loss(mag_sph, mag_enh)
        spec_lv = 0.1 * F.mse_loss(xk_sph, xk_enh)

        loss_lv = time_lv + mag_lv + spec_lv + 0.3 * phase_lv
        meta.update(
            {
                "loss": loss_lv,
                "time_lv": time_lv.detach(),
                "mag_lv": mag_lv.detach(),
                "spec_lv": spec_lv.detach(),
                "phase_lv": 0.3 * phase_lv.detach(),
            }
        )
        return meta


if __name__ == "__main__":
    inp = torch.randn(1, 16000).float()
    lbl = torch.randn(1, 16000).float()
    hl = torch.randn(1, 6).float()
    net = MPNetTFIG6()
    check_flops(net, inp, hl)

    net.loss(lbl, inp)
