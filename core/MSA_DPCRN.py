import sys
from pathlib import Path

from einops import rearrange

sys.path.append(str(Path(__file__).parent.parent))
from typing import List, Optional

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from models.complexnn import (
    ComplexConv2d,
    ComplexGateConvTranspose2d,
    InstanceNorm,
    complex_apply_mask,
    complex_cat,
)
from models.conv_stft import STFT
from models.ft_lstm import FTLSTM_RESNET
from models.Fusion.ms_cam import MS_CAM_F
from utils.register import tables


class TFusion(nn.Module):
    def __init__(self, inp_channel: int) -> None:
        super().__init__()
        self.l = nn.Sequential(
            nn.Conv2d(
                in_channels=inp_channel,
                out_channels=inp_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Tanh(),
        )

    def forward(self, x, y):
        return self.l(x) * y


class FTAttention(nn.Module):
    def __init__(self, inp_channel: int, winL: int = 10) -> None:
        super().__init__()
        self.attn_f = nn.MultiheadAttention(embed_dim=inp_channel, num_heads=2, batch_first=True)

        self.attn_t = nn.MultiheadAttention(embed_dim=inp_channel, num_heads=2, batch_first=True)
        self.attn_window = winL

    def forward(self, k, q, v):
        """b,c,t,f"""
        nB = k.shape[0]
        k_ = rearrange(k, "b c t f -> (b t) f c")
        q_ = rearrange(q, "b c t f -> (b t) f c")
        v_ = rearrange(v, "b c t f -> (b t) f c")

        vf, _ = self.attn_f(k_, q_, v_)
        v = v + rearrange(vf, "(b t) f c -> b c t f", b=nB)

        nT = v.shape[2]
        mask_1 = torch.ones(nT, nT, device=v.device).triu_(1).bool()  # TxT
        mask_2 = torch.ones(nT, nT, device=v.device).tril_(-self.attn_window).bool()  # TxT
        mask = mask_1 + mask_2

        k_ = rearrange(k, "b c t f -> (b f) t c")
        q_ = rearrange(q, "b c t f -> (b f) t c")
        v_ = rearrange(v, "b c t f -> (b f) t c")

        vt, _ = self.attn_t(k_, q_, v_, attn_mask=mask)
        v = v + rearrange(vt, "(b f) t c -> b c t f", b=nB)

        return v


class ChannelFreqAttention(nn.Module):
    def __init__(self, inp_channels: int, feature_size: int) -> None:
        super().__init__()

        self.layer_ch = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1, feature_size), stride=(1, feature_size)),  # B,C,T,1
            nn.Conv2d(
                in_channels=inp_channels,
                out_channels=inp_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            Rearrange("b c t f-> b t c f"),
            nn.LayerNorm(1, inp_channels),
            Rearrange("b t c f-> b c t f"),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=inp_channels,
                out_channels=inp_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=inp_channels,
            ),
            nn.Sigmoid(),
        )

        self.layer_freq = nn.Sequential(
            nn.Conv2d(
                in_channels=inp_channels,
                out_channels=inp_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            Rearrange("b c t f-> b t c f"),
            nn.LayerNorm(feature_size, inp_channels),
            Rearrange("b t c f-> b c t f"),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=inp_channels,
                out_channels=inp_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x * self.layer_ch(x)
        x = x * self.layer_freq(x)

        return x


class CRN_AEC(nn.Module):
    def __init__(
        self,
        nframe: int,
        nhop: int,
        nfft: Optional[int] = None,
        cnn_num: List = [16, 32, 64, 64],
        stride: List = [2, 1, 2, 1],
        rnn_hidden_num: int = 64,
    ):
        super().__init__()
        self.nframe = nframe
        self.nhop = nhop
        self.fft_dim = nframe // 2 + 1
        self.cnn_num = [4] + cnn_num

        self.stft = STFT(nframe, nhop, nfft=nframe if nfft is None else nfft)

        self.conv_l = nn.ModuleList()

        self.encoder_l = nn.ModuleList()
        self.decoder_l = nn.ModuleList()
        self.atten_l = nn.ModuleList()
        n_cnn_layer = len(self.cnn_num) - 1

        nbin = self.fft_dim
        nbinT = (self.fft_dim >> stride.count(2)) + 1

        for idx in range(n_cnn_layer):
            # feat_num = (self.fft_dim >> idx + 1) + 1
            # batchCh = self.cnn_num[idx + 1] * ((self.fft_dim >> idx + 1) + 1)
            nbin = ((nbin >> 1) + 1) if stride[idx] == 2 else nbin
            nbinT = (nbinT << 1) - 1 if stride[-1 - idx] == 2 else nbinT

            self.encoder_l.append(
                nn.Sequential(
                    ComplexConv2d(
                        in_channels=self.cnn_num[idx],
                        out_channels=self.cnn_num[idx + 1],
                        kernel_size=(3, 5),
                        padding=(2, 2),  # (k_h - 1)/2
                        stride=(1, stride[idx]),
                    ),
                    # nn.BatchNorm2d(self.cnn_num[idx + 1]),
                    # nn.InstanceNorm2d(self.cnn_num[idx + 1]),
                    # InstanceNorm(),
                    InstanceNorm(self.cnn_num[idx + 1] * nbin),
                    nn.PReLU(),
                )
            )

            if idx != n_cnn_layer - 1:
                self.decoder_l.append(
                    nn.Sequential(
                        # ComplexConvTranspose2d(
                        ComplexGateConvTranspose2d(
                            in_channels=2 * self.cnn_num[-1 - idx],  # skip_connection
                            out_channels=self.cnn_num[-1 - idx - 1],
                            kernel_size=(1, 5),
                            padding=(0, 2),
                            stride=(1, stride[-1 - idx]),
                        ),
                        # nn.BatchNorm2d(self.cnn_num[-1 - idx - 1] // 2),
                        # nn.InstanceNorm2d(self.cnn_num[-1 - idx - 1] // 2),
                        # InstanceNorm(),
                        InstanceNorm(self.cnn_num[-1 - idx - 1] * nbinT),
                        # * ((self.fft_dim >> n_cnn_layer - idx - 1) + 1)
                        nn.PReLU(),
                    )
                )
            else:
                self.decoder_l.append(
                    nn.Sequential(
                        ComplexGateConvTranspose2d(
                            in_channels=2 * self.cnn_num[-1 - idx],  # skip_connection
                            out_channels=2,
                            kernel_size=(1, 5),
                            padding=(0, 2),
                            stride=(1, stride[-1 - idx]),
                        ),
                        # InstanceNorm(2 * self.fft_dim),
                        # nn.PReLU(),
                    )
                )

        # self.rnns_r = FTLSTM(cnn_num[-1] // 2, rnn_hidden_num)
        # self.rnns_i = FTLSTM(cnn_num[-1] // 2, rnn_hidden_num)
        self.rnns_r = nn.Sequential(
            FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
            FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
        )
        self.rnns_i = nn.Sequential(
            FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
            FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
        )

    def forward(self, mic, ref):
        """
        inputs: shape is [B, T] or [B, 1, T]
        """

        specs_mic = self.stft.transform(mic)  # [B, 2, T, F]
        specs_ref = self.stft.transform(ref)

        specs_mic_real, specs_mic_imag = specs_mic.chunk(2, dim=1)  # B,2,T,F
        specs_ref_real, specs_ref_imag = specs_ref.chunk(2, dim=1)

        feat = torch.stack([specs_mic_real, specs_mic_imag], dim=1)

        x = specs_mix
        feat_store = []
        for idx, layer in enumerate(self.encoder_l):
            x = layer(x)  # x shape [B, C, T, F]
            feat_store.append(x)

        nB, nC, nF, nT = x.shape
        x_r, x_i = torch.chunk(x, 2, dim=1)

        feat_r = self.rnns_r(x_r)
        feat_i = self.rnns_i(x_i)

        # mask_r, mask_i = F.tanh(mask_r), F.tanh(mask_i)
        # cmask = torch.concatenate([mask_r, mask_i], dim=1)
        # feat_r, feat_i = complex_mask_multi(feat, cmask)

        feat = torch.concat([feat_r, feat_i], dim=1)

        # B,C,F,T
        x = feat
        for idx, layer in enumerate(self.decoder_l):
            x = complex_cat([x, feat_store[-idx - 1]], dim=1)
            # print("Tconv", idx, feat.shape)
            x = layer(x)
            # feat = feat[..., 1:]  # padding=(2, 0) will padding 1 col in Time dimension

        feat_r, feat_i = complex_apply_mask(specs_mic, x)
        feat_r = feat_r.squeeze(dim=1).permute(0, 2, 1)  # b1tf -> bft
        feat_i = feat_i.squeeze(dim=1).permute(0, 2, 1)
        feat = torch.concat([feat_r, feat_i], dim=1)  # b,2f,t
        # print("Tconv", feat.shape)
        # B, 2, F, T -> B, F(r, i), T
        # feat = feat.reshape(nB, self.fft_dim * 2, -1)  # B, F, T
        feat = self.post_conv(feat)  # b,f,t
        r, i = feat.permute(0, 2, 1).chunk(2, dim=-1)  # b,t,f
        feat = torch.stack([r, i], dim=1)

        out_wav = self.stft.inverse(feat)
        out_wav = torch.squeeze(out_wav, 1)
        out_wav = torch.clamp(out_wav, -1, 1)

        return out_wav


@tables.register("models", "msa_dpcrn")
class MSA_DPCRN(nn.Module):
    def __init__(
        self,
        nframe: int,
        nhop: int,
        nfft: Optional[int] = None,
        cnn_num: List = [32, 64, 128, 128],
        stride: List = [2, 2, 2, 2],
        rnn_hidden_num: int = 64,
    ):
        super().__init__()
        self.nframe = nframe
        self.nhop = nhop
        self.fft_dim = nframe // 2 + 1
        self.cnn_num = [4] + cnn_num

        self.stft = STFT(nframe, nhop, nfft=nframe if nfft is None else nfft)

        self.encoder_rel = nn.ModuleList()
        self.encoder_mic = nn.ModuleList()
        self.decoder_l = nn.ModuleList()
        n_cnn_layer = len(self.cnn_num) - 1

        nbin = self.fft_dim
        nbinT = (self.fft_dim >> stride.count(2)) + 1

        for idx in range(n_cnn_layer):
            # feat_num = (self.fft_dim >> idx + 1) + 1
            # batchCh = self.cnn_num[idx + 1] * ((self.fft_dim >> idx + 1) + 1)
            nbin = ((nbin >> 1) + 1) if stride[idx] == 2 else nbin
            nbinT = (nbinT << 1) - 1 if stride[-1 - idx] == 2 else nbinT

            self.encoder_rel.append(
                nn.Sequential(
                    ComplexConv2d(
                        in_channels=self.cnn_num[idx],
                        out_channels=self.cnn_num[idx + 1],
                        kernel_size=(3, 5),
                        padding=(2, 2),  # (k_h - 1)/2
                        stride=(1, stride[idx]),
                    ),
                    # nn.BatchNorm2d(self.cnn_num[idx + 1]),
                    # nn.InstanceNorm2d(self.cnn_num[idx + 1]),
                    # InstanceNorm(),
                    InstanceNorm(self.cnn_num[idx + 1] * nbin),
                    nn.PReLU(),
                )
            )

            self.encoder_mic.append(
                nn.Sequential(
                    ComplexConv2d(
                        in_channels=2 if idx == 0 else self.cnn_num[idx],
                        out_channels=self.cnn_num[idx + 1],
                        kernel_size=(1, 5),
                        padding=(0, 2),
                        stride=(1, stride[idx]),
                    ),
                    # nn.BatchNorm2d(self.cnn_num[idx + 1] // 2),
                    # nn.InstanceNorm2d(self.cnn_num[idx + 1] // 2),
                    # InstanceNorm(),
                    InstanceNorm(self.cnn_num[idx + 1] * nbin),
                    nn.PReLU(),
                )
            )

            if idx != n_cnn_layer - 1:
                self.decoder_l.append(
                    nn.Sequential(
                        # ComplexConvTranspose2d(
                        ComplexGateConvTranspose2d(
                            in_channels=2 * self.cnn_num[-1 - idx],  # skip_connection
                            out_channels=self.cnn_num[-1 - idx - 1],
                            kernel_size=(1, 5),
                            padding=(0, 2),
                            stride=(1, stride[-1 - idx]),
                        ),
                        InstanceNorm(self.cnn_num[-1 - idx - 1] * nbinT),
                        nn.PReLU(),
                    )
                )
            else:
                self.decoder_l.append(
                    nn.Sequential(
                        ComplexGateConvTranspose2d(
                            in_channels=2 * self.cnn_num[-1 - idx],  # skip_connection
                            out_channels=2,
                            kernel_size=(1, 5),
                            padding=(0, 2),
                            stride=(1, stride[-1 - idx]),
                        ),
                        # InstanceNorm(2 * self.fft_dim),
                        # nn.PReLU(),
                    )
                )

        self.encoder_fusion = MS_CAM_F(inp_channels=self.cnn_num[-1], feature_size=nbin, r=1)

        self.rnns_r = nn.ModuleList(
            [
                FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
                FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
            ]
        )

        self.rnns_i = nn.ModuleList(
            [
                FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
                FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
            ]
        )

    def forward(self, mic, ref):
        """
        inputs: shape is [B, T] or [B, 1, T]
        """

        specs_mic = self.stft.transform(mic)  # [B, 2, T, F]
        specs_ref = self.stft.transform(ref)

        specs_mic_real, specs_mic_imag = specs_mic.chunk(2, dim=1)  # B,1,T,F
        specs_ref_real, specs_ref_imag = specs_ref.chunk(2, dim=1)

        # mag_spec_mic = torch.sqrt(specs_mic_real**2 + specs_mic_imag**2)
        # phs_spec_mic = torch.atan2(specs_mic_imag, specs_mic_real)

        specs_mix = torch.concat(
            [specs_mic_real, specs_ref_real, specs_mic_imag, specs_ref_imag], dim=1
        )  # [B, 4, F, T]

        spec_store = []
        spec = specs_mic
        x = specs_mix
        for idx, (lm, lr) in enumerate(zip(self.encoder_mic, self.encoder_rel)):
            spec = lm(spec)
            x = lr(x)  # x shape [B, C, T, F]
            spec_store.append(spec)
            # spec_store.append(x)
            # mx = complex_cat([spec, x], dim=1)
            # spec_store.append(mx)

        x = self.encoder_fusion(x, spec)
        x_r, x_i = torch.chunk(x, 2, dim=1)

        # x_r = self.rnns_r(x_r)
        # x_i = self.rnns_i(x_i)
        for idx, l in enumerate(self.rnns_r):
            x_r, _ = l(x_r)

        # print("1:", x_r[0, 0, 0, :])
        # print("1:", x_r[0, 0, 1, :])
        for idx, l in enumerate(self.rnns_i):
            x_i, _ = l(x_i)

        x = torch.concatenate([x_r, x_i], dim=1)

        # x = self.dilateds[2](x)

        # feat_r, feat_i = complex_mask_multi(feat, cmask)

        # feat = torch.concat([feat_r, feat_i], dim=1)

        for idx, layer in enumerate(self.decoder_l):
            x = complex_cat([x, spec_store[-idx - 1]], dim=1)
            x = layer(x)

        # print("1:", x[0, 0, 1, :])
        feat_r, feat_i = complex_apply_mask(specs_mic, x)
        x = torch.concat([feat_r, feat_i], dim=1)

        feat = x

        out_wav = self.stft.inverse(feat)  # B, 1, T
        out_wav = torch.squeeze(out_wav, 1)
        out_wav = torch.clamp(out_wav, -1, 1)

        return out_wav


@tables.register("models", "msa_dpcrn_inp_spec")
class MSA_DPCRN_SPEC(nn.Module):
    def __init__(
        self,
        nframe: int,
        nhop: int,
        nfft: Optional[int] = None,
        cnn_num: List = [32, 64, 128, 128],
        stride: List = [2, 2, 2, 2],
        rnn_hidden_num: int = 64,
    ):
        super().__init__()
        self.nframe = nframe
        self.nhop = nhop
        self.fft_dim = nframe // 2 + 1
        self.cnn_num = [4] + cnn_num

        self.stft = STFT(nframe, nhop, nfft=nframe if nfft is None else nfft)

        self.encoder_rel = nn.ModuleList()
        self.encoder_mic = nn.ModuleList()
        self.decoder_l = nn.ModuleList()
        n_cnn_layer = len(self.cnn_num) - 1

        nbin = self.fft_dim
        nbinT = (self.fft_dim >> stride.count(2)) + 1

        for idx in range(n_cnn_layer):
            # feat_num = (self.fft_dim >> idx + 1) + 1
            # batchCh = self.cnn_num[idx + 1] * ((self.fft_dim >> idx + 1) + 1)
            nbin = ((nbin >> 1) + 1) if stride[idx] == 2 else nbin
            nbinT = (nbinT << 1) - 1 if stride[-1 - idx] == 2 else nbinT

            self.encoder_rel.append(
                nn.Sequential(
                    ComplexConv2d(
                        in_channels=self.cnn_num[idx],
                        out_channels=self.cnn_num[idx + 1],
                        kernel_size=(3, 5),
                        padding=(2, 2),  # (k_h - 1)/2
                        stride=(1, stride[idx]),
                    ),
                    # nn.BatchNorm2d(self.cnn_num[idx + 1]),
                    # nn.InstanceNorm2d(self.cnn_num[idx + 1]),
                    # InstanceNorm(),
                    InstanceNorm(self.cnn_num[idx + 1] * nbin),
                    nn.PReLU(),
                )
            )

            self.encoder_mic.append(
                nn.Sequential(
                    ComplexConv2d(
                        in_channels=2 if idx == 0 else self.cnn_num[idx],
                        out_channels=self.cnn_num[idx + 1],
                        kernel_size=(1, 5),
                        padding=(0, 2),
                        stride=(1, stride[idx]),
                    ),
                    # nn.BatchNorm2d(self.cnn_num[idx + 1] // 2),
                    # nn.InstanceNorm2d(self.cnn_num[idx + 1] // 2),
                    # InstanceNorm(),
                    InstanceNorm(self.cnn_num[idx + 1] * nbin),
                    nn.PReLU(),
                )
            )

            if idx != n_cnn_layer - 1:
                self.decoder_l.append(
                    nn.Sequential(
                        # ComplexConvTranspose2d(
                        ComplexGateConvTranspose2d(
                            in_channels=2 * self.cnn_num[-1 - idx],  # skip_connection
                            out_channels=self.cnn_num[-1 - idx - 1],
                            kernel_size=(1, 5),
                            padding=(0, 2),
                            stride=(1, stride[-1 - idx]),
                        ),
                        InstanceNorm(self.cnn_num[-1 - idx - 1] * nbinT),
                        nn.PReLU(),
                    )
                )
            else:
                self.decoder_l.append(
                    nn.Sequential(
                        ComplexGateConvTranspose2d(
                            in_channels=2 * self.cnn_num[-1 - idx],  # skip_connection
                            out_channels=2,
                            kernel_size=(1, 5),
                            padding=(0, 2),
                            stride=(1, stride[-1 - idx]),
                        ),
                        # InstanceNorm(2 * self.fft_dim),
                        # nn.PReLU(),
                    )
                )

        self.encoder_fusion = MS_CAM_F(inp_channels=self.cnn_num[-1], feature_size=nbin, r=1)

        self.rnns_r = nn.ModuleList(
            [
                FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
                FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
            ]
        )

        self.rnns_i = nn.ModuleList(
            [
                FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
                FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
            ]
        )

    def pack_state(self, state):
        pass

    def unpack_state(self, state):
        pass

    def forward(self, specs_mic, specs_ref):
        """
        inputs: shape is [B, 2, T, F]
        output: B,2,T,F
        """

        specs_mic_real, specs_mic_imag = specs_mic.chunk(2, dim=1)  # B,1,T,F
        specs_ref_real, specs_ref_imag = specs_ref.chunk(2, dim=1)

        # mag_spec_mic = torch.sqrt(specs_mic_real**2 + specs_mic_imag**2)
        # phs_spec_mic = torch.atan2(specs_mic_imag, specs_mic_real)

        specs_mix = torch.concat(
            [specs_mic_real, specs_ref_real, specs_mic_imag, specs_ref_imag], dim=1
        )  # [B, 4, F, T]

        spec_store = []
        spec = specs_mic
        x = specs_mix
        for idx, (lm, lr) in enumerate(zip(self.encoder_mic, self.encoder_rel)):
            spec = lm(spec)
            x = lr(x)  # x shape [B, C, T, F]
            spec_store.append(spec)

        x = self.encoder_fusion(x, spec)
        x_r, x_i = torch.chunk(x, 2, dim=1)

        # x_r = self.rnns_r(x_r)
        # x_i = self.rnns_i(x_i)
        for idx, l in enumerate(self.rnns_r):
            x_r, _ = l(x_r)

        for idx, l in enumerate(self.rnns_i):
            x_i, _ = l(x_i)

        x = torch.concatenate([x_r, x_i], dim=1)

        for idx, layer in enumerate(self.decoder_l):
            x = complex_cat([x, spec_store[-idx - 1]], dim=1)
            x = layer(x)

        feat_r, feat_i = complex_apply_mask(specs_mic, x)
        spec_enh = torch.concat([feat_r, feat_i], dim=1)

        return spec_enh


@tables.register("models", "msa_dpcrn_inp_spec_online")
class MSA_DPCRN_SPEC_online(nn.Module):
    def __init__(
        self,
        nframe: int,
        nhop: int,
        nfft: Optional[int] = None,
        cnn_num: List = [32, 64, 128, 128],
        stride: List = [2, 2, 2, 2],
        rnn_hidden_num: int = 64,
    ):
        super().__init__()
        self.nframe = nframe
        self.nhop = nhop
        self.fft_dim = nframe // 2 + 1
        self.cnn_num = [4] + cnn_num

        self.encoder_rel = nn.ModuleList()
        self.encoder_mic = nn.ModuleList()
        self.decoder_l = nn.ModuleList()

        self.stft = STFT(nframe, nhop, nfft=nframe if nfft is None else nfft)

        n_cnn_layer = len(self.cnn_num) - 1

        nbin = self.fft_dim
        nbinT = (self.fft_dim >> stride.count(2)) + 1
        self.buff_x = []

        for idx in range(n_cnn_layer):
            # B,C,T,F_
            self.buff_x.append(torch.zeros(1, self.cnn_num[idx], 3, nbin).float())

            nbin = ((nbin >> 1) + 1) if stride[idx] == 2 else nbin
            nbinT = (nbinT << 1) - 1 if stride[-1 - idx] == 2 else nbinT

            self.encoder_rel.append(
                nn.Sequential(
                    ComplexConv2d(
                        in_channels=self.cnn_num[idx],
                        out_channels=self.cnn_num[idx + 1],
                        kernel_size=(3, 5),
                        padding=(0, 2),  # (k_h - 1)/2
                        stride=(1, stride[idx]),
                    ),
                    InstanceNorm(self.cnn_num[idx + 1] * nbin),
                    nn.PReLU(),
                )
            )

            self.encoder_mic.append(
                nn.Sequential(
                    ComplexConv2d(
                        in_channels=2 if idx == 0 else self.cnn_num[idx],
                        out_channels=self.cnn_num[idx + 1],
                        kernel_size=(1, 5),
                        padding=(0, 2),
                        stride=(1, stride[idx]),
                    ),
                    InstanceNorm(self.cnn_num[idx + 1] * nbin),
                    nn.PReLU(),
                )
            )

            if idx != n_cnn_layer - 1:
                self.decoder_l.append(
                    nn.Sequential(
                        ComplexGateConvTranspose2d(
                            in_channels=2 * self.cnn_num[-1 - idx],  # skip_connection
                            out_channels=self.cnn_num[-1 - idx - 1],
                            kernel_size=(1, 5),
                            padding=(0, 2),
                            stride=(1, stride[-1 - idx]),
                        ),
                        InstanceNorm(self.cnn_num[-1 - idx - 1] * nbinT),
                        nn.PReLU(),
                    )
                )
            else:
                self.decoder_l.append(
                    nn.Sequential(
                        ComplexGateConvTranspose2d(
                            in_channels=2 * self.cnn_num[-1 - idx],  # skip_connection
                            out_channels=2,
                            kernel_size=(1, 5),
                            padding=(0, 2),
                            stride=(1, stride[-1 - idx]),
                        ),
                    )
                )

        self.encoder_fusion = MS_CAM_F(inp_channels=self.cnn_num[-1], feature_size=nbin, r=1)

        self.rnns_r = nn.ModuleList(
            [
                FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
                FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
            ]
        )

        self.rnns_i = nn.ModuleList(
            [
                FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
                FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
            ]
        )

    def forward(self, specs_mic, specs_ref, state):
        """
        inputs: shape is [B, 2, T, F]
        output: B,2,T,F
        """

        specs_mic_real, specs_mic_imag = specs_mic.chunk(2, dim=1)  # B,1,T,F
        specs_ref_real, specs_ref_imag = specs_ref.chunk(2, dim=1)

        specs_mix = torch.concat(
            [specs_mic_real, specs_ref_real, specs_mic_imag, specs_ref_imag], dim=1
        )  # [B, 4, T, F]

        spec_store = []
        spec = specs_mic
        # b,2,t,f
        x = specs_mix

        for idx, (lm, lr) in enumerate(zip(self.encoder_mic, self.encoder_rel)):
            spec = lm(spec)

            self.buff_x[idx] = torch.concat([self.buff_x[idx][..., 1:, :], x], dim=-2)
            x = lr(self.buff_x[idx])  # x shape [B, C, T, F]
            spec_store.append(spec)

        x = self.encoder_fusion(x, spec)
        x_r, x_i = torch.chunk(x, 2, dim=1)

        state_r, state_i = [], []

        for idx, l in enumerate(self.rnns_r):
            stat = state[0][idx] if state is not None else None
            x_r, stat = l(x_r, stat)
            state_r.append(stat)

        # print("2:", x_r[0, 0, 0, :])
        # state_r = torch.concat(state_r, dim=0)

        for idx, l in enumerate(self.rnns_i):
            x_i, stat = l(x_i, None if state is None else state[1][idx])
            state_i.append(stat)

        state = (state_r, state_i)
        # state = torch.stack([state_r, state_i], dim=0)

        x = torch.concatenate([x_r, x_i], dim=1)

        for idx, layer in enumerate(self.decoder_l):
            x = complex_cat([x, spec_store[-idx - 1]], dim=1)
            x = layer(x)

        # print("2:", x[0, 0, 0, :])
        feat_r, feat_i = complex_apply_mask(specs_mic, x)
        spec_enh = torch.concat([feat_r, feat_i], dim=1)

        return spec_enh, state


class MSA_DPCRN_SPEC_onnx(nn.Module):
    def __init__(
        self,
        nframe: int,
        nhop: int,
        nfft: Optional[int] = None,
        cnn_num: List = [32, 64, 128, 128],
        stride: List = [2, 2, 2, 2],
        rnn_hidden_num: int = 64,
    ):
        super().__init__()
        self.nframe = nframe
        self.nhop = nhop
        self.fft_dim = nframe // 2 + 1
        self.cnn_num = [4] + cnn_num

        self.encoder_rel = nn.ModuleList()
        self.encoder_mic = nn.ModuleList()
        self.decoder_l = nn.ModuleList()

        # self.register_buffer("buff_x", torch.zeros(1, 4, 3, self.fft_dim).float())

        n_cnn_layer = len(self.cnn_num) - 1
        self.stft = STFT(nframe, nhop, nfft=nframe if nfft is None else nfft)

        nbin = self.fft_dim
        nbinT = (self.fft_dim >> stride.count(2)) + 1
        self.buff_x = []

        for idx in range(n_cnn_layer):
            self.buff_x.append(torch.zeros(1, self.cnn_num[idx], 3, nbin).float())
            nbin = ((nbin >> 1) + 1) if stride[idx] == 2 else nbin
            nbinT = (nbinT << 1) - 1 if stride[-1 - idx] == 2 else nbinT

            self.encoder_rel.append(
                nn.Sequential(
                    ComplexConv2d(
                        in_channels=self.cnn_num[idx],
                        out_channels=self.cnn_num[idx + 1],
                        kernel_size=(3, 5),
                        padding=(0, 2),  # (k_h - 1)/2
                        stride=(1, stride[idx]),
                    ),
                    InstanceNorm(self.cnn_num[idx + 1] * nbin),
                    nn.PReLU(),
                )
            )

            self.encoder_mic.append(
                nn.Sequential(
                    ComplexConv2d(
                        in_channels=2 if idx == 0 else self.cnn_num[idx],
                        out_channels=self.cnn_num[idx + 1],
                        kernel_size=(1, 5),
                        padding=(0, 2),
                        stride=(1, stride[idx]),
                    ),
                    InstanceNorm(self.cnn_num[idx + 1] * nbin),
                    nn.PReLU(),
                )
            )

            if idx != n_cnn_layer - 1:
                self.decoder_l.append(
                    nn.Sequential(
                        ComplexGateConvTranspose2d(
                            in_channels=2 * self.cnn_num[-1 - idx],  # skip_connection
                            out_channels=self.cnn_num[-1 - idx - 1],
                            kernel_size=(1, 5),
                            padding=(0, 2),
                            stride=(1, stride[-1 - idx]),
                        ),
                        InstanceNorm(self.cnn_num[-1 - idx - 1] * nbinT),
                        nn.PReLU(),
                    )
                )
            else:
                self.decoder_l.append(
                    nn.Sequential(
                        ComplexGateConvTranspose2d(
                            in_channels=2 * self.cnn_num[-1 - idx],  # skip_connection
                            out_channels=2,
                            kernel_size=(1, 5),
                            padding=(0, 2),
                            stride=(1, stride[-1 - idx]),
                        ),
                    )
                )

        self.encoder_fusion = MS_CAM_F(inp_channels=self.cnn_num[-1], feature_size=nbin, r=1)

        self.rnns_r = nn.ModuleList(
            [
                FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
                FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
            ]
        )

        self.rnns_i = nn.ModuleList(
            [
                FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
                FTLSTM_RESNET(cnn_num[-1] // 2, rnn_hidden_num),
            ]
        )

    def unpack_state(self, state):
        """
        state: (state_r, state_i)

        state_r: [(t_h, t_c), (...), ...]
        """
        h_l, c_l = [], []

        for ri in state:
            for h, c in ri:
                h_l.append(h)
                c_l.append(c)

        t_h_out = torch.stack(h_l, dim=0)
        t_c_out = torch.stack(c_l, dim=0)
        return (t_h_out, t_c_out)

    def pack_state(self, state):
        """
        state: (4,1,17,64),(4,1,17,64)
        pack to [(h, c), (...), ...]
        """
        h_l, c_l = state
        state = []

        for i, (h, c) in enumerate(zip(h_l, c_l)):
            ele = (h, c)
            state.append(ele)

        state_r = state[:2]
        state_i = state[2:]

        return (state_r, state_i)

    def forward(self, specs_mic, specs_ref, *state):
        """
        inputs: shape is [B, 2, T, F]
        output: B,2,T,F
        """

        # print(len(args))
        state = self.pack_state(state)

        specs_mic_real, specs_mic_imag = specs_mic.chunk(2, dim=1)  # B,1,T,F
        specs_ref_real, specs_ref_imag = specs_ref.chunk(2, dim=1)

        specs_mix = torch.concat(
            [specs_mic_real, specs_ref_real, specs_mic_imag, specs_ref_imag], dim=1
        )  # [B, 4, T, F]

        spec_store = []
        spec = specs_mic
        # b,2,t,f
        x = specs_mix

        for idx, (lm, lr) in enumerate(zip(self.encoder_mic, self.encoder_rel)):
            spec = lm(spec)

            self.buff_x[idx] = torch.concat([self.buff_x[idx][..., 1:, :], x], dim=-2)
            # x = self.buff_x[idx]
            x = lr(self.buff_x[idx])  # x shape [B, C, T, F]
            spec_store.append(spec)

        x = self.encoder_fusion(x, spec)
        x_r, x_i = torch.chunk(x, 2, dim=1)

        state_r, state_i = [], []
        for idx, l in enumerate(self.rnns_r):
            x_r, stat = l(x_r, None if state is None else state[0][idx])
            state_r.append(stat)

        # state_r = torch.concat(state_r, dim=0)

        for idx, l in enumerate(self.rnns_i):
            x_i, stat = l(x_i, None if state is None else state[1][idx])
            state_i.append(stat)

        state = (state_r, state_i)
        # state = torch.stack([state_r, state_i], dim=0)
        # x_r = self.rnns_r(x_r)
        # x_i = self.rnns_i(x_i)

        x = torch.concatenate([x_r, x_i], dim=1)

        for idx, layer in enumerate(self.decoder_l):
            x = complex_cat([x, spec_store[-idx - 1]], dim=1)
            x = layer(x)

        feat_r, feat_i = complex_apply_mask(specs_mic, x)
        spec_enh = torch.concat([feat_r, feat_i], dim=1)

        return spec_enh, self.unpack_state(state)


def check_flops():
    from thop import profile
    import warnings

    net = MSA_DPCRN_SPEC(
        nframe=128,
        nhop=64,
        nfft=128,
        cnn_num=[16, 32, 64],
        stride=[2, 2, 1],
        rnn_hidden_num=64,
    )
    inp = torch.randn(1, 2, 10, 65)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="This API is being deprecated")
        flops, params = profile(net, inputs=(inp, inp), verbose=False)

    print(f"FLOPs={flops / 1e9}, params={params/1e6:.2f}")


def check_model(model1, model2):
    params1 = list(model1.parameters())
    params2 = list(model2.parameters())

    if len(params1) != len(params2):
        return False

    for param1, param2 in zip(params1, params2):
        if not torch.equal(param1.data, param2.data):
            return False

    return True


def verify_spec():
    from models.conv_stft import STFT

    stft = STFT(128, 64, 128)

    net = MSA_DPCRN(
        nframe=128,
        nhop=64,
        nfft=128,
        cnn_num=[16, 32, 64],
        stride=[2, 2, 1],
        rnn_hidden_num=64,
    )
    net.eval()

    net_spec = MSA_DPCRN_SPEC(
        nframe=128,
        nhop=64,
        nfft=128,
        cnn_num=[16, 32, 64],
        stride=[2, 2, 1],
        rnn_hidden_num=64,
    )
    net_spec.load_state_dict(net.state_dict())
    net_spec.eval()
    print(check_model(net, net_spec))

    mic = torch.randn(1, 1600)
    ref = torch.randn(1, 1600)
    xk_mic = stft.transform(mic)
    xk_ref = stft.transform(ref)
    t = stft.inverse(xk_ref)
    print(((t - ref) ** 2).sum())

    with torch.no_grad():
        out = net(mic, ref)
        out_spec = net_spec(xk_mic, xk_ref)
    out_spec = stft.inverse(out_spec)
    out_spec = torch.clamp(out_spec, -1, 1)

    print(out.shape, out_spec.shape, ((out - out_spec) ** 2).sum())


def verify_online():
    from tqdm import tqdm
    from models.conv_stft import STFT

    stft = STFT(128, 64, 128)

    net = MSA_DPCRN(
        nframe=128,
        nhop=64,
        nfft=128,
        cnn_num=[16, 32, 64],
        stride=[2, 2, 1],
        rnn_hidden_num=64,
    )
    net.eval()
    ckpt = torch.load("test_msadpcrn.pth")
    net.load_state_dict(ckpt)

    net_spec = MSA_DPCRN_SPEC_online(
        nframe=128,
        nhop=64,
        nfft=128,
        cnn_num=[16, 32, 64],
        stride=[2, 2, 1],
        rnn_hidden_num=64,
    )
    net_spec.load_state_dict(net.state_dict())
    net_spec.eval()
    print(check_model(net, net_spec))
    # torch.save(net.state_dict(),"test_msadpcrn.pth")

    mic = torch.ones(1, 1600)
    ref = torch.ones(1, 1600)
    xk_mic = stft.transform(mic)
    xk_ref = stft.transform(ref)

    with torch.no_grad():
        out = net(mic, ref)

    state = None
    out_list = []
    # for nt in tqdm(range(xk_mic.size(2)), leave=False, ncols=50):
    for nt in range(xk_mic.size(2)):
        # b,2,1,f
        mic_frame = xk_mic[..., (nt,), :]
        ref_frame = xk_ref[..., (nt,), :]

        with torch.no_grad():
            out_f, state = net_spec(mic_frame, ref_frame, state)

        # sys.exit() if nt == 1 else None

        # b,2,1,f
        out_list.append(out_f)

    out_spec = torch.concat(out_list, dim=2)
    out_spec = stft.inverse(out_spec)
    out_spec = torch.clamp(out_spec, -1, 1)

    print(out.shape, out_spec.shape, (torch.abs(out - out_spec)).sum())


def verify_onnx():
    from tqdm import tqdm
    from models.conv_stft import STFT

    stft = STFT(128, 64, 128)

    net = MSA_DPCRN(
        nframe=128,
        nhop=64,
        nfft=128,
        cnn_num=[16, 32, 64],
        stride=[2, 2, 1],
        rnn_hidden_num=64,
    )
    net.eval()
    ckpt = torch.load("test_msadpcrn.pth")
    net.load_state_dict(ckpt)

    net_spec = MSA_DPCRN_SPEC_onnx(
        # net_spec = MSA_DPCRN_SPEC_onnx(
        nframe=128,
        nhop=64,
        nfft=128,
        cnn_num=[16, 32, 64],
        stride=[2, 2, 1],
        rnn_hidden_num=64,
    )
    net_spec.load_state_dict(net.state_dict())
    net_spec.eval()
    print(check_model(net, net_spec))
    # torch.save(net.state_dict(),"test_msadpcrn.pth")

    mic = torch.ones(1, 160000)
    ref = torch.ones(1, 160000)
    xk_mic = stft.transform(mic)
    xk_ref = stft.transform(ref)

    with torch.no_grad():
        out = net(mic, ref)

    state = [
        torch.zeros([4, 1, 17, 64]).float(),
        torch.zeros([4, 1, 17, 64]).float(),
    ]
    out_list = []
    # for nt in tqdm(range(xk_mic.size(2)), leave=False, ncols=50):
    for nt in tqdm(range(xk_mic.size(2))):
        # b,2,1,f
        mic_frame = xk_mic[..., (nt,), :]
        ref_frame = xk_ref[..., (nt,), :]

        with torch.no_grad():
            out_f, state = net_spec(mic_frame, ref_frame, *state)

        # sys.exit() if nt == 1 else None

        # b,2,1,f
        out_list.append(out_f)

    out_spec = torch.concat(out_list, dim=2)
    out_spec = stft.inverse(out_spec)
    out_spec = torch.clamp(out_spec, -1, 1)

    print(out.shape, out_spec.shape, (torch.abs(out - out_spec)).sum())


if __name__ == "__main__":
    # verify_spec()
    # verify_online()
    verify_onnx()
    # check_flops()
