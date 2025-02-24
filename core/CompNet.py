from typing import List, Tuple, Union

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor
from torch.autograd import Variable

from JointNSHModel import HLModule
from utils.check_flops import check_flops
from utils.register import tables

torch_eps = torch.finfo(torch.float32).eps

# 320w160, 320-159-79-39-19-9
# frame_encoder_list = [159, 79, 39, 19, 9, 4]
# frame_decoder_list = [9, 19, 39, 79, 159, 320]
# frequency_encoder_list = [79, 39, 19, 9, 4]
# frequency_decoder_list = [9, 19, 39, 79, 161]

# 512w256
frame_encoder_list = [255, 127, 63, 31, 15, 8]
frame_decoder_list = [15, 31, 63, 127, 255, 512]
frequency_encoder_list = [127, 63, 31, 15, 8]
frequency_decoder_list = [15, 31, 63, 127, 257]


class MagEuclideanLoss(object):
    def __init__(self, l_type="L2"):
        self.l_type = l_type

    def __call__(self, esti, label: torch.Tensor):
        """
        esti: (B,T,F)
        label: (B,T,F)
        frame_list: list
        """
        b_size, seq_len, freq_num = esti.shape

        if self.l_type == "L1" or self.l_type == "l1":
            loss_mag = torch.abs(esti - label).mean()
        elif self.l_type == "L2" or self.l_type == "l2":
            loss_mag = torch.square(esti - label).mean()
        else:
            raise RuntimeError("only L1 and L2 are supported")
        return loss_mag


class ComMagEuclideanLoss(object):
    def __init__(self, alpha=0.5, l_type="L2"):
        self.alpha = alpha
        self.l_type = l_type

    def __call__(self, est, label: torch.Tensor):
        """
        est: (B,2,T,F)
        label: (B,2,T,F)
        frame_list: list
        alpha: scalar
        l_type: str, L1 or L2
        """
        b_size, _, seq_len, freq_num = est.shape

        est_mag, label_mag = torch.norm(est, dim=1), torch.norm(label, dim=1)

        if self.l_type == "L1" or self.l_type == "l1":
            loss_com = torch.abs(est - label).mean()
            loss_mag = torch.abs(est_mag - label_mag).mean()
        elif self.l_type == "L2" or self.l_type == "l2":
            loss_com = torch.square(est - label).mean()
            loss_mag = torch.square(est_mag - label_mag).mean()
        else:
            raise RuntimeError("only L1 and L2 are supported!")
        return self.alpha * loss_com + (1 - self.alpha) * loss_mag


class CumulativeLayerNorm2d(nn.Module):
    def __init__(
        self,
        frequency_num: int,
        channel_num: int,
        affine=True,
        eps=1e-5,
    ):
        super(CumulativeLayerNorm2d, self).__init__()
        self.frequency_num = frequency_num
        self.channel_num = channel_num
        self.eps = eps
        self.affine = affine

        if affine:
            self.gain = nn.Parameter(
                torch.ones(1, channel_num, 1, frequency_num), requires_grad=True
            )
            self.bias = nn.Parameter(
                torch.zeros(1, channel_num, 1, frequency_num), requires_grad=True
            )
        else:
            self.gain = Variable(torch.ones(1, channel_num, 1, frequency_num))
            self.bias = Variable(torch.zeros(1, channel_num, 1, frequency_num))

    def forward(self, inpt):
        """
        :param inpt: (B,C,T,F)
        :return:
        """
        b_size, channel, seq_len, freq_num = inpt.shape
        step_sum = inpt.sum([1, 3], keepdim=True)  # (B,1,T,1)
        step_pow_sum = inpt.pow(2).sum([1, 3], keepdim=True)  # (B,1,T,1)
        cum_sum = torch.cumsum(step_sum, dim=-2)  # (B,1,T,1)
        cum_pow_sum = torch.cumsum(step_pow_sum, dim=-2)  # (B,1,T,1)

        entry_cnt = np.arange(
            channel * freq_num, channel * freq_num * (seq_len + 1), channel * freq_num
        )
        entry_cnt = torch.from_numpy(entry_cnt).type(inpt.type())
        entry_cnt = entry_cnt.view(1, 1, seq_len, 1).expand_as(cum_sum)
        cum_mean = cum_sum / entry_cnt
        cum_var = (cum_pow_sum - 2 * cum_mean * cum_sum) / entry_cnt + cum_mean.pow(2)
        cum_std = (cum_var + self.eps).sqrt()

        x = (inpt - cum_mean) / cum_std
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(x.type())


class CumulativeLayerNorm1d(nn.Module):
    def __init__(
        self,
        channel_num,
        affine=True,
        eps=1e-5,
    ):
        super(CumulativeLayerNorm1d, self).__init__()
        self.channel_num = channel_num
        self.affine = affine
        self.eps = eps

        if affine:
            self.gain = nn.Parameter(torch.ones(1, channel_num, 1), requires_grad=True)
            self.bias = nn.Parameter(torch.zeros(1, channel_num, 1), requires_grad=True)
        else:
            self.gain = Variable(torch.ones(1, channel_num, 1))
            self.bias = Variable(torch.zeros(1, channel_num, 1))

    def forward(self, inpt):
        # inpt: (B,C,T)
        b_size, channel, seq_len = inpt.shape
        cum_sum = torch.cumsum(inpt.sum(1), dim=1)  # (B,T)
        cum_power_sum = torch.cumsum(inpt.pow(2).sum(1), dim=1)  # (B,T)

        entry_cnt = np.arange(channel, channel * (seq_len + 1), channel)
        entry_cnt = torch.from_numpy(entry_cnt).type(inpt.type())
        entry_cnt = entry_cnt.view(1, -1).expand_as(cum_sum)  # (B,T)

        cum_mean = cum_sum / entry_cnt  # (B,T)
        cum_var = (cum_power_sum - 2 * cum_mean * cum_sum) / entry_cnt + cum_mean.pow(2)
        cum_std = (cum_var + self.eps).sqrt()

        x = (inpt - cum_mean.unsqueeze(dim=1).expand_as(inpt)) / cum_std.unsqueeze(dim=1).expand_as(
            inpt
        )
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(x.type())


class InstantLayerNorm1d(nn.Module):
    def __init__(
        self,
        channel_num: int,
        affine: bool = True,
        eps=1e-5,
    ):
        super(InstantLayerNorm1d, self).__init__()
        self.channel_num = channel_num
        self.affine = affine
        self.eps = eps

        if affine:
            self.gain = nn.Parameter(torch.ones(1, channel_num, 1), requires_grad=True)
            self.bias = nn.Parameter(torch.zeros(1, channel_num, 1), requires_grad=True)
        else:
            self.gain = Variable(torch.ones(1, channel_num, 1))
            self.bias = Variable(torch.zeros(1, channel_num, 1))

    def forward(self, inpt):
        # inpt: (B,C,T)
        b_size, channel, seq_len = inpt.shape
        ins_mean = torch.mean(inpt, dim=1, keepdim=True)  # (B,1,T)
        ins_std = (torch.var(inpt, dim=1, keepdim=True) + self.eps).pow(0.5)  # (B,1,T)
        x = (inpt - ins_mean) / ins_std
        return x * self.gain.type(x.type()) + self.bias.type(x.type())


class InstantLayerNorm2d(nn.Module):
    def __init__(
        self,
        frequency_num: int,
        channel_num: int,
        affine: bool = True,
        eps=1e-5,
    ):
        super(InstantLayerNorm2d, self).__init__()
        self.frequency_num = frequency_num
        self.channel_num = channel_num
        self.affine = affine
        self.eps = eps

        if affine:
            self.gain = nn.Parameter(
                torch.ones(1, channel_num, 1, frequency_num), requires_grad=True
            )
            self.bias = nn.Parameter(
                torch.zeros(1, channel_num, 1, frequency_num), requires_grad=True
            )
        else:
            self.gain = Variable(torch.ones(1, channel_num, 1, frequency_num))
            self.bias = Variable(torch.zeros(1, channel_num, 1, frequency_num))

    def forward(self, inpt):
        # inpt: (B,C,T,F)
        ins_mean = torch.mean(inpt, dim=[1, 3], keepdim=True)  # (B,1,T,1)
        ins_std = (torch.std(inpt, dim=[1, 3], keepdim=True) + self.eps).pow(0.5)  # (B,1,T,1)
        x = (inpt - ins_mean) / ins_std
        return x * self.gain.type(x.type()) + self.bias.type(x.type())


class NormSwitch(nn.Module):
    def __init__(
        self,
        norm_type: str,
        format: str,
        channel_num: int,
        frequency_num: int = 0,
        affine: bool = True,
    ):
        super(NormSwitch, self).__init__()
        self.norm_type = norm_type
        self.format = format
        self.channel_num = channel_num
        self.frequency_num = frequency_num
        self.affine = affine

        if norm_type == "BN":
            if format in ["1D", "1d"]:
                self.norm = nn.BatchNorm1d(channel_num, affine=True)
            elif format in ["2D", "2d"]:
                self.norm = nn.BatchNorm2d(channel_num, affine=True)
        elif norm_type == "IN":
            if format in ["1D", "1d"]:
                self.norm = nn.InstanceNorm1d(channel_num, affine=True)
            elif format in ["2D", "2d"]:
                self.norm = nn.InstanceNorm2d(channel_num, affine=True)
        elif norm_type == "iLN":
            if format in ["1D", "1d"]:
                self.norm = InstantLayerNorm1d(channel_num, affine=True)
            elif format in ["2D", "2d"]:
                self.norm = InstantLayerNorm2d(frequency_num, channel_num, affine=True)
        elif norm_type == "cLN":
            if format in ["1D", "1d"]:
                self.norm = CumulativeLayerNorm1d(channel_num, affine=True)
            elif format in ["2D", "2d"]:
                self.norm = CumulativeLayerNorm2d(frequency_num, channel_num, affine=True)
        else:
            raise RuntimeError("Only BN, IN, iLN and cLN are supported currently")

    def forward(self, inpt):
        return self.norm(inpt)


class TorchSignalToFrames(object):
    def __init__(self, frame_size=320, frame_shift=160):
        super(TorchSignalToFrames, self).__init__()
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.pad = frame_size // 2

    def __call__(self, in_sig):
        in_sig = F.pad(in_sig, (self.pad, self.pad))
        sig_len = in_sig.shape[-1]
        # nframes = (sig_len // self.frame_shift)
        nframes = (sig_len - self.frame_size) // self.frame_shift + 1
        a = torch.zeros(tuple(in_sig.shape[:-1]) + (nframes, self.frame_size), device=in_sig.device)
        start = 0
        end = start + self.frame_size
        k = 0
        for i in range(nframes):
            if end < sig_len:
                a[..., i, :] = in_sig[..., start:end]
                k += 1
            else:
                tail_size = sig_len - start
                a[..., i, :tail_size] = in_sig[..., start:]

            start = start + self.frame_shift
            end = start + self.frame_size
        return a


class TorchOLA(nn.Module):
    r"""Overlap and add on gpu using torch tensor"""

    # Expects signal at last dimension
    def __init__(self, frame_shift=160):
        super(TorchOLA, self).__init__()
        self.frame_shift = frame_shift

    def forward(self, inputs):
        nframes = inputs.shape[-2]
        frame_size = inputs.shape[-1]
        frame_step = self.frame_shift
        sig_length = (nframes - 1) * frame_step + frame_size
        sig = torch.zeros(
            list(inputs.shape[:-2]) + [sig_length], dtype=inputs.dtype, device=inputs.device
        )
        ones = torch.zeros_like(sig)
        start = 0
        end = start + frame_size
        for i in range(nframes):
            sig[..., start:end] += inputs[..., i, :]
            ones[..., start:end] += 1.0
            start = start + frame_step
            end = start + frame_size
        return (sig / ones)[:, self.frame_shift : -self.frame_shift]


class InterIntraRNN(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        hidden_num: int,
        rnn_type: str,
        is_causal: bool,
    ):
        super(InterIntraRNN, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.hidden_num = hidden_num
        self.rnn_type = rnn_type
        self.is_causal = is_causal
        #
        self.norm = nn.LayerNorm([embed_dim])
        p = 2 if not is_causal else 1
        self.intra_rnn = getattr(nn, rnn_type)(
            input_size=embed_dim,
            hidden_size=hidden_dim // 2,
            num_layers=hidden_num // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.inter_rnn = getattr(nn, rnn_type)(
            input_size=hidden_dim,
            hidden_size=hidden_dim // p,
            num_layers=hidden_num // 2,
            batch_first=True,
            bidirectional=not is_causal,
        )
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1)
        )

    def forward(self, inpt):
        """
        inpt: (B, C, T, K)
        return:
                x: (B, T, K)
        """
        b_size, embed_dim, seq_len, k = inpt.shape
        inpt = self.norm(inpt.permute(0, 2, 3, 1))
        # intra part
        x = inpt.contiguous().view(b_size * seq_len, k, embed_dim)  # (BT, K, C)
        x, _ = self.intra_rnn(x)
        # inter part
        x = (
            x.view(b_size, seq_len, k, -1)
            .transpose(1, 2)
            .contiguous()
            .view(b_size * k, seq_len, -1)
        )
        x, _ = self.inter_rnn(x)
        x = self.ff(x).squeeze(dim=-1).view(b_size, k, seq_len).transpose(1, 2)
        return x


class FrameUNetEncoder(nn.Module):
    def __init__(self, cin: int, k1: tuple, c: int, norm_type: str):
        super(FrameUNetEncoder, self).__init__()
        self.cin = cin
        self.k1 = k1
        self.c = c
        self.norm_type = norm_type
        stride = (1, 2)
        global frame_encoder_list
        c_final = 32  # 64
        unet = []
        unet.append(
            nn.Sequential(
                GateConv2d(cin, c, k1, stride, padding=(0, 0, k1[0] - 1, 0)),
                NormSwitch(norm_type, "2D", c, frame_encoder_list[0], affine=True),
                nn.PReLU(c),
            )
        )
        unet.append(
            nn.Sequential(
                GateConv2d(c, c, k1, stride, padding=(0, 0, k1[0] - 1, 0)),
                NormSwitch(norm_type, "2D", c, frame_encoder_list[1], affine=True),
                nn.PReLU(c),
            )
        )
        unet.append(
            nn.Sequential(
                GateConv2d(c, c, k1, stride, padding=(0, 0, k1[0] - 1, 0)),
                NormSwitch(norm_type, "2D", c, frame_encoder_list[2], affine=True),
                nn.PReLU(c),
            )
        )
        unet.append(
            nn.Sequential(
                GateConv2d(c, c, k1, stride, padding=(0, 0, k1[0] - 1, 0)),
                NormSwitch(norm_type, "2D", c, frame_encoder_list[3], affine=True),
                nn.PReLU(c),
            )
        )
        unet.append(
            nn.Sequential(
                GateConv2d(c, c, k1, stride, padding=(0, 0, k1[0] - 1, 0)),
                NormSwitch(norm_type, "2D", c, frame_encoder_list[4], affine=True),
                nn.PReLU(c),
            )
        )
        unet.append(
            nn.Sequential(
                GateConv2d(c, c_final, k1, stride, padding=(1, 1, k1[0] - 1, 0)),
                NormSwitch(norm_type, "2D", c_final, frame_encoder_list[-1], affine=True),
                nn.PReLU(c_final),
            )
        )
        self.unet = nn.ModuleList(unet)

    def forward(self, x: Tensor) -> tuple:
        """
        x: (B, 1, T, F) or (B, T, F)
        """
        if x.ndim == 3:
            x = x.unsqueeze(dim=1)
        en_list = []
        for i in range(len(self.unet)):
            x = self.unet[i](x)
            en_list.append(x)
        return x, en_list


class FrameUNetDecoder(nn.Module):
    def __init__(
        self,
        c: int,
        embed_dim: int,
        k1: tuple,
        inter_connect: str,
        norm_type: str,
    ):
        super(FrameUNetDecoder, self).__init__()
        self.k1 = k1
        self.c = c
        self.embed_dim = embed_dim
        self.inter_connect = inter_connect
        self.norm_type = norm_type
        c_begin = 32  # 64
        stride = (1, 2)
        global frame_decoder_list
        unet = []
        base_num = 2 if inter_connect == "cat" else 1
        unet.append(
            nn.Sequential(
                GateConvTranspose2d(c_begin * base_num, c, k1, stride, chomp_f=(1, 1)),
                NormSwitch(norm_type, "2D", channel_num=c, frequency_num=frame_decoder_list[0]),
                nn.PReLU(c),
            )
        )
        unet.append(
            nn.Sequential(
                GateConvTranspose2d(c * base_num, c, k1, stride),
                NormSwitch(norm_type, "2D", channel_num=c, frequency_num=frame_decoder_list[1]),
                nn.PReLU(c),
            )
        )
        unet.append(
            nn.Sequential(
                GateConvTranspose2d(c * base_num, c, k1, stride),
                NormSwitch(norm_type, "2D", channel_num=c, frequency_num=frame_decoder_list[2]),
                nn.PReLU(c),
            )
        )
        unet.append(
            nn.Sequential(
                GateConvTranspose2d(c * base_num, c, k1, stride),
                NormSwitch(norm_type, "2D", channel_num=c, frequency_num=frame_decoder_list[3]),
                nn.PReLU(c),
            )
        )
        unet.append(
            nn.Sequential(
                GateConvTranspose2d(c * base_num, c, k1, stride),
                NormSwitch(norm_type, "2D", channel_num=c, frequency_num=frame_decoder_list[4]),
                nn.PReLU(c),
            )
        )
        unet.append(
            nn.Sequential(
                GateConvTranspose2d(c * base_num, embed_dim, k1, stride),
                nn.ConstantPad2d((1, 0, 0, 0), value=0.0),
            )
        )
        self.unet_list = nn.ModuleList(unet)

    def forward(self, x: Tensor, en_list: list) -> Tensor:
        if self.inter_connect == "cat":
            for i in range(len(self.unet_list)):
                tmp = torch.cat((x, en_list[-(i + 1)]), dim=1)
                x = self.unet_list[i](tmp)
        elif self.inter_connect == "add":
            for i in range(len(self.unet_list)):
                tmp = x + en_list[-(i + 1)]
                x = self.unet_list[i](tmp)
        return x


class FreqUNetEncoder(nn.Module):
    def __init__(
        self,
        cin: int,
        k1: tuple,
        c: int,
        norm_type: str,
    ):
        super(FreqUNetEncoder, self).__init__()
        self.cin = cin
        self.k1 = k1
        self.c = c
        self.norm_type = norm_type
        kernel_begin = (k1[0], 5)
        stride = (1, 2)
        c_final = 64
        global frequency_encoder_list
        unet = []
        unet.append(
            nn.Sequential(
                GateConv2d(cin, c, kernel_begin, stride, padding=(0, 0, k1[0] - 1, 0)),
                NormSwitch(norm_type, "2D", c, frequency_encoder_list[0], affine=True),
                nn.PReLU(c),
            )
        )
        unet.append(
            nn.Sequential(
                GateConv2d(c, c, k1, stride, padding=(0, 0, k1[0] - 1, 0)),
                NormSwitch(norm_type, "2D", c, frequency_encoder_list[1], affine=True),
                nn.PReLU(c),
            )
        )
        unet.append(
            nn.Sequential(
                GateConv2d(c, c, k1, stride, padding=(0, 0, k1[0] - 1, 0)),
                NormSwitch(norm_type, "2D", c, frequency_encoder_list[2], affine=True),
                nn.PReLU(c),
            )
        )
        unet.append(
            nn.Sequential(
                GateConv2d(c, c, k1, stride, padding=(0, 0, k1[0] - 1, 0)),
                NormSwitch(norm_type, "2D", c, frequency_encoder_list[3], affine=True),
                nn.PReLU(c),
            )
        )
        unet.append(
            nn.Sequential(
                GateConv2d(c, c_final, k1, (1, 2), padding=(0, 0, k1[0] - 1, 0)),
                NormSwitch(norm_type, "2D", c_final, frequency_encoder_list[-1], affine=True),
                nn.PReLU(c_final),
            )
        )
        self.unet_list = nn.ModuleList(unet)

    def forward(self, x: Tensor) -> tuple:
        en_list = []
        for i in range(len(self.unet_list)):
            x = self.unet_list[i](x)
            en_list.append(x)
        return x, en_list


class FreqUNetDecoder(nn.Module):
    def __init__(
        self,
        c: int,
        k1: tuple,
        embed_dim: int,
        inter_connect: str,
        norm_type: str,
    ):
        super(FreqUNetDecoder, self).__init__()
        self.k1 = k1
        self.c = c
        self.embed_dim = embed_dim
        self.inter_connect = inter_connect
        self.norm_type = norm_type
        c_begin = 64
        kernel_end = (k1[0], 5)
        stride = (1, 2)
        global frequency_decoder_list
        unet = []
        base_num = 2 if inter_connect == "add" else 1
        unet.append(
            nn.Sequential(
                GateConvTranspose2d(c_begin * base_num, c, k1, stride),
                NormSwitch(norm_type, "2D", channel_num=c, frequency_num=frequency_decoder_list[0]),
                nn.PReLU(c),
            )
        )
        unet.append(
            nn.Sequential(
                GateConvTranspose2d(c * base_num, c, k1, stride),
                NormSwitch(norm_type, "2D", channel_num=c, frequency_num=frequency_decoder_list[1]),
                nn.PReLU(c),
            )
        )
        unet.append(
            nn.Sequential(
                GateConvTranspose2d(c * base_num, c, k1, stride),
                NormSwitch(norm_type, "2D", channel_num=c, frequency_num=frequency_decoder_list[2]),
                nn.PReLU(c),
            )
        )
        unet.append(
            nn.Sequential(
                GateConvTranspose2d(c * base_num, c, k1, stride),
                NormSwitch(norm_type, "2D", channel_num=c, frequency_num=frequency_decoder_list[3]),
                nn.PReLU(c),
            )
        )
        unet.append(
            nn.Sequential(
                GateConvTranspose2d(c * base_num, embed_dim, kernel_end, stride),
                NormSwitch(
                    norm_type, "2D", channel_num=c, frequency_num=frequency_decoder_list[-1]
                ),
                nn.PReLU(embed_dim),
                nn.Conv2d(embed_dim, embed_dim, (1, 1), (1, 1)),
            )
        )
        self.unet_list = nn.ModuleList(unet)

    def forward(self, x: Tensor, en_list: list) -> Tensor:
        if self.inter_connect == "cat":
            for i in range(len(self.unet_list)):
                tmp = torch.cat((x, en_list[-(i + 1)]), dim=1)
                x = self.unet_list[i](tmp)
        elif self.inter_connect == "add":
            for i in range(len(self.unet_list)):
                tmp = x + en_list[-(i + 1)]
                x = self.unet_list[i](tmp)
        return x


class FreqU2NetEncoder(nn.Module):
    def __init__(
        self,
        cin: int,
        k1: tuple,
        k2: tuple,
        c: int,
        intra_connect: str,
        norm_type: str,
    ):
        super(FreqU2NetEncoder, self).__init__()
        self.cin = cin
        self.k1 = k1
        self.k2 = k2
        self.c = c
        self.intra_connect = intra_connect
        self.norm_type = norm_type
        c_last = 32  # 64
        kernel_begin = (k1[0], 5)
        stride = (1, 2)
        global frequency_encoder_list
        meta_unet = []
        meta_unet.append(
            EnUnetModule(
                cin,
                c,
                kernel_begin,
                k2,
                stride,
                intra_connect,
                norm_type,
                scale=4,
                padding=(0, 0, k1[0] - 1, 0),
                de_flag=False,
            )
        )
        meta_unet.append(
            EnUnetModule(
                c,
                c,
                k1,
                k2,
                stride,
                intra_connect,
                norm_type,
                scale=3,
                padding=(0, 0, k1[0] - 1, 0),
                de_flag=False,
            )
        )
        meta_unet.append(
            EnUnetModule(
                c,
                c,
                k1,
                k2,
                stride,
                intra_connect,
                norm_type,
                scale=2,
                padding=(0, 0, k1[0] - 1, 0),
                de_flag=False,
            )
        )
        meta_unet.append(
            EnUnetModule(
                c,
                c,
                k1,
                k2,
                stride,
                intra_connect,
                norm_type,
                scale=1,
                padding=(0, 0, k1[0] - 1, 0),
                de_flag=False,
            )
        )
        self.meta_unet_list = nn.ModuleList(meta_unet)
        self.last_conv = nn.Sequential(
            GateConv2d(c, c_last, k1, stride, (1, 1, k1[0] - 1, 0)),
            NormSwitch(norm_type, "2D", c_last, frequency_num=frequency_encoder_list[-1]),
            nn.PReLU(c_last),
        )

    def forward(self, x: Tensor) -> tuple:
        en_list = []
        for i in range(len(self.meta_unet_list)):
            x = self.meta_unet_list[i](x)
            en_list.append(x)

        x = self.last_conv(x)
        en_list.append(x)
        return x, en_list


class FreqU2NetDecoder(nn.Module):
    def __init__(
        self,
        c: int,
        k1: tuple,
        k2: tuple,
        embed_dim: int,
        intra_connect: str,
        inter_connect: str,
        norm_type: str,
    ):
        super(FreqU2NetDecoder, self).__init__()
        self.c = c
        self.k1 = k1
        self.k2 = k2
        self.embed_dim = embed_dim
        self.intra_connect = intra_connect
        self.inter_connect = inter_connect
        self.norm_type = norm_type
        c_begin = 64
        kernel_end = (k1[0], 5)
        stride = (1, 2)
        global frequency_decoder_list
        meta_unet = []
        base_num = 2 if inter_connect == "cat" else 1
        meta_unet.append(
            EnUnetModule(
                c_begin * base_num,
                c,
                k1,
                k2,
                stride,
                intra_connect,
                norm_type,
                scale=1,
                de_flag=True,
            )
        )
        meta_unet.append(
            EnUnetModule(
                c * base_num, c, k1, k2, stride, intra_connect, norm_type, scale=2, de_flag=True
            )
        )
        meta_unet.append(
            EnUnetModule(
                c * base_num, c, k1, k2, stride, intra_connect, norm_type, scale=3, de_flag=True
            )
        )
        meta_unet.append(
            EnUnetModule(
                c * base_num, c, k1, k2, stride, intra_connect, norm_type, scale=4, de_flag=True
            )
        )
        self.meta_unet_list = nn.ModuleList(meta_unet)
        self.last_conv = nn.Sequential(
            GateConvTranspose2d(c * base_num, embed_dim, kernel_end, stride),
            NormSwitch(norm_type, "2D", embed_dim, frequency_decoder_list[-1], affine=True),
            nn.PReLU(embed_dim),
            nn.Conv2d(embed_dim, embed_dim, (1, 1), (1, 1)),
        )

    def forward(self, x: Tensor, en_list: list) -> Tensor:
        if self.inter_connect == "add":
            for i in range(len(self.meta_unet_list)):
                tmp = x + en_list[-(i + 1)]
                x = self.meta_unet_list[i](tmp)
            x = x + en_list[0]
            x = self.last_conv(x)
        elif self.inter_connect == "cat":
            for i in range(len(self.meta_unet_list)):
                tmp = torch.cat((x, en_list[-(i + 1)]), dim=1)
                x = self.meta_unet_list[i](tmp)
            x = torch.cat((x, en_list[0]), dim=1)
            x = self.last_conv(x)
        return x


class EnUnetModule(nn.Module):
    def __init__(
        self,
        cin: int,
        cout: int,
        k1: tuple,
        k2: tuple,
        stride: tuple,
        intra_connect: str,
        norm_type: str,
        scale: int,
        padding: tuple = (0, 0, 0, 0),
        chomp: tuple = (0, 0),  # only in the freq-axis
        de_flag: bool = False,
    ):
        super(EnUnetModule, self).__init__()
        self.cin = cin
        self.cout = cout
        self.k1 = k1
        self.k2 = k2
        self.stride = stride
        self.padding = padding
        self.chomp = chomp
        self.intra_connect = intra_connect
        self.norm_type = norm_type
        self.scale = scale
        self.de_flag = de_flag

        global frequency_encoder_list, frequency_decoder_list

        in_conv_list = []
        if not de_flag:
            in_conv_list.append(GateConv2d(cin, cout, k1, stride, padding))
            in_conv_list.append(
                NormSwitch(
                    norm_type,
                    "2D",
                    channel_num=cout,
                    frequency_num=frequency_encoder_list[len(frequency_encoder_list) - scale - 1],
                )
            )
        else:
            in_conv_list.append(GateConvTranspose2d(cin, cout, k1, stride, chomp))
            in_conv_list.append(
                NormSwitch(
                    norm_type,
                    "2D",
                    channel_num=cout,
                    frequency_num=frequency_decoder_list[scale - 1],
                )
            )
        in_conv_list.append(nn.PReLU(cout))
        self.in_conv = nn.Sequential(*in_conv_list)

        enco_list, deco_list = [], []
        for i in range(scale):
            pad = (1, 0, 0, 0) if (scale - 1) == i else (0, 0, 0, 0)
            enco_list.append(
                Conv2dunit(
                    k2,
                    cout,
                    frequency_encoder_list[len(frequency_encoder_list) - scale + i],
                    norm_type,
                    pad=pad,
                )
            )
        for i in range(scale):
            if i == 0:
                deco_list.append(
                    Deconv2dunit(
                        k2, cout, frequency_decoder_list[i], "add", norm_type, chomp_f=(1, 1)
                    )
                )
            else:
                deco_list.append(
                    Deconv2dunit(k2, cout, frequency_decoder_list[i], intra_connect, norm_type)
                )

        self.enco = nn.ModuleList(enco_list)
        self.deco = nn.ModuleList(deco_list)
        self.skip_connect = SkipConnect(intra_connect)

    def forward(self, inputs: Tensor) -> Tensor:
        x_resi = self.in_conv(inputs)
        x = x_resi
        x_list = []
        for i in range(len(self.enco)):
            x = self.enco[i](x)
            x_list.append(x)

        for i in range(len(self.deco)):
            if i == 0:
                x = self.deco[i](x)
            else:
                x_con = self.skip_connect(x, x_list[-(i + 1)])
                x = self.deco[i](x_con)
        x_resi = x_resi + x
        del x_list
        return x_resi


class Conv2dunit(nn.Module):
    def __init__(self, k: tuple, c: int, freq: int, norm_type: str, pad: tuple = (0, 0, 0, 0)):
        super(Conv2dunit, self).__init__()
        self.k = k
        self.c = c
        self.freq = freq
        self.norm_type = norm_type
        self.pad = pad

        k_t = k[0]
        stride = (1, 2)
        self.conv = nn.Sequential(
            nn.ConstantPad2d((pad[0], pad[0], k_t - 1, 0), value=0.0),
            nn.Conv2d(c, c, k, stride),
            NormSwitch(norm_type, "2D", channel_num=c, frequency_num=freq, affine=True),
            nn.PReLU(c),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


class Deconv2dunit(nn.Module):
    def __init__(
        self,
        k: tuple,
        c: int,
        freq: int,
        intra_connect: str,
        norm_type: str,
        chomp_f: tuple = (0, 0),
    ):
        super(Deconv2dunit, self).__init__()
        self.k = k
        self.c = c
        self.freq = freq
        self.intra_connect = intra_connect
        self.norm_type = norm_type
        self.chomp_f = chomp_f

        k_t = k[0]
        stride = (1, 2)
        deconv_list = []
        if self.intra_connect == "add":
            real_c = c
        else:
            real_c = c * 2
        deconv_list.append(nn.ConvTranspose2d(real_c, c, k, stride))
        if k_t > 1:
            deconv_list.append(ChompT(k_t - 1))
        deconv_list.append(ChompF(chomp_f[0], chomp_f[1]))
        deconv_list.append(
            NormSwitch(norm_type, "2D", channel_num=c, frequency_num=freq, affine=True)
        )
        deconv_list.append(nn.PReLU(c))
        self.deconv = nn.Sequential(*deconv_list)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(dim=1)
        return self.deconv(inputs)


class GateConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple,
        padding: tuple,
    ):
        super(GateConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv = nn.Sequential(
            nn.ConstantPad2d(padding, value=0.0),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels * 2,
                kernel_size=kernel_size,
                stride=stride,
            ),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(dim=1)
        x = self.conv(inputs)
        outputs, gate = x.chunk(2, dim=1)
        return outputs * gate.sigmoid()


class GateConvTranspose2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple,
        chomp_f: tuple = (0, 0),
    ):
        super(GateConvTranspose2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.chomp_f = chomp_f

        k_t = kernel_size[0]
        if k_t > 1:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels * 2,
                    kernel_size=kernel_size,
                    stride=stride,
                ),
                ChompT(k_t - 1),
                ChompF(chomp_f[0], chomp_f[-1]),
            )
        else:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels * 2,
                    kernel_size=kernel_size,
                    stride=stride,
                ),
                ChompF(chomp_f[0], chomp_f[-1]),
            )

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(dim=1)
        x = self.conv(inputs)
        outputs, gate = x.chunk(2, dim=1)
        return outputs * gate.sigmoid()


class SkipConnect(nn.Module):
    def __init__(self, connect):
        super(SkipConnect, self).__init__()
        self.connect = connect

    def forward(self, x_main, x_aux):
        if self.connect == "add":
            x = x_main + x_aux
        elif self.connect == "cat":
            x = torch.cat((x_main, x_aux), dim=1)
        return x


class TCMList(nn.Module):
    def __init__(
        self,
        kd1: int,
        cd1: int,
        d_feat: int,
        norm_type: str,
        dilations: tuple = (1, 2, 5, 9),
        is_causal: bool = True,
    ):
        super(TCMList, self).__init__()
        self.kd1 = kd1
        self.cd1 = cd1
        self.d_feat = d_feat
        self.norm_type = norm_type
        self.dilations = dilations
        self.is_causal = is_causal
        tcm_list = []
        for i in range(len(dilations)):
            tcm_list.append(
                SqueezedTCM(
                    kd1,
                    cd1,
                    dilation=dilations[i],
                    d_feat=d_feat,
                    norm_type=norm_type,
                    is_causal=is_causal,
                )
            )
        self.tcm_list = nn.ModuleList(tcm_list)

    def forward(self, inputs: Tensor) -> Tensor:
        x = inputs
        for i in range(len(self.dilations)):
            x = self.tcm_list[i](x)
        return x


class SqueezedTCM(nn.Module):
    def __init__(
        self,
        kd1: int,
        cd1: int,
        dilation: int,
        d_feat: int,
        norm_type: str,
        is_causal: bool = True,
    ):
        super(SqueezedTCM, self).__init__()
        self.kd1 = kd1
        self.cd1 = cd1
        self.dilation = dilation
        self.d_feat = d_feat
        self.norm_type = norm_type
        self.is_causal = is_causal

        self.in_conv = nn.Conv1d(d_feat, cd1, kernel_size=1, bias=False)
        if is_causal:
            pad = ((kd1 - 1) * dilation, 0)
        else:
            pad = ((kd1 - 1) * dilation // 2, (kd1 - 1) * dilation // 2)
        self.left_conv = nn.Sequential(
            nn.PReLU(cd1),
            NormSwitch(norm_type, "1D", cd1, affine=True),
            nn.ConstantPad1d(pad, value=0.0),
            nn.Conv1d(cd1, cd1, kernel_size=kd1, dilation=dilation, bias=False),
        )
        self.right_conv = nn.Sequential(
            nn.PReLU(cd1),
            NormSwitch(norm_type, "1D", cd1, affine=True),
            nn.ConstantPad1d(pad, value=0.0),
            nn.Conv1d(cd1, cd1, kernel_size=kd1, dilation=dilation, bias=False),
            nn.Sigmoid(),
        )
        self.out_conv = nn.Sequential(
            nn.PReLU(cd1),
            NormSwitch(norm_type, "1D", cd1, affine=True),
            nn.Conv1d(cd1, d_feat, kernel_size=1, bias=False),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        resi = inputs
        x = self.in_conv(inputs)
        x = self.left_conv(x) * self.right_conv(x)
        x = self.out_conv(x)
        x = x + resi
        return x


class ChompT(nn.Module):
    def __init__(
        self,
        t: int,
    ):
        super(ChompT, self).__init__()
        self.t = t

    def forward(self, x: Tensor) -> Tensor:
        return x[..., : -self.t, :]


class ChompF(nn.Module):
    def __init__(
        self,
        front_f: int = 0,
        end_f: int = 0,
    ):
        super(ChompF, self).__init__()
        self.front_f = front_f
        self.end_f = end_f

    def forward(self, x):
        if self.end_f != 0:
            return x[..., self.front_f : -self.end_f]
        else:
            return x[..., self.front_f :]


class CollaborativePostProcessing(nn.Module):
    def __init__(
        self,
        cin: int,
        k1: Union[Tuple, List],
        k2: Union[Tuple, List],
        c: int,
        kd1: int,
        cd1: int,
        d_feat: int,
        fft_num: int,
        dilations: Union[Tuple, List],
        intra_connect: str,
        norm_type: str,
        is_causal: bool,
        is_u2: bool,
        group_num: int = 2,
    ):
        super(CollaborativePostProcessing, self).__init__()
        self.cin = cin
        self.k1 = k1
        self.k2 = k2
        self.c = c
        self.kd1 = kd1
        self.cd1 = cd1
        self.d_feat = d_feat
        self.fft_num = fft_num
        self.dilations = dilations
        self.intra_connect = intra_connect
        self.norm_type = norm_type
        self.is_causal = is_causal
        self.is_u2 = is_u2
        self.group_num = group_num
        #
        freq_dim = fft_num // 2 + 1
        if self.is_u2:
            self.en = FreqU2NetEncoder(
                cin=cin, k1=k1, k2=k2, c=c, intra_connect=intra_connect, norm_type=norm_type
            )
        else:
            self.en = FreqUNetEncoder(cin=cin, k1=k1, c=c, norm_type=norm_type)
        self.gain_in_conv = nn.Conv1d(d_feat + freq_dim, d_feat, kernel_size=1)
        self.resi_in_conv = nn.Conv1d(d_feat + freq_dim * 2, d_feat, kernel_size=1)
        stcns_gain, stcns_resi = [], []
        for i in range(group_num):
            stcns_gain.append(TCMList(kd1, cd1, d_feat, norm_type, dilations, is_causal))
            stcns_resi.append(TCMList(kd1, cd1, d_feat, norm_type, dilations, is_causal))
        self.stcns_gain, self.stcns_resi = nn.ModuleList(stcns_gain), nn.ModuleList(stcns_resi)
        self.gain_out_conv = nn.Sequential(nn.Conv1d(d_feat, freq_dim, kernel_size=1), nn.Sigmoid())
        self.resi_out_conv = nn.Conv1d(d_feat, freq_dim * 2, kernel_size=1)
        #
        self.real_rnn = nn.GRU(
            input_size=freq_dim * 2, hidden_size=freq_dim, num_layers=1, batch_first=True
        )
        self.imag_rnn = nn.GRU(
            input_size=freq_dim * 2, hidden_size=freq_dim, num_layers=1, batch_first=True
        )
        self.real_decode = nn.Linear(freq_dim, freq_dim, bias=False)
        self.imag_decode = nn.Linear(freq_dim, freq_dim, bias=False)

    def forward(self, comp_mag: Tensor, comp_phase: Tensor) -> Tensor:
        """
        comp_mag: (B, 2, T, F)
        comp_phase: (B, 2, T, F)
        """
        inpt_x = torch.cat((comp_mag, comp_phase), dim=1)
        en_x, _ = self.en(inpt_x)
        b_size, c, seq_len, f = en_x.shape
        en_x = en_x.transpose(-2, -1).contiguous().view(b_size, -1, seq_len)
        # gain branch
        gain_branch_inpt = torch.cat((en_x, torch.norm(comp_mag, dim=1).transpose(-2, -1)), dim=1)
        gain_branch_x = self.gain_in_conv(gain_branch_inpt)
        # resi branch
        resi_branch_inpt = torch.cat(
            (en_x, comp_phase.transpose(-2, -1).contiguous().view(b_size, -1, seq_len)), dim=1
        )
        resi_branch_x = self.resi_in_conv(resi_branch_inpt)
        #
        gain_x, resi_x = gain_branch_x.clone(), resi_branch_x.clone()
        for i in range(self.group_num):
            gain_x = self.stcns_gain[i](gain_x)
            resi_x = self.stcns_resi[i](resi_x)
        gain = self.gain_out_conv(gain_x).transpose(-2, -1)
        com_resi = self.resi_out_conv(resi_x)
        resi_r, resi_i = torch.chunk(com_resi, 2, dim=1)
        resi = torch.stack((resi_r, resi_i), dim=1).transpose(-2, -1)
        # collaboratively recover
        comp_phase = comp_phase * gain.unsqueeze(dim=1)
        comp_mag = comp_mag + resi
        # fusion
        real_x = torch.cat((comp_phase[:, 0, ...], comp_mag[:, 0, ...]), dim=-1)
        # print("real_x.shape", real_x.shape)
        imag_x = torch.cat((comp_phase[:, -1, ...], comp_mag[:, -1, ...]), dim=-1)
        # print("imag_x.shape", imag_x.shape)

        real_x, imag_x = self.real_decode(self.real_rnn(real_x)[0]), self.imag_decode(
            self.imag_rnn(imag_x)[0]
        )
        return torch.stack((real_x, imag_x), dim=1)


class VanillaPostProcessing(nn.Module):
    def __init__(
        self,
        cin: int,
        k1: Union[Tuple, List],
        k2: Union[Tuple, List],
        c: int,
        kd1: int,
        cd1: int,
        d_feat: int,
        fft_num: int,
        dilations: Union[Tuple, List],
        intra_connect: str,
        norm_type: str,
        is_causal: bool,
        is_u2: bool,
        group_num: int = 2,
    ):
        super(VanillaPostProcessing, self).__init__()
        self.cin = cin
        self.k1 = k1
        self.k2 = k2
        self.c = c
        self.kd1 = kd1
        self.cd1 = cd1
        self.d_feat = d_feat
        self.fft_num = fft_num
        self.dilations = dilations
        self.intra_connect = intra_connect
        self.norm_type = norm_type
        self.is_causal = is_causal
        self.is_u2 = is_u2
        self.group_num = group_num
        #
        freq_dim = fft_num // 2 + 1
        if self.is_u2:
            self.en = FreqU2NetEncoder(
                cin=cin, k1=k1, k2=k2, c=c, intra_connect=intra_connect, norm_type=norm_type
            )
        else:
            self.en = FreqUNetEncoder(cin=cin, k1=k1, c=c, norm_type=norm_type)
        self.in_conv = nn.Conv1d(d_feat + freq_dim * 4, d_feat, kernel_size=1)
        stcns = []
        for i in range(group_num):
            stcns.append(TCMList(kd1, cd1, d_feat, norm_type, dilations, is_causal))
        self.stcns = nn.ModuleList(stcns)
        self.real_conv, self.imag_conv = nn.Conv1d(d_feat, freq_dim, kernel_size=1), nn.Conv1d(
            d_feat, freq_dim, kernel_size=1
        )

    def forward(self, comp_mag: Tensor, comp_phase: Tensor) -> Tensor:
        """
        comp_mag: (B, 2, T, F)
        comp_phase: (B, 2, T, F)
        return:
                (B, 2, T, F)
        """
        inpt_x = torch.cat((comp_mag, comp_phase), dim=1)
        en_x, _ = self.en(inpt_x)
        b_size, c, seq_len, f = en_x.shape
        en_x = en_x.transpose(-2, -1).contiguous().view(b_size, -1, seq_len)
        en_x = torch.cat(
            (
                en_x,
                comp_mag.transpose(-2, -1).contiguous().view(b_size, -1, seq_len),
                comp_phase.transpose(-2, -1).contiguous().view(b_size, -1, seq_len),
            ),
            dim=1,
        )
        x = self.in_conv(en_x)
        x_acc = Variable(torch.ones_like(x, device=x.device), requires_grad=True)
        for i in range(self.group_num):
            x = self.stcns[i](x)
            x_acc = x_acc + x
        x_real, x_imag = self.real_conv(x_acc).transpose(-2, -1), self.imag_conv(x_acc).transpose(
            -2, -1
        )
        return torch.stack((x_real, x_imag), dim=1)


class TCNN(nn.Module):
    def __init__(
        self,
        cin: int,
        k1: Union[Tuple, List],
        c: int,
        embed_dim: int,
        kd1: int,
        cd1: int,
        d_feat: int,
        hidden_dim: int,
        hidden_num: int,
        group_num: int,
        dilations: Union[Tuple, List],
        inter_connect: str,
        norm_type: str,
        rnn_type: str,
        is_dual_rnn: bool,
        is_causal: bool = True,
    ):
        super(TCNN, self).__init__()
        self.cin = cin
        self.k1 = tuple(k1)
        self.c = c
        self.embed_dim = embed_dim
        self.kd1 = kd1
        self.cd1 = cd1
        self.d_feat = d_feat
        self.hidden_dim = hidden_dim
        self.hidden_num = hidden_num
        self.group_num = group_num
        self.dilations = dilations
        self.inter_connect = inter_connect
        self.norm_type = norm_type
        self.rnn_type = rnn_type
        self.is_dual_rnn = is_dual_rnn
        self.is_causal = is_causal
        #
        self.en = FrameUNetEncoder(cin=cin, k1=k1, c=c, norm_type=norm_type)
        self.de = FrameUNetDecoder(
            c=c, embed_dim=embed_dim, k1=k1, inter_connect=inter_connect, norm_type=norm_type
        )
        stcns = []
        for i in range(group_num):
            stcns.append(TCMList(kd1, cd1, d_feat, norm_type, dilations, is_causal))
        self.stcns = nn.ModuleList(stcns)
        if is_dual_rnn:
            self.dual_rnn = InterIntraRNN(
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                hidden_num=hidden_num,
                rnn_type=rnn_type,
                is_causal=is_causal,
            )
        else:
            assert embed_dim == 1, "the embed_dim should be 1 if no dual-rnn is adopted!"

    def forward(self, inpt: Tensor) -> Tensor:
        """
        inpt: (B, T, K) or (B, 1, T, K)
        return:
                (B, T, K)
        """
        if inpt.ndim == 3:
            inpt = inpt.unsqueeze(dim=1)
        en_x, en_list = self.en(inpt)
        b_size, c, seq_len, k = en_x.shape
        x = en_x.transpose(-2, -1).contiguous().view(b_size, -1, seq_len)
        x_acc = Variable(torch.zeros_like(x, device=x.device), requires_grad=True)
        for i in range(self.group_num):
            x = self.stcns[i](x)
            x_acc = x_acc + x
        x = x_acc.clone()
        x = x.view(b_size, c, k, seq_len).transpose(-2, -1)
        embed_x = self.de(x, en_list)
        #
        if self.is_dual_rnn:
            x = self.dual_rnn(embed_x)
        else:
            x = embed_x.squeeze(dim=1)
        return x


class CompNet(nn.Module):
    """
    CompNet: Complementary network for single-channel speech enhancement
    """

    def __init__(
        self,
        win_size: int = 320,
        win_shift: int = 160,
        fft_num: int = 320,
        k1: Union[Tuple, List] = (2, 3),
        k2: Union[Tuple, List] = (2, 3),
        c: int = 64,
        embed_dim: int = 64,
        kd1: int = 5,
        cd1: int = 64,
        d_feat: int = 256,
        hidden_dim: int = 64,
        hidden_num: int = 2,
        group_num: int = 2,
        dilations: Union[Tuple, List] = (1, 2, 5, 9),
        inter_connect: str = "cat",
        intra_connect: str = "cat",
        norm_type: str = "iLN",
        rnn_type: str = "LSTM",
        post_type: str = "collaborative",
        is_dual_rnn: bool = True,
        is_causal: bool = True,
        is_u2: bool = True,
        is_mu_compress: bool = True,
    ):
        super(CompNet, self).__init__()
        self.win_size = win_size
        self.win_shift = win_shift
        self.fft_num = fft_num
        self.k1 = tuple(k1)
        self.k2 = tuple(k2)
        self.c = c
        self.embed_dim = embed_dim
        self.kd1 = kd1
        self.cd1 = cd1
        self.d_feat = d_feat
        self.hidden_dim = hidden_dim
        self.hidden_num = hidden_num
        self.group_num = group_num
        self.dilations = tuple(dilations)
        self.inter_connect = inter_connect
        self.intra_connect = intra_connect
        self.norm_type = norm_type
        self.rnn_type = rnn_type
        self.post_type = post_type
        self.is_dual_rnn = is_dual_rnn
        self.is_causal = is_causal
        self.is_u2 = is_u2
        self.is_mu_compress = is_mu_compress
        #
        # first simultaneously pre-estimate mag and phase
        pre_kwargs = {
            "k1": k1,
            "c": c,
            "embed_dim": embed_dim,
            "kd1": kd1,
            "cd1": cd1,
            "d_feat": d_feat,
            "hidden_dim": hidden_dim,
            "hidden_num": hidden_num,
            "group_num": group_num,
            "dilations": dilations,
            "inter_connect": inter_connect,
            "norm_type": norm_type,
            "rnn_type": rnn_type,
            "is_dual_rnn": is_dual_rnn,
            "is_causal": is_causal,
        }
        self.pre_seperate = TCNN(cin=1, **pre_kwargs)
        post_kwargs = {
            "k1": k1,
            "k2": k2,
            "c": c,
            "kd1": kd1,
            "cd1": cd1,
            "d_feat": d_feat,
            "fft_num": fft_num,
            "dilations": dilations,
            "intra_connect": intra_connect,
            "norm_type": norm_type,
            "is_causal": is_causal,
            "is_u2": is_u2,
            "group_num": group_num,
        }

        if self.post_type == "vanilla":
            self.post = VanillaPostProcessing(cin=4, **post_kwargs)
        elif self.post_type == "collaborative":
            self.post = CollaborativePostProcessing(cin=4, **post_kwargs)

        # enframe and ola
        self.enframe = TorchSignalToFrames(frame_size=win_size, frame_shift=win_shift)
        self.ola = TorchOLA(frame_shift=win_shift)

    def forward(self, inpt: Tensor) -> tuple:
        """
        inpt: (B, L)
        """
        #
        frame_inpt = self.enframe(inpt)
        stft_inpt = torch.stft(
            inpt,
            self.fft_num,
            self.win_shift,
            self.win_size,
            window=torch.sqrt(torch.hann_window(self.win_size).to(inpt.device)),
            return_complex=True,
        )
        stft_inpt = torch.view_as_real(stft_inpt)
        esti_x = self.pre_seperate(frame_inpt)

        esti_wav = self.ola(esti_x)
        esti_stft = torch.stft(
            esti_wav,
            self.fft_num,
            self.win_shift,
            self.win_size,
            window=torch.sqrt(torch.hann_window(self.win_size).to(inpt.device)),
            return_complex=True,  # B,F,T,2 if false
        )
        esti_stft = torch.view_as_real(esti_stft)
        p = 0.5 if self.is_mu_compress else 1.0

        # B,T,F
        esti_mag, esti_phase = ((torch.norm(esti_stft, dim=-1) + torch_eps) ** p).transpose(
            -2, -1
        ), torch.atan2(esti_stft[..., -1], esti_stft[..., 0]).transpose(-2, -1)
        mix_mag, mix_phase = ((torch.norm(stft_inpt, dim=-1) + torch_eps) ** p).transpose(
            -2, -1
        ), torch.atan2(stft_inpt[..., -1], stft_inpt[..., 0]).transpose(-2, -1)

        comp_phase = torch.stack(
            (mix_mag * torch.cos(esti_phase), mix_mag * torch.sin(esti_phase)), dim=1
        )
        comp_mag = torch.stack(
            (esti_mag * torch.cos(mix_phase), esti_mag * torch.sin(mix_phase)), dim=1
        )

        post_x = self.post(comp_mag, comp_phase)
        return [esti_wav, esti_mag], post_x


class CollaborativePostProcessingFIG6(nn.Module):
    def __init__(
        self,
        cin: int,
        k1: Union[Tuple, List],
        k2: Union[Tuple, List],
        c: int,
        kd1: int,
        cd1: int,
        d_feat: int,
        fft_num: int,
        dilations: Union[Tuple, List],
        intra_connect: str,
        norm_type: str,
        is_causal: bool,
        is_u2: bool,
        group_num: int = 2,
    ):
        super().__init__()
        self.cin = cin
        self.k1 = k1
        self.k2 = k2
        self.c = c
        self.kd1 = kd1
        self.cd1 = cd1
        self.d_feat = d_feat
        self.fft_num = fft_num
        self.dilations = dilations
        self.intra_connect = intra_connect
        self.norm_type = norm_type
        self.is_causal = is_causal
        self.is_u2 = is_u2
        self.group_num = group_num
        #
        freq_dim = fft_num // 2 + 1
        if self.is_u2:
            self.en = FreqU2NetEncoder(
                cin=cin, k1=k1, k2=k2, c=c, intra_connect=intra_connect, norm_type=norm_type
            )
        else:
            self.en = FreqUNetEncoder(cin=cin, k1=k1, c=c, norm_type=norm_type)
        self.gain_in_conv = nn.Conv1d(d_feat + freq_dim, d_feat, kernel_size=1)
        self.resi_in_conv = nn.Conv1d(d_feat + freq_dim * 2, d_feat, kernel_size=1)
        stcns_gain, stcns_resi = [], []
        for i in range(group_num):
            stcns_gain.append(TCMList(kd1, cd1, d_feat, norm_type, dilations, is_causal))
            stcns_resi.append(TCMList(kd1, cd1, d_feat, norm_type, dilations, is_causal))
        self.stcns_gain, self.stcns_resi = nn.ModuleList(stcns_gain), nn.ModuleList(stcns_resi)
        # self.gain_out_conv = nn.Sequential(nn.Conv1d(d_feat, freq_dim, kernel_size=1), nn.Sigmoid())
        self.gain_out_conv = nn.Sequential(nn.Conv1d(d_feat, freq_dim, kernel_size=1), nn.PReLU())
        self.resi_out_conv = nn.Conv1d(d_feat, freq_dim * 2, kernel_size=1)
        #
        self.real_rnn = nn.GRU(
            input_size=freq_dim * 2, hidden_size=freq_dim, num_layers=1, batch_first=True
        )
        self.imag_rnn = nn.GRU(
            input_size=freq_dim * 2, hidden_size=freq_dim, num_layers=1, batch_first=True
        )
        self.real_decode = nn.Linear(freq_dim, freq_dim, bias=False)
        self.imag_decode = nn.Linear(freq_dim, freq_dim, bias=False)

    def forward(self, comp_mag: Tensor, comp_phase: Tensor, hl: Tensor) -> Tensor:
        """
        comp_mag: (B, 2, T, F)
        comp_phase: (B, 2, T, F)
        """
        inpt_x = torch.cat((comp_mag, comp_phase, hl), dim=1)
        en_x, _ = self.en(inpt_x)
        b_size, c, seq_len, f = en_x.shape
        en_x = en_x.transpose(-2, -1).contiguous().view(b_size, -1, seq_len)
        # gain branch
        gain_branch_inpt = torch.cat((en_x, torch.norm(comp_mag, dim=1).transpose(-2, -1)), dim=1)
        gain_branch_x = self.gain_in_conv(gain_branch_inpt)
        # resi branch
        resi_branch_inpt = torch.cat(
            (en_x, comp_phase.transpose(-2, -1).contiguous().view(b_size, -1, seq_len)), dim=1
        )
        resi_branch_x = self.resi_in_conv(resi_branch_inpt)
        #
        gain_x, resi_x = gain_branch_x.clone(), resi_branch_x.clone()
        for i in range(self.group_num):
            gain_x = self.stcns_gain[i](gain_x)
            resi_x = self.stcns_resi[i](resi_x)
        gain = self.gain_out_conv(gain_x).transpose(-2, -1)
        com_resi = self.resi_out_conv(resi_x)
        resi_r, resi_i = torch.chunk(com_resi, 2, dim=1)
        resi = torch.stack((resi_r, resi_i), dim=1).transpose(-2, -1)
        # collaboratively recover
        comp_phase = comp_phase * gain.unsqueeze(dim=1)
        comp_mag = comp_mag + resi
        # fusion
        real_x = torch.cat((comp_phase[:, 0, ...], comp_mag[:, 0, ...]), dim=-1)
        # print("real_x.shape", real_x.shape)
        imag_x = torch.cat((comp_phase[:, -1, ...], comp_mag[:, -1, ...]), dim=-1)
        # print("imag_x.shape", imag_x.shape)

        real_x, imag_x = self.real_decode(self.real_rnn(real_x)[0]), self.imag_decode(
            self.imag_rnn(imag_x)[0]
        )
        return torch.stack((real_x, imag_x), dim=1)


@tables.register("models", "CompNetFIG6")
class CompNetFIG6(nn.Module):
    """
    CompNet: Complementary network for single-channel speech enhancement
    """

    def __init__(
        self,
        win_size: int = 320,
        win_shift: int = 160,
        fft_num: int = 320,
        k1: Union[Tuple, List] = (2, 3),
        k2: Union[Tuple, List] = (2, 3),
        c: int = 64,
        embed_dim: int = 64,
        kd1: int = 5,
        cd1: int = 64,
        d_feat: int = 256,
        hidden_dim: int = 64,
        hidden_num: int = 2,
        group_num: int = 2,
        dilations: Union[Tuple, List] = (1, 2, 5, 9),
        inter_connect: str = "cat",
        intra_connect: str = "cat",
        norm_type: str = "iLN",
        rnn_type: str = "LSTM",
        post_type: str = "collaborative",
        is_dual_rnn: bool = True,
        is_causal: bool = True,
        is_u2: bool = True,
        is_mu_compress: bool = True,
    ):
        super().__init__()
        self.win_size = win_size
        self.win_shift = win_shift
        self.fft_num = fft_num
        self.k1 = tuple(k1)
        self.k2 = tuple(k2)
        self.c = c
        self.embed_dim = embed_dim
        self.kd1 = kd1
        self.cd1 = cd1
        self.d_feat = d_feat
        self.hidden_dim = hidden_dim
        self.hidden_num = hidden_num
        self.group_num = group_num
        self.dilations = tuple(dilations)
        self.inter_connect = inter_connect
        self.intra_connect = intra_connect
        self.norm_type = norm_type
        self.rnn_type = rnn_type
        self.post_type = post_type
        self.is_dual_rnn = is_dual_rnn
        self.is_causal = is_causal
        self.is_u2 = is_u2
        self.is_mu_compress = is_mu_compress
        #
        # first simultaneously pre-estimate mag and phase
        pre_kwargs = {
            "k1": k1,
            "c": c,
            "embed_dim": embed_dim,
            "kd1": kd1,
            "cd1": cd1,
            "d_feat": d_feat,
            "hidden_dim": hidden_dim,
            "hidden_num": hidden_num,
            "group_num": group_num,
            "dilations": dilations,
            "inter_connect": inter_connect,
            "norm_type": norm_type,
            "rnn_type": rnn_type,
            "is_dual_rnn": is_dual_rnn,
            "is_causal": is_causal,
        }
        self.pre_seperate = TCNN(cin=1, **pre_kwargs)
        post_kwargs = {
            "k1": k1,
            "k2": k2,
            "c": c,
            "kd1": kd1,
            "cd1": cd1,
            "d_feat": d_feat,
            "fft_num": fft_num,
            "dilations": dilations,
            "intra_connect": intra_connect,
            "norm_type": norm_type,
            "is_causal": is_causal,
            "is_u2": is_u2,
            "group_num": group_num,
        }

        if self.post_type == "vanilla":
            self.post = VanillaPostProcessing(cin=4, **post_kwargs)
        elif self.post_type == "collaborative":
            self.post = CollaborativePostProcessingFIG6(cin=5, **post_kwargs)

        # enframe and ola
        self.enframe = TorchSignalToFrames(frame_size=win_size, frame_shift=win_shift)
        self.ola = TorchOLA(frame_shift=win_shift)

        nbin = win_shift + 1
        self.freqs = torch.linspace(0, 8000, nbin)  # []
        self.preprocess = HLModule(nbin, HL_freq_extend=self.freqs)

    def forward(self, inpt: Tensor, HL: Tensor) -> tuple:
        """
        inpt: (B, L)
        """
        #
        hl_b = self.preprocess.extend_with_linear(HL)  # b,1,1,f

        frame_inpt = self.enframe(inpt)
        stft_inpt = torch.stft(
            inpt,
            self.fft_num,
            self.win_shift,
            self.win_size,
            window=torch.sqrt(torch.hann_window(self.win_size).to(inpt.device)),
            return_complex=True,
        )
        stft_inpt = torch.view_as_real(stft_inpt)
        #### NOTE stage 1
        esti_x = self.pre_seperate(frame_inpt)

        esti_wav = self.ola(esti_x)
        esti_stft = torch.stft(
            esti_wav,
            self.fft_num,
            self.win_shift,
            self.win_size,
            window=torch.sqrt(torch.hann_window(self.win_size).to(inpt.device)),
            return_complex=True,  # B,F,T,2 if false
        )
        esti_stft = torch.view_as_real(esti_stft)
        p = 0.5 if self.is_mu_compress else 1.0

        # B,T,F
        esti_mag, esti_phase = ((torch.norm(esti_stft, dim=-1) + torch_eps) ** p).transpose(
            -2, -1
        ), torch.atan2(esti_stft[..., -1], esti_stft[..., 0]).transpose(-2, -1)
        mix_mag, mix_phase = ((torch.norm(stft_inpt, dim=-1) + torch_eps) ** p).transpose(
            -2, -1
        ), torch.atan2(stft_inpt[..., -1], stft_inpt[..., 0]).transpose(-2, -1)

        comp_phase = torch.stack(
            (mix_mag * torch.cos(esti_phase), mix_mag * torch.sin(esti_phase)), dim=1
        )
        comp_mag = torch.stack(
            (esti_mag * torch.cos(mix_phase), esti_mag * torch.sin(mix_phase)), dim=1
        )
        nT = comp_mag.size(-2)
        hl_b = hl_b.repeat(1, 1, nT, 1)

        #### NOTE stage 2
        post_x = self.post(comp_mag, comp_phase, hl_b)  # b,c,t,f
        post_wav = einops.rearrange(post_x, "b c t f->b f t c")
        post_wav = torch.view_as_complex(post_wav.contiguous())
        post_wav = torch.istft(
            post_wav,
            self.fft_num,
            self.win_shift,
            self.win_size,
            window=torch.sqrt(torch.hann_window(self.win_size).to(inpt.device)),
        )
        return [esti_wav, esti_mag, post_x], post_wav


if __name__ == "__main__":
    net = CompNetFIG6(
        # win_size=320,
        # win_shift=160,
        # fft_num=320,
        win_size=512,
        win_shift=256,
        fft_num=512,
        # k1=(2, 3),
        # k2=(2, 3),
        # c=32,
        # embed_dim=32,
        # kd1=5,
        # cd1=64,
        # d_feat=128,
        # hidden_dim=48,
        # hidden_num=2,
        # group_num=2,
        # dilations=(1, 2, 5, 9),
        # inter_connect="cat",
        # intra_connect="cat",
        # norm_type="iLN",
        # rnn_type="LSTM",
        # # post_type="vanilla",
        # is_dual_rnn=True,
        # is_causal=True,
        # is_u2=True,
        # is_mu_compress=True,
    ).cuda()
    hl = torch.randn(1, 6).cuda()
    x = torch.rand([1, 16000]).cuda()
    _, y = net(x, hl)
    print(f"{x.shape}->{y.shape}")

    check_flops(net, x, hl)
