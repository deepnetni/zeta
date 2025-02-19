"""
Our proposed NUNet-TLS
(Baseline+TLS+CTFA)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import get_window

from utils.check_flops import check_flops
from JointNSHModel import expand_HT
from utils.register import tables


# this is from conv_stft https://github.com/huyanxin/DeepComplexCRN
def init_kernels(win_len, fft_len, win_type=None, invers=False):
    if win_type == "None" or win_type is None:
        window = np.ones(win_len)
    else:
        window = get_window(win_type, win_len, fftbins=True)  # **0.5

    N = fft_len
    fourier_basis = np.fft.rfft(np.eye(N))[:win_len]
    real_kernel = np.real(fourier_basis)
    imag_kernel = np.imag(fourier_basis)
    kernel = np.concatenate([real_kernel, imag_kernel], 1).T

    if invers:
        kernel = np.linalg.pinv(kernel).T

    kernel = kernel * window
    kernel = kernel[:, None, :]
    return torch.from_numpy(kernel.astype(np.float32)), torch.from_numpy(
        window[None, :, None].astype(np.float32)
    )


class ConvSTFT(nn.Module):
    def __init__(self, win_len, win_inc, fft_len=None, win_type="hamming", feature_type="real"):
        super(ConvSTFT, self).__init__()

        if fft_len is None:
            self.fft_len = np.int(2 ** np.ceil(np.log2(win_len)))
        else:
            self.fft_len = fft_len

        kernel, _ = init_kernels(win_len, self.fft_len, win_type)
        self.register_buffer("weight", kernel)
        self.feature_type = feature_type
        self.stride = win_inc
        self.win_len = win_len
        self.dim = self.fft_len

    def forward(self, inputs):
        if inputs.dim() == 2:
            inputs = torch.unsqueeze(inputs, 1)
        inputs = F.pad(inputs, [self.win_len - self.stride, self.win_len - self.stride])
        outputs = F.conv1d(inputs, self.weight, stride=self.stride)

        if self.feature_type == "complex":
            return outputs
        else:
            dim = self.dim // 2 + 1
            real = outputs[:, :dim, :]
            imag = outputs[:, dim:, :]
            mags = torch.sqrt(real**2 + imag**2)
            phase = torch.atan2(imag, real)
            return mags, phase  # , real, imag


class ConviSTFT(nn.Module):
    def __init__(self, win_len, win_inc, fft_len=None, win_type="hamming", feature_type="real"):
        super(ConviSTFT, self).__init__()
        if fft_len is None:
            self.fft_len = np.int(2 ** np.ceil(np.log2(win_len)))
        else:
            self.fft_len = fft_len
        kernel, window = init_kernels(win_len, self.fft_len, win_type, invers=True)
        self.register_buffer("weight", kernel)
        self.feature_type = feature_type
        self.win_type = win_type
        self.win_len = win_len
        self.stride = win_inc
        self.dim = self.fft_len
        self.register_buffer("window", window)
        self.register_buffer("enframe", torch.eye(win_len)[:, None, :])

    def forward(self, inputs, phase=None):
        """
        inputs : [B, N+2, T] (complex spec) or [B, N//2+1, T] (mags)
        phase: [B, N//2+1, T] (if not none)
        """

        if phase is not None:
            real = inputs * torch.cos(phase)
            imag = inputs * torch.sin(phase)
            inputs = torch.cat([real, imag], 1)

        outputs = F.conv_transpose1d(inputs, self.weight, stride=self.stride)

        # this is from torch-stft: https://github.com/pseeth/torch-stft
        t = self.window.repeat(1, 1, inputs.size(-1)) ** 2
        coff = F.conv_transpose1d(t, self.enframe, stride=self.stride)

        outputs = outputs / (coff + 1e-8)

        outputs = outputs[..., self.win_len - self.stride : -(self.win_len - self.stride)]

        return outputs


# causal convolution
class causalConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=1, dilation=1, groups=1):
        super(causalConv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=(padding[0], 0),
            dilation=dilation,
            groups=groups,
        )
        self.padding = padding[1]

    def forward(self, x):
        x = F.pad(x, [self.padding, 0, 0, 0])
        out = self.conv(x)
        return out


# convolution block
class CONV(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CONV, self).__init__()
        self.conv = causalConv2d(in_ch, out_ch, kernel_size=(3, 2), stride=(2, 1), padding=(1, 1))
        self.ln = nn.GroupNorm(1, out_ch, eps=1e-8)
        self.prelu = nn.PReLU()

    def forward(self, x):
        return self.prelu(self.ln(self.conv(x)))


# convolution block for input layer
class INCONV(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(INCONV, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.ln = nn.GroupNorm(1, out_ch, eps=1e-8)
        self.prelu = nn.PReLU()

    def forward(self, x):
        return self.prelu(self.ln(self.conv(x)))


# sub-pixel convolution block
class SPCONV(nn.Module):
    def __init__(self, in_ch, out_ch, scale_factor=2):
        super(SPCONV, self).__init__()
        self.conv = causalConv2d(in_ch, out_ch * scale_factor, kernel_size=(3, 2), padding=(1, 1))
        self.ln = nn.GroupNorm(1, out_ch, eps=1e-8)
        self.prelu = nn.PReLU()

        self.n = scale_factor

    def forward(self, x):
        x = self.conv(x)  # [B, C, F, T]

        x = x.permute(0, 3, 2, 1)  # [B, T, F, C]
        r = torch.reshape(
            x, (x.size(0), x.size(1), x.size(2), x.size(3) // self.n, self.n)
        )  # [B, T, F, C//2 , 2]
        r = r.permute(0, 1, 2, 4, 3)  # [B, T, F, 2, C//2]
        r = torch.reshape(
            r, (x.size(0), x.size(1), x.size(2) * self.n, x.size(3) // self.n)
        )  # [B, T, F*2, C//2]
        r = r.permute(0, 3, 2, 1)  # [B, C, F, T]

        out = self.ln(r)
        out = self.prelu(out)
        return out


# 1x1 conv for down-sampling
class down_sampling(nn.Module):
    def __init__(self, in_ch):
        super(down_sampling, self).__init__()
        self.down_sampling = nn.Conv2d(
            in_ch, in_ch, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)
        )

    def forward(self, x):
        return self.down_sampling(x)


# 1x1 conv for up-sampling
class upsampling(nn.Module):
    def __init__(self, in_ch):
        super(upsampling, self).__init__()
        self.upsampling = nn.ConvTranspose2d(
            in_ch, in_ch, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0), output_padding=(1, 0)
        )

    def forward(self, x):
        out = self.upsampling(x)
        return out


# dilated dense block
class dilatedDenseBlock(nn.Module):
    def __init__(self, in_ch, out_ch, n_layers):
        super(dilatedDenseBlock, self).__init__()

        self.input_layer = causalConv2d(
            in_ch, in_ch // 2, kernel_size=(3, 2), padding=(1, 1)
        )  # channel half
        self.prelu1 = nn.PReLU()

        # dilated dense layer
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.caus_padd = ((2**i) // 2) * 2
            if i == 0:
                self.caus_padd = 1

            self.layers.append(
                nn.Sequential(
                    # depth-wise separable conv
                    causalConv2d(
                        in_ch // 2 + i * in_ch // 2,
                        in_ch // 2,
                        kernel_size=(3, 2),
                        padding=(2**i, self.caus_padd),
                        dilation=2**i,
                        groups=in_ch // 2,
                    ),
                    # depth-wise
                    nn.Conv2d(in_ch // 2, in_ch // 2, kernel_size=1),  # pointwise
                    nn.GroupNorm(1, in_ch // 2, eps=1e-8),
                    nn.PReLU(),
                )
            )

        self.out_layer = causalConv2d(
            in_ch // 2, out_ch, kernel_size=(3, 2), padding=(1, 1)
        )  # channel revert
        self.prelu2 = nn.PReLU()

    def forward(self, x):
        x = self.input_layer(x)  # C: in_ch//2
        x = self.prelu1(x)

        out1 = self.layers[0](x)

        # out2 = self.layers[1](torch.cat([out1, x], dim=1))
        out2 = torch.cat([out1, x], dim=1)  # C: in_ch//2 * 2
        out2 = self.layers[1](out2)

        # out3 = self.layers[2](torch.cat([out2, out1, x], dim=1))
        out3 = torch.cat([out2, out1], dim=1)
        out3 = torch.cat([out3, x], dim=1)  # C: in_ch//2 * 3
        out3 = self.layers[2](out3)

        # out4 = self.layers[3](torch.cat([out3, out2, out1, x], dim=1))
        out4 = torch.cat([out3, out2], dim=1)  # C: in_ch//2 * 4
        out4 = torch.cat([out4, out1], dim=1)
        out4 = torch.cat([out4, x], dim=1)
        out4 = self.layers[3](out4)

        # out5 = self.layers[4](torch.cat([out4, out3, out2, out1, x], dim=1))
        out5 = torch.cat([out4, out3], dim=1)  # C: in_ch//2 * 5
        out5 = torch.cat([out5, out2], dim=1)
        out5 = torch.cat([out5, out1], dim=1)
        out5 = torch.cat([out5, x], dim=1)
        out5 = self.layers[4](out5)

        # out = self.layers[5](torch.cat([out5, out4, out3, out2, out1, x], dim=1))
        out = torch.cat([out5, out4], dim=1)  # C: in_ch//2 * 6
        out = torch.cat([out, out3], dim=1)
        out = torch.cat([out, out2], dim=1)
        out = torch.cat([out, out1], dim=1)
        out = torch.cat([out, x], dim=1)
        out = self.layers[5](out)

        out = self.out_layer(out)
        out = self.prelu2(out)

        return out


# causal version of a time-frequency attention (TFA) module
# The paper of the original TFA : https://arxiv.org/abs/2111.07518
class CTFA(nn.Module):
    def __init__(self, in_ch, out_ch=16, time_seq=32):
        super(CTFA, self).__init__()
        # time attention
        self.time_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.time_conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=1)
        self.time_relu = nn.ReLU()
        self.time_conv2 = nn.Conv1d(out_ch, in_ch, kernel_size=1)
        self.time_sigmoid = nn.Sigmoid()

        # frequency attention
        self.freq_avg_pool = nn.AvgPool1d(time_seq, stride=1)
        self.freq_conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.freq_relu = nn.ReLU()
        self.freq_conv2 = nn.Conv2d(out_ch, in_ch, kernel_size=1)
        self.freq_sigmoid = nn.Sigmoid()

        # for real-time
        self.padd = time_seq - 1

    def forward(self, x):
        B, C, D, T = x.size()

        # time attention
        Z_T = x.permute(0, 1, 3, 2).reshape([B, C * T, D])
        TA = self.time_avg_pool(Z_T)  # [B, C*T, 1]
        TA = TA.reshape([B, C, T])
        TA = self.time_conv1(TA)
        TA = self.time_relu(TA)
        TA = self.time_conv2(TA)
        TA = self.time_sigmoid(TA)
        TA = TA.reshape([B, C, T, 1])
        TA = TA.expand(B, C, T, D).permute(0, 1, 3, 2)

        # frequency attention
        x_pad = F.pad(x, [self.padd, 0, 0, 0])
        Z_F = x_pad.reshape([B, C * D, T + self.padd])
        FA = self.freq_avg_pool(Z_F)  # [B, C*F, T]
        FA = FA.reshape([B, C, D, T])
        FA = self.freq_conv1(FA)
        FA = self.freq_relu(FA)
        FA = self.freq_conv2(FA)
        FA = self.freq_sigmoid(FA)

        # multiply
        TFA = FA * TA
        out = x * TFA

        return out


# Multi-Scale Feature Extraction (MSFE) - e6 (for encoder part)
class MSFEe6(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(MSFEe6, self).__init__()
        self.input_layer = INCONV(in_ch, out_ch)

        # encoder
        self.en1 = CONV(out_ch, mid_ch)
        self.en2 = CONV(mid_ch, mid_ch)
        self.en3 = CONV(mid_ch, mid_ch)
        self.en4 = CONV(mid_ch, mid_ch)
        self.en5 = CONV(mid_ch, mid_ch)
        self.en6 = CONV(mid_ch, mid_ch)

        # bottleneck
        self.ddense = dilatedDenseBlock(mid_ch, mid_ch, 6)

        # decoder
        self.de1 = SPCONV(mid_ch * 2, mid_ch)
        self.de2 = SPCONV(mid_ch * 2, mid_ch)
        self.de3 = SPCONV(mid_ch * 2, mid_ch)
        self.de4 = SPCONV(mid_ch * 2, mid_ch)
        self.de5 = SPCONV(mid_ch * 2, mid_ch)
        self.de6 = SPCONV(mid_ch * 2, out_ch)

        # attention
        self.ctfa = CTFA(out_ch)

    def forward(self, x):
        x = self.input_layer(x)

        # encoder
        out1 = self.en1(x)
        out2 = self.en2(out1)
        out3 = self.en3(out2)
        out4 = self.en4(out3)
        out5 = self.en5(out4)
        out6 = self.en6(out5)

        # bottleneck
        out = self.ddense(out6)

        # decoder
        out6 = self.de1(torch.cat([out, out6], dim=1))
        out5 = self.de2(torch.cat([out6, out5], dim=1))
        out4 = self.de3(torch.cat([out5, out4], dim=1))
        out3 = self.de4(torch.cat([out4, out3], dim=1))
        out2 = self.de5(torch.cat([out3, out2], dim=1))
        out1 = self.de6(torch.cat([out2, out1], dim=1))

        # attention and residual
        out = self.ctfa(out1) + x
        return out, out1, out2, out3, out4, out5, out6


# Multi-Scale Feature Extraction (MSFE) - e5
class MSFEe5(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(MSFEe5, self).__init__()
        self.input_layer = INCONV(in_ch, out_ch)

        # encoder
        self.en1 = CONV(out_ch, mid_ch)
        self.en2 = CONV(mid_ch, mid_ch)
        self.en3 = CONV(mid_ch, mid_ch)
        self.en4 = CONV(mid_ch, mid_ch)
        self.en5 = CONV(mid_ch, mid_ch)

        # bottleneck
        self.ddense = dilatedDenseBlock(mid_ch, mid_ch, 6)

        # decoder
        self.de1 = SPCONV(mid_ch * 2, mid_ch)
        self.de2 = SPCONV(mid_ch * 2, mid_ch)
        self.de3 = SPCONV(mid_ch * 2, mid_ch)
        self.de4 = SPCONV(mid_ch * 2, mid_ch)
        self.de5 = SPCONV(mid_ch * 2, out_ch)

        # attention
        self.ctfa = CTFA(out_ch)

    def forward(self, x):
        x = self.input_layer(x)

        # encoder
        out1 = self.en1(x)
        out2 = self.en2(out1)
        out3 = self.en3(out2)
        out4 = self.en4(out3)
        out5 = self.en5(out4)

        # bottleneck
        out = self.ddense(out5)

        # decoder
        out5 = self.de1(torch.cat([out, out5], dim=1))
        out4 = self.de2(torch.cat([out5, out4], dim=1))
        out3 = self.de3(torch.cat([out4, out3], dim=1))
        out2 = self.de4(torch.cat([out3, out2], dim=1))
        out1 = self.de5(torch.cat([out2, out1], dim=1))

        # attention and residual
        out = self.ctfa(out1) + x
        return out, out1, out2, out3, out4, out5


# Multi-Scale Feature Extraction (MSFE) - e4
class MSFEe4(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(MSFEe4, self).__init__()
        self.input_layer = INCONV(in_ch, out_ch)

        # encoder
        self.en1 = CONV(out_ch, mid_ch)
        self.en2 = CONV(mid_ch, mid_ch)
        self.en3 = CONV(mid_ch, mid_ch)
        self.en4 = CONV(mid_ch, mid_ch)

        # bottleneck
        self.ddense = dilatedDenseBlock(mid_ch, mid_ch, 6)

        # decoder
        self.de1 = SPCONV(mid_ch * 2, mid_ch)
        self.de2 = SPCONV(mid_ch * 2, mid_ch)
        self.de3 = SPCONV(mid_ch * 2, mid_ch)
        self.de4 = SPCONV(mid_ch * 2, out_ch)

        # attention
        self.ctfa = CTFA(out_ch)

    def forward(self, x):
        x = self.input_layer(x)

        # encoder
        out1 = self.en1(x)
        out2 = self.en2(out1)
        out3 = self.en3(out2)
        out4 = self.en4(out3)

        # bottleneck
        out = self.ddense(out4)

        # decoder
        out4 = self.de1(torch.cat([out, out4], dim=1))
        out3 = self.de2(torch.cat([out4, out3], dim=1))
        out2 = self.de3(torch.cat([out3, out2], dim=1))
        out1 = self.de4(torch.cat([out2, out1], dim=1))

        # attention and residual
        out = self.ctfa(out1) + x
        return out, out1, out2, out3, out4


# Multi-Scale Feature Extraction (MSFE) - e3
class MSFEe3(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(MSFEe3, self).__init__()
        self.input_layer = INCONV(in_ch, out_ch)

        # encoder
        self.en1 = CONV(out_ch, mid_ch)
        self.en2 = CONV(mid_ch, mid_ch)
        self.en3 = CONV(mid_ch, mid_ch)

        # bottleneck
        self.ddense = dilatedDenseBlock(mid_ch, mid_ch, 6)

        # decoder
        self.de1 = SPCONV(mid_ch * 2, mid_ch)
        self.de2 = SPCONV(mid_ch * 2, mid_ch)
        self.de3 = SPCONV(mid_ch * 2, out_ch)

        # attention
        self.ctfa = CTFA(out_ch)

    def forward(self, x):
        x = self.input_layer(x)

        # encoder
        out1 = self.en1(x)
        out2 = self.en2(out1)
        out3 = self.en3(out2)

        # bottleneck
        out = self.ddense(out3)

        # decoder
        out3 = self.de1(torch.cat([out, out3], dim=1))
        out2 = self.de2(torch.cat([out3, out2], dim=1))
        out1 = self.de3(torch.cat([out2, out1], dim=1))

        # attention and residual
        out = self.ctfa(out1) + x
        return out, out1, out2, out3


# Multi-Scale Feature Extraction (MSFE) - d6  (for decoder part)
class MSFEd6(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(MSFEd6, self).__init__()
        self.input_layer = INCONV(in_ch, out_ch)

        # encoder
        self.en1 = CONV(out_ch * 2, mid_ch)
        self.en2 = CONV(mid_ch * 2, mid_ch)
        self.en3 = CONV(mid_ch * 2, mid_ch)
        self.en4 = CONV(mid_ch * 2, mid_ch)
        self.en5 = CONV(mid_ch * 2, mid_ch)
        self.en6 = CONV(mid_ch * 2, mid_ch)

        # bottleneck
        self.ddense = dilatedDenseBlock(mid_ch, mid_ch, 6)

        # decoder
        self.de1 = SPCONV(mid_ch * 2, mid_ch)
        self.de2 = SPCONV(mid_ch * 2, mid_ch)
        self.de3 = SPCONV(mid_ch * 2, mid_ch)
        self.de4 = SPCONV(mid_ch * 2, mid_ch)
        self.de5 = SPCONV(mid_ch * 2, mid_ch)
        self.de6 = SPCONV(mid_ch * 2, out_ch)

        # attention
        self.ctfa = CTFA(out_ch)

    def forward(self, x, ed1, ed2, ed3, ed4, ed5, ed6):
        x = self.input_layer(x)

        # encoder
        out1 = self.en1(torch.cat([x, ed1], dim=1))
        out2 = self.en2(torch.cat([out1, ed2], dim=1))
        out3 = self.en3(torch.cat([out2, ed3], dim=1))
        out4 = self.en4(torch.cat([out3, ed4], dim=1))
        out5 = self.en5(torch.cat([out4, ed5], dim=1))
        out6 = self.en6(torch.cat([out5, ed6], dim=1))

        # bottleneck
        out = self.ddense(out6)

        # decoder
        out = self.de1(torch.cat([out, out6], dim=1))
        out = self.de2(torch.cat([out, out5], dim=1))
        out = self.de3(torch.cat([out, out4], dim=1))
        out = self.de4(torch.cat([out, out3], dim=1))
        out = self.de5(torch.cat([out, out2], dim=1))
        out = self.de6(torch.cat([out, out1], dim=1))

        # attention
        out = self.ctfa(out)
        out += x
        return out


# Multi-Scale Feature Extraction (MSFE) - d5
class MSFEd5(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(MSFEd5, self).__init__()
        self.input_layer = INCONV(in_ch, out_ch)

        # encoder
        self.en1 = CONV(out_ch * 2, mid_ch)
        self.en2 = CONV(mid_ch * 2, mid_ch)
        self.en3 = CONV(mid_ch * 2, mid_ch)
        self.en4 = CONV(mid_ch * 2, mid_ch)
        self.en5 = CONV(mid_ch * 2, mid_ch)

        # bottleneck
        self.ddense = dilatedDenseBlock(mid_ch, mid_ch, 6)

        # decoder
        self.de1 = SPCONV(mid_ch * 2, mid_ch)
        self.de2 = SPCONV(mid_ch * 2, mid_ch)
        self.de3 = SPCONV(mid_ch * 2, mid_ch)
        self.de4 = SPCONV(mid_ch * 2, mid_ch)
        self.de5 = SPCONV(mid_ch * 2, out_ch)

        # attention
        self.ctfa = CTFA(out_ch)

    def forward(self, x, ed1, ed2, ed3, ed4, ed5):
        x = self.input_layer(x)

        # encoder
        out1 = self.en1(torch.cat([x, ed1], dim=1))
        out2 = self.en2(torch.cat([out1, ed2], dim=1))
        out3 = self.en3(torch.cat([out2, ed3], dim=1))
        out4 = self.en4(torch.cat([out3, ed4], dim=1))
        out5 = self.en5(torch.cat([out4, ed5], dim=1))

        # bottleneck
        out = self.ddense(out5)

        # decoder
        out = self.de1(torch.cat([out, out5], dim=1))
        out = self.de2(torch.cat([out, out4], dim=1))
        out = self.de3(torch.cat([out, out3], dim=1))
        out = self.de4(torch.cat([out, out2], dim=1))
        out = self.de5(torch.cat([out, out1], dim=1))

        # attention
        out = self.ctfa(out)
        out += x
        return out


# Multi-Scale Feature Extraction (MSFE) - d4
class MSFEd4(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(MSFEd4, self).__init__()
        self.input_layer = INCONV(in_ch, out_ch)

        # encoder
        self.en1 = CONV(out_ch * 2, mid_ch)
        self.en2 = CONV(mid_ch * 2, mid_ch)
        self.en3 = CONV(mid_ch * 2, mid_ch)
        self.en4 = CONV(mid_ch * 2, mid_ch)

        # bottleneck
        self.ddense = dilatedDenseBlock(mid_ch, mid_ch, 6)

        # decoder
        self.de1 = SPCONV(mid_ch * 2, mid_ch)
        self.de2 = SPCONV(mid_ch * 2, mid_ch)
        self.de3 = SPCONV(mid_ch * 2, mid_ch)
        self.de4 = SPCONV(mid_ch * 2, out_ch)

        # attention
        self.ctfa = CTFA(out_ch)

    def forward(self, x, ed1, ed2, ed3, ed4):
        x = self.input_layer(x)

        # encoder
        out1 = self.en1(torch.cat([x, ed1], dim=1))
        out2 = self.en2(torch.cat([out1, ed2], dim=1))
        out3 = self.en3(torch.cat([out2, ed3], dim=1))
        out4 = self.en4(torch.cat([out3, ed4], dim=1))

        # bottleneck
        out = self.ddense(out4)

        # decoder
        out = self.de1(torch.cat([out, out4], dim=1))
        out = self.de2(torch.cat([out, out3], dim=1))
        out = self.de3(torch.cat([out, out2], dim=1))
        out = self.de4(torch.cat([out, out1], dim=1))

        # attention
        out = self.ctfa(out)
        out += x
        return out


# Multi-Scale Feature Extraction (MSFE) - d3
class MSFEd3(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(MSFEd3, self).__init__()
        self.input_layer = INCONV(in_ch, out_ch)

        # encoder
        self.en1 = CONV(out_ch * 2, mid_ch)
        self.en2 = CONV(mid_ch * 2, mid_ch)
        self.en3 = CONV(mid_ch * 2, mid_ch)

        # bottleneck
        self.ddense = dilatedDenseBlock(mid_ch, mid_ch, 6)

        # decoder
        self.de1 = SPCONV(mid_ch * 2, mid_ch)
        self.de2 = SPCONV(mid_ch * 2, mid_ch)
        self.de3 = SPCONV(mid_ch * 2, out_ch)

        # attention
        self.ctfa = CTFA(out_ch)

    def forward(self, x, ed1, ed2, ed3):
        x = self.input_layer(x)

        # encoder
        out1 = self.en1(torch.cat([x, ed1], dim=1))
        out2 = self.en2(torch.cat([out1, ed2], dim=1))
        out3 = self.en3(torch.cat([out2, ed3], dim=1))

        # bottleneck
        out = self.ddense(out3)

        # decoder
        out = self.de1(torch.cat([out, out3], dim=1))
        out = self.de2(torch.cat([out, out2], dim=1))
        out = self.de3(torch.cat([out, out1], dim=1))

        # attention
        out = self.ctfa(out)
        out += x
        return out


# Nested UNet using two-level skip connection (NUNet-TLS)
@tables.register("models", "NUNet_TLS")
class NUNet_TLS(nn.Module):
    # https://github.com/seorim0/NUNet-TLS/blob/main/NUNet_TLS.py
    def __init__(self, in_ch=2, mid_ch=32, out_ch=64, nframe=512, nhop=256):
        super(NUNet_TLS, self).__init__()

        # input layer
        self.input_layer = INCONV(in_ch, out_ch)

        # encoder
        self.en1 = MSFEe6(out_ch, mid_ch, out_ch)
        self.down_sampling1 = down_sampling(out_ch)

        self.en2 = MSFEe5(out_ch, mid_ch, out_ch)
        self.down_sampling2 = down_sampling(out_ch)

        self.en3 = MSFEe4(out_ch, mid_ch, out_ch)
        self.down_sampling3 = down_sampling(out_ch)

        self.en4 = MSFEe4(out_ch, mid_ch, out_ch)
        self.down_sampling4 = down_sampling(out_ch)

        self.en5 = MSFEe4(out_ch, mid_ch, out_ch)
        self.down_sampling5 = down_sampling(out_ch)

        self.en6 = MSFEe3(out_ch, mid_ch, out_ch)
        self.down_sampling6 = down_sampling(out_ch)

        # Bottleneck block
        self.DDense = nn.Sequential(dilatedDenseBlock(out_ch, out_ch, 6))

        # decoder
        self.upsampling1 = upsampling(out_ch * 2)
        self.de1 = MSFEd3(out_ch * 2, mid_ch, out_ch)

        self.upsampling2 = upsampling(out_ch * 2)
        self.de2 = MSFEd4(out_ch * 2, mid_ch, out_ch)

        self.upsampling3 = upsampling(out_ch * 2)
        self.de3 = MSFEd4(out_ch * 2, mid_ch, out_ch)

        self.upsampling4 = upsampling(out_ch * 2)
        self.de4 = MSFEd4(out_ch * 2, mid_ch, out_ch)

        self.upsampling5 = upsampling(out_ch * 2)
        self.de5 = MSFEd5(out_ch * 2, mid_ch, out_ch)

        self.upsampling6 = upsampling(out_ch * 2)
        self.de6 = MSFEd6(out_ch * 2, mid_ch, out_ch)

        # output layer
        self.output_layer = nn.Conv2d(out_ch, 1, kernel_size=1)

        # for feature extract
        self.stft = ConvSTFT(nframe, nhop, nframe, feature_type="real")
        self.istft = ConviSTFT(nframe, nhop, nframe, feature_type="real")
        self.reso = 16000 / nframe

    def forward(self, x, HL):
        # STFT
        mags, phase = self.stft(x)  # [B, F, T]
        hx = mags.unsqueeze(1)  # [B, 1, F, T]
        hx = hx[:, :, 1:]

        hl = expand_HT(HL, hx.shape[-1], self.reso)  # B,C(1),T,F
        hl = hl.permute(0, 1, 3, 2)
        hl = hl[..., 1:, :]
        hx = torch.cat([hx, hl], dim=1)

        # input layer
        hx = self.input_layer(hx)

        # encoder stage 1
        hx1, hx1_1, hx1_2, hx1_3, hx1_4, hx1_5, hx1_6 = self.en1(hx)
        hx1 = self.down_sampling1(hx1)

        # encoder stage 2
        hx2, hx2_1, hx2_2, hx2_3, hx2_4, hx2_5 = self.en2(hx1)
        hx2 = self.down_sampling2(hx2)

        # encoder stage 3
        hx3, hx3_1, hx3_2, hx3_3, hx3_4 = self.en3(hx2)
        hx3 = self.down_sampling3(hx3)

        # encoder stage 4
        hx4, hx4_1, hx4_2, hx4_3, hx4_4 = self.en4(hx3)
        hx4 = self.down_sampling4(hx4)

        # encoder stage 5
        hx5, hx5_1, hx5_2, hx5_3, hx5_4 = self.en5(hx4)
        hx5 = self.down_sampling5(hx5)

        # encoder stage 6
        hx6, hx6_1, hx6_2, hx6_3 = self.en6(hx5)
        hx6 = self.down_sampling6(hx6)

        # dilated dense block
        out = self.DDense(hx6)

        # decoder stage 1
        out = self.upsampling1(torch.cat([out, hx6], dim=1))
        out = self.de1(out, hx6_1, hx6_2, hx6_3)

        # decoder stage 2
        out = self.upsampling2(torch.cat([out, hx5], dim=1))
        out = self.de2(out, hx5_1, hx5_2, hx5_3, hx5_4)

        # decoder stage 3
        out = self.upsampling3(torch.cat([out, hx4], dim=1))
        out = self.de3(out, hx4_1, hx4_2, hx4_3, hx4_4)

        # decoder stage 4
        out = self.upsampling4(torch.cat([out, hx3], dim=1))
        out = self.de4(out, hx3_1, hx3_2, hx3_3, hx3_4)

        # decoder stage 5
        out = self.upsampling5(torch.cat([out, hx2], dim=1))
        out = self.de5(out, hx2_1, hx2_2, hx2_3, hx2_4, hx2_5)

        # decoder stage 6
        out = self.upsampling6(torch.cat([out, hx1], dim=1))
        out = self.de6(out, hx1_1, hx1_2, hx1_3, hx1_4, hx1_5, hx1_6)

        # output layer
        out = self.output_layer(out)

        out = F.pad(out, [0, 0, 1, 0])

        # ISTFT
        out_wav = self.istft(out.squeeze(1), phase).squeeze(1)
        out_wav = torch.clamp_(out_wav, -1, 1)  # clipping [-1, 1]
        return out_wav

    def loss(self, target, enhanced):
        # mag_target, _ = self.stft(target)
        # mag_enh, _ = self.stft(enhanced)
        l1 = F.mse_loss(enhanced, target, reduction="mean")
        # l2 = nn.L1Loss()(mag_enh, mag_target)
        # l = (l1 + l2) * 0.5
        # return dict(loss=l, mse_lv=l1, mae_lv=l2)
        return dict(loss=l1)


if __name__ == "__main__":
    inp = torch.randn(1, 16000)
    lbl = torch.randn(1, 16000)
    hl = torch.randn(1, 6)
    net = NUNet_TLS(2)
    check_flops(net, inp, hl)
    loss = net.loss(lbl, inp)
    print(loss)
