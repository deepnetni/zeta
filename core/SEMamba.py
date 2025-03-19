from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn
import yaml
from einops import rearrange

from JointNSHModel import expand_HT
from mamba_modules.mamba_block import TFMambaBlock
from models.conv_stft import STFT
from models.lsigmoid import LearnableSigmoid2D
from utils.check_flops import check_flops

# Reference: https://github.com/yxlu-0102/MP-SENet/blob/main/models/generator.py


@dataclass
class Model_cfg:
    hid_feature: int = 64  # Channels in dense layers.
    compress_factor: float = 0.3  # Compression factor applied to extracted features.
    num_tfmamba: int = 4  # Number of Time-Frequency Mamba (TFMamba) blocks in the model.
    d_state: int = 16  # Dimensionality of the state vector in Mamba blocks.
    d_conv: int = 4  # Convolutional layer dimensionality within Mamba blocks.
    expand: int = 4  # Expansion factor for the layers within the Mamba blocks.
    norm_epsilon: float = (
        0.00001  # Numerical stability in normalization layers within the Mamba blocks.
    )
    beta: float = 2.0  # Hyperparameter for the Learnable Sigmoid function.
    input_channel: int = 2  # Magnitude and Phase
    output_channel: int = 1  # Single Channel Speech Enhancement


@dataclass
class Stft_cfg:
    sampling_rate: int = 16000  # Audio sampling rate in Hz.
    n_fft: int = 512  # FFT components for transforming audio signals.
    hop_size: int = 256  # Samples between successive frames.
    win_size: int = 512  # Window size used in FFT.


@dataclass
class Cfg:
    model_cfg: Model_cfg = Model_cfg()
    stft_cfg: Stft_cfg = Stft_cfg()
    # stft_cfg = field(default_factory=Stft_cfg)


cfg = asdict(Cfg())


def get_padding(kernel_size, dilation=1):
    """
    Calculate the padding size for a convolutional layer.

    Args:
    - kernel_size (int): Size of the convolutional kernel.
    - dilation (int, optional): Dilation rate of the convolution. Defaults to 1.

    Returns:
    - int: Calculated padding size.
    """
    return int((kernel_size * dilation - dilation) / 2)


def get_padding_2d(kernel_size, dilation=(1, 1)):
    """
    Calculate the padding size for a 2D convolutional layer.

    Args:
    - kernel_size (tuple): Size of the convolutional kernel (height, width).
    - dilation (tuple, optional): Dilation rate of the convolution (height, width). Defaults to (1, 1).

    Returns:
    - tuple: Calculated padding size (height, width).
    """
    return (
        int((kernel_size[0] * dilation[0] - dilation[0]) / 2),
        int((kernel_size[1] * dilation[1] - dilation[1]) / 2),
    )


class DenseBlock(nn.Module):
    """
    DenseBlock module consisting of multiple convolutional layers with dilation.
    """

    def __init__(self, cfg, kernel_size=(3, 3), depth=4):
        super(DenseBlock, self).__init__()
        self.cfg = cfg
        self.depth = depth
        self.dense_block = nn.ModuleList()
        self.hid_feature = cfg["model_cfg"]["hid_feature"]

        for i in range(depth):
            dil = 2**i
            dense_conv = nn.Sequential(
                nn.Conv2d(
                    self.hid_feature * (i + 1),
                    self.hid_feature,
                    kernel_size,
                    dilation=(dil, 1),
                    padding=get_padding_2d(kernel_size, (dil, 1)),
                ),
                nn.InstanceNorm2d(self.hid_feature, affine=True),
                nn.PReLU(self.hid_feature),
            )
            self.dense_block.append(dense_conv)

    def forward(self, x):
        """
        Forward pass for the DenseBlock module.

        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor after processing through the dense block.
        """
        skip = x
        for i in range(self.depth):
            x = self.dense_block[i](skip)
            skip = torch.cat([x, skip], dim=1)
        return x


class DenseEncoder(nn.Module):
    """
    DenseEncoder module consisting of initial convolution, dense block, and a final convolution.
    """

    def __init__(self, cfg):
        super(DenseEncoder, self).__init__()
        self.cfg = cfg
        self.input_channel = cfg["model_cfg"]["input_channel"]
        self.hid_feature = cfg["model_cfg"]["hid_feature"]

        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(self.input_channel, self.hid_feature, (1, 1)),
            nn.InstanceNorm2d(self.hid_feature, affine=True),
            nn.PReLU(self.hid_feature),
        )

        self.dense_block = DenseBlock(cfg, depth=4)

        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(self.hid_feature, self.hid_feature, (1, 3), stride=(1, 2)),
            nn.InstanceNorm2d(self.hid_feature, affine=True),
            nn.PReLU(self.hid_feature),
        )

    def forward(self, x):
        """
        Forward pass for the DenseEncoder module.

        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Encoded tensor.
        """
        x = self.dense_conv_1(x)  # [batch, hid_feature, time, freq]
        x = self.dense_block(x)  # [batch, hid_feature, time, freq]
        x = self.dense_conv_2(x)  # [batch, hid_feature, time, freq//2]
        return x


class MagDecoder(nn.Module):
    """
    MagDecoder module for decoding magnitude information.
    """

    def __init__(self, cfg):
        super(MagDecoder, self).__init__()
        self.dense_block = DenseBlock(cfg, depth=4)
        self.hid_feature = cfg["model_cfg"]["hid_feature"]
        self.output_channel = cfg["model_cfg"]["output_channel"]
        self.n_fft = cfg["stft_cfg"]["n_fft"]
        self.beta = cfg["model_cfg"]["beta"]

        self.mask_conv = nn.Sequential(
            nn.ConvTranspose2d(self.hid_feature, self.hid_feature, (1, 3), stride=(1, 2)),
            nn.Conv2d(self.hid_feature, self.output_channel, (1, 1)),
            nn.InstanceNorm2d(self.output_channel, affine=True),
            nn.PReLU(self.output_channel),
            nn.Conv2d(self.output_channel, self.output_channel, (1, 1)),
        )
        self.lsigmoid = LearnableSigmoid2D(self.n_fft // 2 + 1, beta=self.beta)

    def forward(self, x):
        """
        Forward pass for the MagDecoder module.

        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Decoded tensor with magnitude information.
        """
        x = self.dense_block(x)
        x = self.mask_conv(x)
        x = rearrange(x, "b c t f -> b f t c").squeeze(-1)
        x = self.lsigmoid(x)
        x = rearrange(x, "b f t -> b t f").unsqueeze(1)
        return x


class PhaseDecoder(nn.Module):
    """
    PhaseDecoder module for decoding phase information.
    """

    def __init__(self, cfg):
        super(PhaseDecoder, self).__init__()
        self.dense_block = DenseBlock(cfg, depth=4)
        self.hid_feature = cfg["model_cfg"]["hid_feature"]
        self.output_channel = cfg["model_cfg"]["output_channel"]

        self.phase_conv = nn.Sequential(
            nn.ConvTranspose2d(self.hid_feature, self.hid_feature, (1, 3), stride=(1, 2)),
            nn.InstanceNorm2d(self.hid_feature, affine=True),
            nn.PReLU(self.hid_feature),
        )

        self.phase_conv_r = nn.Conv2d(self.hid_feature, self.output_channel, (1, 1))
        self.phase_conv_i = nn.Conv2d(self.hid_feature, self.output_channel, (1, 1))

    def forward(self, x):
        """
        Forward pass for the PhaseDecoder module.

        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Decoded tensor with phase information.
        """
        x = self.dense_block(x)
        x = self.phase_conv(x)
        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)
        x = torch.atan2(x_i, x_r)
        return x


class SEMamba(nn.Module):
    """
    SEMamba model for speech enhancement using Mamba blocks.

    This model uses a dense encoder, multiple Mamba blocks, and separate magnitude
    and phase decoders to process noisy magnitude and phase inputs.
    """

    def __init__(self, cfg=cfg):
        """
        Initialize the SEMamba model.

        Args:
        - cfg: Configuration object containing model parameters.
        """
        super(SEMamba, self).__init__()
        self.cfg = cfg
        self.num_tscblocks = (
            cfg["model_cfg"]["num_tfmamba"] if cfg["model_cfg"]["num_tfmamba"] is not None else 4
        )  # default tfmamba: 4

        # Initialize dense encoder
        self.dense_encoder = DenseEncoder(cfg)

        # Initialize Mamba blocks
        self.TSMamba = nn.ModuleList([TFMambaBlock(cfg) for _ in range(self.num_tscblocks)])

        # Initialize decoders
        self.mask_decoder = MagDecoder(cfg)
        self.phase_decoder = PhaseDecoder(cfg)

        self.stft = STFT(cfg["stft_cfg"]["n_fft"], cfg["stft_cfg"]["hop_size"])

    def forward(self, inp):
        """
        Forward pass for the SEMamba model.

        Args:
        - noisy_mag (torch.Tensor): Noisy magnitude input tensor [B, F, T].
        - noisy_pha (torch.Tensor): Noisy phase input tensor [B, F, T].

        Returns:
        - denoised_mag (torch.Tensor): Denoised magnitude tensor [B, F, T].
        - denoised_pha (torch.Tensor): Denoised phase tensor [B, F, T].
        - denoised_com (torch.Tensor): Denoised complex tensor [B, F, T, 2].
        """
        xk = self.stft.transform(inp)
        noisy_mag = torch.norm(xk, dim=1).unsqueeze(1)  # b,t,f
        noisy_pha = torch.atan2(xk[:, 1, ...], xk[:, 0, ...]).unsqueeze(1)  # b,t,f
        # Reshape inputs
        # noisy_mag = rearrange(noisy_mag, "b f t -> b t f").unsqueeze(1)  # [B, 1, T, F]
        # noisy_pha = rearrange(noisy_pha, "b f t -> b t f").unsqueeze(1)  # [B, 1, T, F]

        # Concatenate magnitude and phase inputs
        x = torch.cat((noisy_mag, noisy_pha), dim=1)  # [B, 2, T, F]

        # Encode input
        x = self.dense_encoder(x)

        # Apply Mamba blocks
        for block in self.TSMamba:
            x = block(x)

        # Decode magnitude and phase
        denoised_mag = rearrange(self.mask_decoder(x) * noisy_mag, "b c t f -> b f t c").squeeze(-1)
        denoised_pha = rearrange(self.phase_decoder(x), "b c t f -> b f t c").squeeze(-1)

        # Combine denoised magnitude and phase into a complex representation
        denoised_com = torch.stack(
            (denoised_mag * torch.cos(denoised_pha), denoised_mag * torch.sin(denoised_pha)), dim=1
        ).permute(0, 1, 3, 2)

        out = self.stft.inverse(denoised_com)

        return out


class SEMambaFIG6(nn.Module):
    """
    SEMamba model for speech enhancement using Mamba blocks.

    This model uses a dense encoder, multiple Mamba blocks, and separate magnitude
    and phase decoders to process noisy magnitude and phase inputs.
    """

    def __init__(self, cfg=cfg):
        """
        Initialize the SEMamba model.

        Args:
        - cfg: Configuration object containing model parameters.
        """
        super(SEMambaFIG6, self).__init__()
        cfg["model_cfg"]["input_channel"] = 3
        self.cfg = cfg
        self.num_tscblocks = (
            cfg["model_cfg"]["num_tfmamba"] if cfg["model_cfg"]["num_tfmamba"] is not None else 4
        )  # default tfmamba: 4

        # Initialize dense encoder
        self.dense_encoder = DenseEncoder(cfg)

        # Initialize Mamba blocks
        self.TSMamba = nn.ModuleList([TFMambaBlock(cfg) for _ in range(self.num_tscblocks)])

        # Initialize decoders
        self.mask_decoder = MagDecoder(cfg)
        self.phase_decoder = PhaseDecoder(cfg)

        self.stft = STFT(cfg["stft_cfg"]["n_fft"], cfg["stft_cfg"]["hop_size"])
        self.reso = 16000 / cfg["stft_cfg"]["n_fft"]

    def forward(self, inp, HL):
        """
        Forward pass for the SEMamba model.

        Args:
        - noisy_mag (torch.Tensor): Noisy magnitude input tensor [B, F, T].
        - noisy_pha (torch.Tensor): Noisy phase input tensor [B, F, T].

        Returns:
        - denoised_mag (torch.Tensor): Denoised magnitude tensor [B, F, T].
        - denoised_pha (torch.Tensor): Denoised phase tensor [B, F, T].
        - denoised_com (torch.Tensor): Denoised complex tensor [B, F, T, 2].
        """
        xk = self.stft.transform(inp)
        hl = expand_HT(HL, xk.shape[-2], self.reso)  # B,C(1),T,F
        noisy_mag = torch.norm(xk, dim=1).unsqueeze(1)  # b,t,f
        noisy_pha = torch.atan2(xk[:, 1, ...], xk[:, 0, ...]).unsqueeze(1)  # b,t,f
        # Reshape inputs
        # noisy_mag = rearrange(noisy_mag, "b f t -> b t f").unsqueeze(1)  # [B, 1, T, F]
        # noisy_pha = rearrange(noisy_pha, "b f t -> b t f").unsqueeze(1)  # [B, 1, T, F]

        # Concatenate magnitude and phase inputs
        x = torch.cat((noisy_mag, noisy_pha, hl), dim=1)  # [B, 2, T, F]

        # Encode input
        x = self.dense_encoder(x)

        # Apply Mamba blocks
        for block in self.TSMamba:
            x = block(x)

        # Decode magnitude and phase
        denoised_mag = rearrange(self.mask_decoder(x) * noisy_mag, "b c t f -> b f t c").squeeze(-1)
        denoised_pha = rearrange(self.phase_decoder(x), "b c t f -> b f t c").squeeze(-1)

        # Combine denoised magnitude and phase into a complex representation
        denoised_com = torch.stack(
            (denoised_mag * torch.cos(denoised_pha), denoised_mag * torch.sin(denoised_pha)), dim=1
        ).permute(0, 1, 3, 2)

        out = self.stft.inverse(denoised_com)

        return out


if __name__ == "__main__":
    inp = torch.randn(1, 16000).cuda()
    hl = torch.randn(1, 6).cuda()
    net = SEMambaFIG6().cuda()

    check_flops(net, inp, hl)
