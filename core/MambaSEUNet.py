# Reference: https://github.com/huaidanquede/MUSE-Speech-Enhancement/tree/main/models/generator
# Reference: https://github.com/RoyChao19477/SEMamba/models/mamba_block

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from einops import rearrange
from mamba_ssm.models.mixer_seq_simple import _init_weights
from mamba_ssm.modules.mamba_simple import Block, Mamba
from mamba_ssm.ops.triton.layernorm import RMSNorm
from torchvision.ops.deform_conv import DeformConv2d

from JointNSHModel import expand_HT
from models.conv_stft import STFT


# def load_config(config_path="core/MambaSEUNet.yaml"):
def load_config(config_path="MambaSEUNet.yaml"):
    """Load configuration from a YAML file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


class LearnableSigmoid1D(nn.Module):
    """
    Learnable Sigmoid Activation Function for 1D inputs.

    This module applies a learnable slope parameter to the sigmoid activation function.
    """

    def __init__(self, in_features, beta=1):
        """
        Initialize the LearnableSigmoid1D module.

        Args:
        - in_features (int): Number of input features.
        - beta (float, optional): Scaling factor for the sigmoid function. Defaults to 1.
        """
        super(LearnableSigmoid1D, self).__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))
        self.slope.requires_grad = True

    def forward(self, x):
        """
        Forward pass for the LearnableSigmoid1D module.

        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor after applying the learnable sigmoid activation.
        """
        return self.beta * torch.sigmoid(self.slope * x)


class LearnableSigmoid2D(nn.Module):
    """
    Learnable Sigmoid Activation Function for 2D inputs.

    This module applies a learnable slope parameter to the sigmoid activation function for 2D inputs.
    """

    def __init__(self, in_features, beta=1):
        """
        Initialize the LearnableSigmoid2D module.

        Args:
        - in_features (int): Number of input features.
        - beta (float, optional): Scaling factor for the sigmoid function. Defaults to 1.
        """
        super(LearnableSigmoid2D, self).__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1))
        self.slope.requires_grad = True

    def forward(self, x):
        """
        Forward pass for the LearnableSigmoid2D module.

        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor after applying the learnable sigmoid activation.
        """
        return self.beta * torch.sigmoid(self.slope * x)


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
            nn.Conv2d(self.hid_feature, self.hid_feature, (1, 3), stride=(1, 2), padding=(0, 1)),
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
            nn.Conv2d(self.hid_feature, self.hid_feature * 4, 1, 1, 0, bias=False),
            nn.PixelShuffle(2),
            nn.Conv2d(
                self.hid_feature,
                self.hid_feature,
                kernel_size=(1, 3),
                stride=(2, 1),
                padding=(0, 1),
                groups=self.hid_feature,
                bias=False,
            ),
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
            nn.Conv2d(self.hid_feature, self.hid_feature * 4, 1, 1, 0, bias=False),
            nn.PixelShuffle(2),
            nn.Conv2d(
                self.hid_feature,
                self.hid_feature,
                kernel_size=(1, 3),
                stride=(2, 1),
                padding=(0, 1),
                groups=self.hid_feature,
                bias=False,
            ),
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


# github: https://github.com/state-spaces/mamba/blob/9127d1f47f367f5c9cc49c73ad73557089d02cb8/mamba_ssm/models/mixer_seq_simple.py
def create_block(
    d_model,
    cfg,
    layer_idx=0,
    rms_norm=True,
    fused_add_norm=False,
    residual_in_fp32=False,
):
    d_state = cfg["model_cfg"]["d_state"]  # 16
    d_conv = cfg["model_cfg"]["d_conv"]  # 4
    expand = cfg["model_cfg"]["expand"]  # 4
    norm_epsilon = cfg["model_cfg"]["norm_epsilon"]  # 0.00001

    mixer_cls = partial(Mamba, layer_idx=layer_idx, d_state=d_state, d_conv=d_conv, expand=expand)
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon)
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


class BiMambaBlock(nn.Module):
    def __init__(self, in_channels, cfg):
        super().__init__()
        n_layer = 1
        self.forward_blocks = nn.ModuleList(create_block(in_channels, cfg) for i in range(n_layer))
        self.backward_blocks = nn.ModuleList(create_block(in_channels, cfg) for i in range(n_layer))

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
            )
        )

    def forward(self, x):
        x_forward, x_backward = x.clone(), torch.flip(x, [1])
        resi_forward, resi_backward = None, None

        # Forward
        for layer in self.forward_blocks:
            x_forward, resi_forward = layer(x_forward, resi_forward)
        y_forward = (x_forward + resi_forward) if resi_forward is not None else x_forward

        # Backward
        for layer in self.backward_blocks:
            x_backward, resi_backward = layer(x_backward, resi_backward)
        y_backward = (
            torch.flip((x_backward + resi_backward), [1])
            if resi_backward is not None
            else torch.flip(x_backward, [1])
        )

        return torch.cat([y_forward, y_backward], -1)


class MambaBlock(nn.Module):
    def __init__(self, in_channels, cfg):
        super().__init__()
        n_layer = 1
        self.forward_blocks = nn.ModuleList(create_block(in_channels, cfg) for i in range(n_layer))

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
            )
        )

    def forward(self, x):
        x_forward = x
        resi_forward = None

        # Forward
        for layer in self.forward_blocks:
            x_forward, resi_forward = layer(x_forward, resi_forward)
        y = (x_forward + resi_forward) if resi_forward is not None else x_forward

        return y


class TFMambaBlock(nn.Module):
    """
    Temporal-Frequency Mamba block for sequence modeling.

    Attributes:
    cfg (Config): Configuration for the block.
    time_mamba (MambaBlock): Mamba block for temporal dimension.
    freq_mamba (MambaBlock): Mamba block for frequency dimension.
    tlinear (ConvTranspose1d): ConvTranspose1d layer for temporal dimension.
    flinear (ConvTranspose1d): ConvTranspose1d layer for frequency dimension.
    """

    def __init__(self, cfg, inchannels):
        super(TFMambaBlock, self).__init__()
        self.cfg = cfg
        self.hid_feature = inchannels

        # Initialize Mamba blocks
        self.time_mamba = MambaBlock(in_channels=self.hid_feature, cfg=cfg)
        self.freq_mamba = BiMambaBlock(in_channels=self.hid_feature, cfg=cfg)

        # Initialize ConvTranspose1d layers
        # self.tlinear = nn.ConvTranspose1d(self.hid_feature * 2, self.hid_feature, 1, stride=1)
        self.flinear = nn.ConvTranspose1d(self.hid_feature * 2, self.hid_feature, 1, stride=1)

    def forward(self, x):
        """
        Forward pass of the TFMamba block.

        Parameters:
        x (Tensor): Input tensor with shape (batch, channels, time, freq).

        Returns:
        Tensor: Output tensor after applying temporal and frequency Mamba blocks.
        """
        b, c, t, f = x.size()

        x = x.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        # x = self.tlinear(self.time_mamba(x).permute(0, 2, 1)).permute(0, 2, 1) + x
        x = self.time_mamba(x) + x
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b * t, f, c)
        x = self.flinear(self.freq_mamba(x).permute(0, 2, 1)).permute(0, 2, 1) + x
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)
        return x


#####################################
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


class MB_Deform_Embedding(nn.Module):
    def __init__(
        self,
        in_chans=3,
        embed_dim=768,
        patch_size=16,
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


class Patch_Embed_stage(nn.Module):
    """Depthwise Convolutional Patch Embedding stage comprised of
    `DWCPatchEmbed` layers."""

    def __init__(self, in_chans, embed_dim, isPool=False, offset_clamp=(-1, 1)):
        super(Patch_Embed_stage, self).__init__()

        self.patch_embeds = MB_Deform_Embedding(
            in_chans=in_chans,
            embed_dim=embed_dim,
            patch_size=3,
            stride=1,
            offset_clamp=offset_clamp,
        )

    def forward(self, x):
        """foward function"""

        att_inputs = self.patch_embeds(x)

        return att_inputs


#####################################
class Downsample(nn.Module):
    def __init__(self, input_feat, out_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(
            # dw
            nn.Conv2d(
                input_feat,
                input_feat,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=input_feat,
                bias=False,
            ),
            # pw-linear
            nn.Conv2d(input_feat, out_feat // 4, 1, 1, 0, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, input_feat, out_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(
            # dw
            nn.Conv2d(
                input_feat,
                input_feat,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=input_feat,
                bias=False,
            ),
            # pw-linear
            nn.Conv2d(input_feat, out_feat * 4, 1, 1, 0, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class MambaSEUNet(nn.Module):
    """
    SEMamba model for speech enhancement using Mamba blocks.

    This model uses a dense encoder, multiple Mamba blocks, and separate magnitude
    and phase decoders to process noisy magnitude and phase inputs.
    """

    def __init__(self, cfg=load_config()):
        """
        Initialize the SEMamba model.

        Args:
        - cfg: Configuration object containing model parameters.
        """
        super(MambaSEUNet, self).__init__()
        self.cfg = cfg
        self.num_tscblocks = (
            cfg["model_cfg"]["num_tfmamba"] if cfg["model_cfg"]["num_tfmamba"] is not None else 4
        )  # default tfmamba: 4

        self.dim = [
            cfg["model_cfg"]["hid_feature"],
            cfg["model_cfg"]["hid_feature"] * 2,
            cfg["model_cfg"]["hid_feature"] * 3,
        ]
        dim = self.dim

        # Initialize dense encoder
        self.dense_encoder = DenseEncoder(cfg)

        # Initialize Mamba blocks
        self.patch_embed_encoder_level1 = Patch_Embed_stage(dim[0], dim[0])

        self.TSMamba1_encoder = nn.ModuleList(
            [TFMambaBlock(cfg, dim[0]) for _ in range(self.num_tscblocks)]
        )

        self.down1_2 = Downsample(dim[0], dim[1])

        self.patch_embed_encoder_level2 = Patch_Embed_stage(dim[1], dim[1])

        self.TSMamba2_encoder = nn.ModuleList(
            [TFMambaBlock(cfg, dim[1]) for _ in range(self.num_tscblocks)]
        )

        self.down2_3 = Downsample(dim[1], dim[2])

        self.patch_embed_middle = Patch_Embed_stage(dim[2], dim[2])

        self.TSMamba_middle = nn.ModuleList(
            [TFMambaBlock(cfg, dim[2]) for _ in range(self.num_tscblocks)]
        )

        ###########

        self.up3_2 = Upsample(int(dim[2]), dim[1])

        self.concat_level2 = nn.Sequential(
            nn.Conv2d(dim[1] * 2, dim[1], 1, 1, 0, bias=False),
        )

        self.patch_embed_decoder_level2 = Patch_Embed_stage(dim[1], dim[1])

        self.TSMamba2_decoder = nn.ModuleList(
            [TFMambaBlock(cfg, dim[1]) for _ in range(self.num_tscblocks)]
        )

        self.up2_1 = Upsample(int(dim[1]), dim[0])

        self.concat_level1 = nn.Sequential(
            nn.Conv2d(dim[0] * 2, dim[0], 1, 1, 0, bias=False),
        )

        self.patch_embed_decoder_level1 = Patch_Embed_stage(dim[0], dim[0])

        self.TSMamba1_decoder = nn.ModuleList(
            [TFMambaBlock(cfg, dim[0]) for _ in range(self.num_tscblocks)]
        )

        self.mag_patch_embed_refinement = Patch_Embed_stage(dim[0], dim[0])

        self.mag_refinement = nn.ModuleList(
            [TFMambaBlock(cfg, dim[0]) for _ in range(self.num_tscblocks)]
        )

        self.mag_output = nn.Sequential(
            nn.Conv2d(dim[0], dim[0], kernel_size=3, stride=1, padding=1, bias=False),
        )

        self.pha_patch_embed_refinement = Patch_Embed_stage(dim[0], dim[0])

        self.pha_refinement = nn.ModuleList(
            [TFMambaBlock(cfg, dim[0]) for _ in range(self.num_tscblocks)]
        )

        self.pha_output = nn.Sequential(
            nn.Conv2d(dim[0], dim[0], kernel_size=3, stride=1, padding=1, bias=False),
        )

        # Initialize decoders
        self.mask_decoder = MagDecoder(cfg)
        self.phase_decoder = PhaseDecoder(cfg)

    def forward(self, noisy_mag, noisy_pha):
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
        # Reshape inputs
        noisy_mag = rearrange(noisy_mag, "b f t -> b t f").unsqueeze(1)  # [B, 1, T, F]
        noisy_pha = rearrange(noisy_pha, "b f t -> b t f").unsqueeze(1)  # [B, 1, T, F]

        # Concatenate magnitude and phase inputs
        x = torch.cat((noisy_mag, noisy_pha), dim=1)  # [B, 2, T, F]

        # Encode input
        x1 = self.dense_encoder(x)

        # Apply U-Net Mamba blocks
        copy1 = x1
        x1 = self.patch_embed_encoder_level1(x1)
        for block in self.TSMamba1_encoder:
            x1 = block(x1)
        x1 = copy1 + x1

        x2 = self.down1_2(x1)

        copy2 = x2
        x2 = self.patch_embed_encoder_level2(x2)
        for block in self.TSMamba2_encoder:
            x2 = block(x2)
        x2 = copy2 + x2

        x3 = self.down2_3(x2)

        copy3 = x3
        x3 = self.patch_embed_middle(x3)
        for block in self.TSMamba_middle:
            x3 = block(x3)
        x3 = copy3 + x3

        y2 = self.up3_2(x3)
        y2 = torch.cat([y2, x2], 1)
        y2 = self.concat_level2(y2)

        copy_de2 = y2
        y2 = self.patch_embed_decoder_level2(y2)
        for block in self.TSMamba2_decoder:
            y2 = block(y2)
        y2 = copy_de2 + y2

        y1 = self.up2_1(y2)
        y1 = torch.cat([y1, x1], 1)
        y1 = self.concat_level1(y1)

        copy_de1 = y1
        y1 = self.patch_embed_decoder_level1(y1)
        for block in self.TSMamba1_decoder:
            y1 = block(y1)
        y1 = copy_de1 + y1

        mag_input = y1
        pha_input = y1

        # magnitude
        copy_mag = mag_input
        mag_input = self.mag_patch_embed_refinement(mag_input)
        for block in self.mag_refinement:
            mag_input = block(mag_input)
        mag = copy_mag + mag_input
        mag = self.mag_output(mag) + copy1

        # phase
        copy_pha = pha_input
        pha_input = self.pha_patch_embed_refinement(pha_input)
        for block in self.pha_refinement:
            pha_input = block(pha_input)
        pha = copy_pha + pha_input
        pha = self.pha_output(pha) + copy1

        # Decode magnitude and phase
        denoised_mag = rearrange(self.mask_decoder(mag) * noisy_mag, "b c t f -> b f t c").squeeze(
            -1
        )
        denoised_pha = rearrange(self.phase_decoder(pha), "b c t f -> b f t c").squeeze(-1)

        # Combine denoised magnitude and phase into a complex representation
        denoised_com = torch.stack(
            (denoised_mag * torch.cos(denoised_pha), denoised_mag * torch.sin(denoised_pha)), dim=-1
        )

        return denoised_mag, denoised_pha, denoised_com


class MambaSEUNetFIG6(nn.Module):
    """
    SEMamba model for speech enhancement using Mamba blocks.

    This model uses a dense encoder, multiple Mamba blocks, and separate magnitude
    and phase decoders to process noisy magnitude and phase inputs.
    """

    def __init__(self, cfg=load_config()):
        """
        Initialize the SEMamba model.

        Args:
        - cfg: Configuration object containing model parameters.
        """
        super(MambaSEUNetFIG6, self).__init__()
        self.cfg = cfg
        self.num_tscblocks = (
            cfg["model_cfg"]["num_tfmamba"] if cfg["model_cfg"]["num_tfmamba"] is not None else 4
        )  # default tfmamba: 4

        self.dim = [
            cfg["model_cfg"]["hid_feature"],
            cfg["model_cfg"]["hid_feature"] * 2,
            cfg["model_cfg"]["hid_feature"] * 3,
        ]
        dim = self.dim

        # Initialize dense encoder
        self.dense_encoder = DenseEncoder(cfg)

        # Initialize Mamba blocks
        self.patch_embed_encoder_level1 = Patch_Embed_stage(dim[0], dim[0])

        self.TSMamba1_encoder = nn.ModuleList(
            [TFMambaBlock(cfg, dim[0]) for _ in range(self.num_tscblocks)]
        )

        self.down1_2 = Downsample(dim[0], dim[1])

        self.patch_embed_encoder_level2 = Patch_Embed_stage(dim[1], dim[1])

        self.TSMamba2_encoder = nn.ModuleList(
            [TFMambaBlock(cfg, dim[1]) for _ in range(self.num_tscblocks)]
        )

        self.down2_3 = Downsample(dim[1], dim[2])

        self.patch_embed_middle = Patch_Embed_stage(dim[2], dim[2])

        self.TSMamba_middle = nn.ModuleList(
            [TFMambaBlock(cfg, dim[2]) for _ in range(self.num_tscblocks)]
        )

        ###########

        self.up3_2 = Upsample(int(dim[2]), dim[1])

        self.concat_level2 = nn.Sequential(
            nn.Conv2d(dim[1] * 2, dim[1], 1, 1, 0, bias=False),
        )

        self.patch_embed_decoder_level2 = Patch_Embed_stage(dim[1], dim[1])

        self.TSMamba2_decoder = nn.ModuleList(
            [TFMambaBlock(cfg, dim[1]) for _ in range(self.num_tscblocks)]
        )

        self.up2_1 = Upsample(int(dim[1]), dim[0])

        self.concat_level1 = nn.Sequential(
            nn.Conv2d(dim[0] * 2, dim[0], 1, 1, 0, bias=False),
        )

        self.patch_embed_decoder_level1 = Patch_Embed_stage(dim[0], dim[0])

        self.TSMamba1_decoder = nn.ModuleList(
            [TFMambaBlock(cfg, dim[0]) for _ in range(self.num_tscblocks)]
        )

        self.mag_patch_embed_refinement = Patch_Embed_stage(dim[0], dim[0])

        self.mag_refinement = nn.ModuleList(
            [TFMambaBlock(cfg, dim[0]) for _ in range(self.num_tscblocks)]
        )

        self.mag_output = nn.Sequential(
            nn.Conv2d(dim[0], dim[0], kernel_size=3, stride=1, padding=1, bias=False),
        )

        self.pha_patch_embed_refinement = Patch_Embed_stage(dim[0], dim[0])

        self.pha_refinement = nn.ModuleList(
            [TFMambaBlock(cfg, dim[0]) for _ in range(self.num_tscblocks)]
        )

        self.pha_output = nn.Sequential(
            nn.Conv2d(dim[0], dim[0], kernel_size=3, stride=1, padding=1, bias=False),
        )

        # Initialize decoders
        self.mask_decoder = MagDecoder(cfg)
        self.phase_decoder = PhaseDecoder(cfg)

        self.stft = STFT(512, 256, 512)
        self.reso = 16000 / 512

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
        # Reshape inputs
        # noisy_mag = rearrange(noisy_mag, "b f t -> b t f").unsqueeze(1)  # [B, 1, T, F]
        # noisy_pha = rearrange(noisy_pha, "b f t -> b t f").unsqueeze(1)  # [B, 1, T, F]

        xk = self.stft.transform(inp)
        noisy_mag = torch.norm(xk, dim=1).unsqueeze(1)  # b,t,f
        noisy_pha = torch.atan2(xk[:, 1, ...], xk[:, 0, ...]).unsqueeze(1)  # b,t,f

        # Concatenate magnitude and phase inputs
        x = torch.cat((noisy_mag, noisy_pha), dim=1)  # [B, 2, T, F]

        # Encode input
        x1 = self.dense_encoder(x)

        # Apply U-Net Mamba blocks
        copy1 = x1
        x1 = self.patch_embed_encoder_level1(x1)
        for block in self.TSMamba1_encoder:
            x1 = block(x1)
        x1 = copy1 + x1

        x2 = self.down1_2(x1)

        copy2 = x2
        x2 = self.patch_embed_encoder_level2(x2)
        for block in self.TSMamba2_encoder:
            x2 = block(x2)
        x2 = copy2 + x2

        x3 = self.down2_3(x2)

        copy3 = x3
        x3 = self.patch_embed_middle(x3)
        for block in self.TSMamba_middle:
            x3 = block(x3)
        x3 = copy3 + x3

        y2 = self.up3_2(x3)
        y2 = torch.cat([y2, x2], 1)
        y2 = self.concat_level2(y2)

        copy_de2 = y2
        y2 = self.patch_embed_decoder_level2(y2)
        for block in self.TSMamba2_decoder:
            y2 = block(y2)
        y2 = copy_de2 + y2

        y1 = self.up2_1(y2)
        y1 = torch.cat([y1, x1], 1)
        y1 = self.concat_level1(y1)

        copy_de1 = y1
        y1 = self.patch_embed_decoder_level1(y1)
        for block in self.TSMamba1_decoder:
            y1 = block(y1)
        y1 = copy_de1 + y1

        mag_input = y1
        pha_input = y1

        # magnitude
        copy_mag = mag_input
        mag_input = self.mag_patch_embed_refinement(mag_input)
        for block in self.mag_refinement:
            mag_input = block(mag_input)
        mag = copy_mag + mag_input
        mag = self.mag_output(mag) + copy1

        # phase
        copy_pha = pha_input
        pha_input = self.pha_patch_embed_refinement(pha_input)
        for block in self.pha_refinement:
            pha_input = block(pha_input)
        pha = copy_pha + pha_input
        pha = self.pha_output(pha) + copy1

        # Decode magnitude and phase
        denoised_mag = rearrange(self.mask_decoder(mag) * noisy_mag, "b c t f -> b f t c").squeeze(
            -1
        )
        denoised_pha = rearrange(self.phase_decoder(pha), "b c t f -> b f t c").squeeze(-1)

        # Combine denoised magnitude and phase into a complex representation
        denoised_com = torch.stack(
            (denoised_mag * torch.cos(denoised_pha), denoised_mag * torch.sin(denoised_pha)), dim=-1
        )
        print(denoised_com.shape)

        return denoised_mag, denoised_pha, denoised_com


if __name__ == "__main__":
    inp = torch.randn(1, 16000 + 64).cuda()
    HL = torch.randn(1, 6).cuda()
    net = MambaSEUNetFIG6().cuda()
    out, _, _ = net(inp, HL)
