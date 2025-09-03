import torch
import torch.nn as nn
import torch.nn.functional as F

import einops
from einops import rearrange
from einops.layers.torch import Rearrange
from models.conv_stft import STFT
from typing import Optional


from utils.register import tables
from models.CMGAN.conformer import (
    DepthWiseConv1d,
    FeedForward,
    calc_same_padding,
    GLU,
    Swish,
)

# compare the effectiveness of down- and up-sample method and sub-band input;
# compare the effectiveness of FT-LSTM and TCN;


class Norm2d(nn.Module):
    def __init__(self, in_channels, feat_size, method="layernorm") -> None:
        super().__init__()

        if method == "layernorm":
            self.norm = nn.LayerNorm([in_channels, feat_size])
        else:
            raise RuntimeError()

    def forward(self, x):
        x = rearrange(x, "b c t f -> b t c f")
        x = self.norm(x)
        x = rearrange(x, "b t c f -> b c t f")
        return x


class SPConvTranspose2d(nn.Module):  # sub-pixel convolution
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        # upconvolution only along second dimension of image
        # Upsampling using sub pixel layers
        super(SPConvTranspose2d, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1))
        self.r = r

    def forward(self, x):
        out = self.conv(x)
        # batch_size, nchannels, H, W = out.shape
        out = rearrange(out, "b (r c) t f-> b c t (f r)", r=self.r)
        # out = out.view((batch_size, self.r, nchannels // self.r, H, W))  # b,r1,r2,h,w
        # out = out.permute(0, 2, 3, 4, 1)  # b,r2,h,w,r1
        # out = out.contiguous().view(
        #     (batch_size, nchannels // self.r, H, -1)
        # )  # b, r2, h, w*r
        return out


class DilatedDenseNet(nn.Module):
    def __init__(self, depth=4, in_channels=64, feat_size=65):
        super(DilatedDenseNet, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.0)
        self.twidth = 2
        self.kernel_size = (self.twidth, 3)
        for i in range(self.depth):
            dil = 2**i
            pad_length = self.twidth + (dil - 1) * (self.twidth - 1) - 1

            setattr(
                self,
                "conv{}".format(i + 1),
                nn.Sequential(
                    nn.ConstantPad2d((1, 1, pad_length, 0), value=0.0),  # lrtb
                    nn.Conv2d(
                        in_channels * (i + 1),
                        in_channels,
                        kernel_size=self.kernel_size,
                        dilation=(dil, 1),
                    ),
                    # nn.LayerNorm(feat_size),
                    Norm2d(in_channels, feat_size),
                    nn.PReLU(in_channels),
                ),
            )

    def forward(self, x):
        skip = x
        out = None
        for i in range(self.depth):
            out = getattr(self, "conv{}".format(i + 1))(skip)
            skip = torch.cat([out, skip], dim=1)
        return out


class DownSample(nn.Module):
    def __init__(self, in_channels: int, feat_size: int) -> None:
        super().__init__()
        self.ds_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(1, 3),
                stride=(1, 2),  # //2
                padding=(0, 1),
            ),
            # Rearrange("b c t f->b t f c"),
            # nn.LayerNorm([feat_size // 2 + 1, in_channels]),
            # Rearrange("b t f c->b c t f"),
            Norm2d(in_channels, feat_size // 2 + 1),
            nn.PReLU(in_channels),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(1, 3),
                stride=(1, 2),  # no padding
            ),
            # Rearrange("b c t f->b t f c"),
            # nn.LayerNorm([feat_size // 4, in_channels]),
            # Rearrange("b t f c->b c t f"),
            Norm2d(in_channels, feat_size // 4),
            nn.PReLU(in_channels),
        )

    def forward(self, x):
        x = self.ds_conv(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, feat_size) -> None:
        super().__init__()
        self.us_conv = nn.Sequential(
            nn.ConstantPad2d((1, 1, 0, 0), value=0.0),  # padding F
            SPConvTranspose2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(1, 3),
                r=2,
            ),  # F//4 => F//2
            # Rearrange("b c t f->b t f c"),
            # nn.LayerNorm([feat_size * 2, in_channels]),
            # Rearrange("b t f c->b c t f"),
            Norm2d(in_channels, feat_size * 2),
            nn.PReLU(in_channels),
            #
            nn.ConstantPad2d((1, 1, 0, 0), value=0.0),
            SPConvTranspose2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(1, 3),
                r=2,
            ),
            nn.ConstantPad2d((1, 0, 0, 0), value=0.0),
            # nn.LayerNorm(feat_size * 4 + 1),
            Norm2d(in_channels, feat_size * 4 + 1),
            nn.PReLU(in_channels),
        )

    def forward(self, x):
        x = self.us_conv(x)
        return x


class DenseEncoder(nn.Module):
    def __init__(self, in_channel, channels=64, feat_size=65):
        super(DenseEncoder, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, channels, (1, 1), (1, 1)),
            # nn.LayerNorm(feat_size),
            Norm2d(channels, feat_size),
            nn.PReLU(channels),
        )
        self.dilated_dense = DilatedDenseNet(depth=4, in_channels=channels, feat_size=feat_size)
        self.ds = DownSample(channels, feat_size)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.dilated_dense(x)
        x = self.ds(x)
        return x


class DenseDecoder(nn.Module):
    def __init__(self, num_channel=64, out_channel=2, feat_size=65):
        super(DenseDecoder, self).__init__()
        self.dense_block = DilatedDenseNet(depth=4, in_channels=num_channel, feat_size=feat_size)
        self.up_sample = UpSample(num_channel, feat_size)
        self.conv = nn.Conv2d(num_channel, out_channel, (1, 1))

    def forward(self, x):
        x = self.dense_block(x)
        x = self.up_sample(x)
        x = self.conv(x)
        return x


class ConformerConvModule(nn.Module):
    def __init__(self, dim, causal=False, expansion_factor=2, kernel_size=31, dropout=0.0):
        super().__init__()

        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange("b n c -> b c n"),
            nn.Conv1d(dim, inner_dim * 2, 1),
            GLU(dim=1),
            DepthWiseConv1d(inner_dim, inner_dim, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(inner_dim) if not causal else nn.Identity(),
            Swish(),
            nn.Conv1d(inner_dim, dim, 1),
            Rearrange("b c n -> b n c"),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConformerBlock(nn.Module):
    """
    input: bxt,f,c or bxf,t,c
    dim: c
    """

    def __init__(
        self,
        dim,
        heads=8,
        ff_mult=4,
        conv_expansion_factor=2,
        conv_kernel_size=31,
        attn_dropout=0.0,
        ff_dropout=0.0,
        conv_dropout=0.0,
        causal=False,
    ):
        super().__init__()
        # self.attn = Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, dropout=attn_dropout, batch_first=True
        )
        self.conv = ConformerConvModule(
            dim=dim,
            causal=causal,
            expansion_factor=conv_expansion_factor,
            kernel_size=conv_kernel_size,
            dropout=conv_dropout,
        )

        self.pre_attn_norm = nn.LayerNorm(dim)

        self.ff1 = nn.Sequential(
            nn.LayerNorm(dim), FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        )
        self.ff2 = nn.Sequential(
            nn.LayerNorm(dim), FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        )

        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        """
        input: bxt,f,c or bxf,t,c
        """
        x = self.pre_attn_norm(0.5 * self.ff1(x) + x)
        attn_out, w = self.attn(x, x, x, attn_mask=mask, need_weights=True)
        x = x + attn_out
        x = self.conv(x) + x
        x = 0.5 * self.ff2(x) + x
        x = self.post_norm(x)
        return x


class TSCB(nn.Module):
    """
    input: bctf
    """

    def __init__(
        self,
        num_channel=64,
        conv_kernel_size=(31, 31),
        attn_wlen: Optional[int] = None,
    ):
        super(TSCB, self).__init__()
        self.time_conformer = ConformerBlock(
            dim=num_channel,
            heads=4,
            conv_kernel_size=conv_kernel_size[0],
            attn_dropout=0.2,
            ff_dropout=0.2,
            causal=True,
        )
        self.freq_conformer = ConformerBlock(
            dim=num_channel,
            heads=4,
            conv_kernel_size=conv_kernel_size[1],
            attn_dropout=0.2,
            ff_dropout=0.2,
            causal=False,
        )
        self.attn_wlen = attn_wlen

    def forward(self, x_in):
        b, c, t, f = x_in.size()
        # bxf,t,c
        x_t = x_in.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)

        mask_1 = torch.ones(t, t, device=x_in.device, dtype=torch.bool).triu_(1)  # TxT
        mask_2 = (
            torch.ones(t, t, device=x_in.device, dtype=torch.bool).tril_(-self.attn_wlen)
            if self.attn_wlen is not None
            else torch.zeros(t, t, device=x_in.device, dtype=torch.bool)
        )
        mask = mask_1 + mask_2

        x_t = self.time_conformer(x_t, mask) + x_t
        # bxt,f,c
        x_f = x_t.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b * t, f, c)
        x_f = self.freq_conformer(x_f) + x_f
        x_f = x_f.view(b, t, f, c).permute(0, 3, 1, 2)
        return x_f


# @tables.register("models", "crn_sample_ftlstm")
# class CRN_FTLSTM(nn.Module):
@tables.register("models", "conformer_base_1")
class CRN_TSCB(nn.Module):
    def __init__(self, nframe=64, nhop=32, mid_channel=20) -> None:
        super().__init__()

        self.stft = STFT(nframe, nhop, nframe, win="hann sqrt")
        feat_size = nframe // 2 + 1

        self.tscb_1 = TSCB(mid_channel, (31, 8))
        self.tscb_2 = TSCB(mid_channel, (31, 8))

        self.encode = DenseEncoder(2, mid_channel, feat_size)
        self.spec_decode = DenseDecoder(mid_channel, 2, feat_size // 4)
        self.mag_decode = nn.Sequential(DenseDecoder(mid_channel, 1, feat_size // 4), nn.PReLU())

    def forward(self, x):
        """
        Input: b,t
        """

        xk = self.stft.transform(x)  # b,2,t,f

        # real, imag = xk.chunk(2, dim=1)
        # mag = torch.sqrt(real**2 + imag**2)
        # noisy_phase = torch.angle(
        #     torch.complex(x[:, 0, :, :], x[:, 1, :, :])
        # ).unsqueeze(1)

        x = self.encode(xk)

        x = self.tscb_1(x)
        x = self.tscb_2(x)

        spec = self.spec_decode(x)
        mag = self.mag_decode(x)

        # r, i = spec[:, 0, ...], spec[:, 1, ...]
        # pha = torch.angle(torch.complex(r, i))
        r, i = spec.chunk(2, dim=1)
        pha = torch.atan2(i, r + 1e-8)

        r = mag * torch.cos(pha)
        i = mag * torch.sin(pha)

        out = torch.concat([r, i], dim=1)
        out = self.stft.inverse(out)
        return out


if __name__ == "__main__":
    from utils.check_flops import check_flops

    x = torch.randn(1, 16000)
    net = CRN_TSCB()
    x = net(x)
    print(x.shape)

    check_flops(net, x)
