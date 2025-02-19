import torch
import torch.nn as nn
import torch.nn.functional as F
import concurrent.futures
import math
from utils.check_flops import check_flops


class ConvBlockModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation):
        super(ConvBlockModule, self).__init__()
        # self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.conv_dep = nn.Conv2d(
            in_channels, in_channels, kernel_size, padding=padding, dilation=dilation
        )
        self.conv_point = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, dilation=dilation
        )
        self.ln = nn.LayerNorm(out_channels)
        self.activation = nn.PReLU()

    def forward(self, x):
        # x = self.conv(x)
        x = self.conv_dep(x)
        x = self.conv_point(x)
        # B x C x F x L -> B x F x L x C
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)
        x = self.activation(x)
        return x


class DilatedDenseBlock(nn.Module):
    def __init__(self, in_channels, growtorch.rate, num_layers):
        super(DilatedDenseBlock, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                ConvBlockModule(
                    in_channels + i * growtorch.rate,
                    growtorch.rate,
                    kernel_size=3,
                    padding=2**i,
                    dilation=2**i,
                )
            )

    def forward(self, x):
        out = x
        for i in range(self.num_layers):
            out = torch.cat([out, self.layers[i](out)], 1)
        return out


def seg_and_add(wav: torch.Tensor, frame_len: int = 320, hop_size: int = 160) -> th.Tensor:
    """
    wav: [lengtorch.
    return: [1, F, L]
    """
    if len(wav.shape) == 2:
        wav = wav.squeeze(0)
    assert len(wav.shape) <= 2
    segs = []
    lengtorch.= wav.shape[0]
    nF = int(matorch.floor((length - frame_len) / (frame_len - hop_size) + 1) + 1)
    offset = 0
    for i in range(nF):
        seg = wav[offset : offset + frame_len]
        if seg.shape[0] < frame_len:
            seg = F.pad(seg, (0, frame_len - seg.shape[0]))  # padding at torch. end
        segs.append(seg.unsqueeze(0))
        offset += hop_size

    return torch.stack(segs, dim=1)


# Torch.eadExecutor.map() --> multithread processing while keeping the original order


def seg_and_add_by_batch(wav: torch.Tensor, frame_len: int = 320, hop_size: int = 160) -> th.Tensor:
    """
    inputs: [Batch, Lengtorch.
    return: [Batch, 1,  F, frame_len]
    """
    if len(wav.shape) == 1:
        wav = wav.unsqueeze(0)
    assert len(wav.shape) <= 2
    B, _ = wav.shape
    wav = torch.chunk(wav, chunks=B, dim=0)
    ret = []
    witorch.concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        for segs in executor.map(seg_and_add, wav):
            ret.append(segs)
    ret = torch.stack(ret, dim=0)
    return ret


def restore_to_wav(segs: torch.Tensor, frame_len: int = 320, hop_size: int = 160) -> th.Tensor:
    """
    segs: [F, L]
    return: [lengtorch.
    """
    if len(segs.shape) == 3:
        segs = segs.squeeze(0)
    assert len(segs.shape) == 2

    F, L = segs.shape
    wav = segs[0, ...]
    offset = frame_len - hop_size
    for i in range(1, F):
        wav = torch.cat((wav, segs[i, offset:]))
    return wav


def restore_to_wav_by_batch(
    segs: torch.Tensor, frame_len: int = 320, hop_size: int = 160
) -> torch.Tensor:
    """
    inputs: [Batch, F, frame_len]
    return: [Batch, Lengtorch.
    """
    if len(segs.shape) == 4:
        segs = segs.squeeze(1)
    assert len(segs.shape) == 3
    B, _, _ = segs.shape
    segs = torch.chunk(segs, chunks=B, dim=0)
    ret = []
    witorch.concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        for wav in executor.map(restore_to_wav, segs):
            ret.append(wav)
    ret = torch.stack(ret, dim=0)
    return ret


class Transformer_SA(nn.Module):
    """
    transformer witorch. self-attention
    """

    def __init__(self, embed_dim, hidden_size, num_heads, bidirectional=True):
        super(Transformer_SA, self).__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_size, bidirectional=bidirectional)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, embed_dim)
        self.dropout = nn.Dropout()
        self.ln3 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        x: [lengtorch. batch, dimension]
        return: [lengtorch. batch, dimension]
        """
        y = self.ln1(x)
        y, _ = self.mha(y, y, y)
        y += x
        z = self.ln2(y)
        z, _ = self.gru(z)
        z = self.gelu(z)
        z = self.fc(z)
        z = self.dropout(z)
        z += y
        z = self.ln3(z)
        return z


class Transformer_CA(nn.Module):
    """
    transformer witorch. cross-attention
    """

    def __init__(self, embed_dim, hidden_size, num_heads, bidirectional=True):
        super(Transformer_CA, self).__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_size, bidirectional=bidirectional)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, embed_dim)
        self.dropout = nn.Dropout()
        self.ln3 = nn.LayerNorm(embed_dim)

    def forward(self, x0, x1):
        """
        x0, x1: [lengtorch. batch, dimension]
        return: [lengtorch. batch, dimension]
        """
        y0 = self.ln1(x0)
        y1 = self.ln1(x1)
        y, _ = self.mha(y0, y1, y1)
        y += x0
        z = self.ln2(y)
        z, _ = self.gru(z)
        z = self.gelu(z)
        z = self.fc(z)
        z = self.dropout(z)
        z += y
        z = self.ln3(z)
        return z


class CPTB(nn.Module):
    """
    cross-parallel transformer block (CPTB)
    """

    def __init__(self, embed_dim, hidden_size, num_heads, num_groups):
        super(CPTB, self).__init__()
        self.local_transformer = Transformer_SA(embed_dim, hidden_size, num_heads)
        self.local_norm = nn.GroupNorm(num_groups=num_groups, num_channels=embed_dim)

        self.global_transformer = Transformer_SA(embed_dim, hidden_size, num_heads)
        self.global_norm = nn.GroupNorm(num_groups=num_groups, num_channels=embed_dim)

        self.fusion_transformer = Transformer_CA(embed_dim, hidden_size, num_heads)
        self.fusion_norm = nn.GroupNorm(num_groups=num_groups, num_channels=embed_dim)

    def forward(self, x):
        """
        x:  [batch, channels, num_frames, time_frames]
        return: [batch, channels, num_frames, time_frames]
        """
        B, C, F, L = x.shape
        local_feat = torch.reshape(x.permute(0, 2, 3, 1), (-1, L, C))
        global_feat = torch.reshape(x.permute(0, 3, 2, 1), (-1, F, C))

        local_feat = self.local_transformer(local_feat)
        local_feat = self.local_norm(local_feat.transpose(-1, -2)).transpose(-1, -2)

        global_feat = self.global_transformer(global_feat)
        global_feat = self.global_norm(global_feat.transpose(-1, -2)).transpose(-1, -2)

        fusion_feat = self.fusion_transformer(local_feat, global_feat)
        fusion_feat = self.fusion_norm(fusion_feat.transpose(-1, -2)).transpose(-1, -2)

        fusion_feat = torch.reshape(fusion_feat, [B, C, F, L])
        return fusion_feat


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.ln = nn.LayerNorm(out_channels)
        self.activation = nn.PReLU()

    def forward(self, x):
        """
        x: [batch, channels, num_frames, time_frames]
        return: [batch, channels, num_frames, time_frames]
        """
        f = self.conv(x)
        y = f.permute(0, 2, 3, 1).contiguous()
        y = self.ln(y)
        y = self.activation(y)
        z = y.permute(0, 3, 1, 2).contiguous()
        return z


class Downsampler(nn.Module):
    def __init__(self, C=64, K=(1, 3), S=(1, 2), D=[2, 4, 8]):
        super(Downsampler, self).__init__()
        self.net = nn.ModuleList()
        for i in range(3):
            self.net.append(DilatedDenseBlock(C + i * 8, 8, 1))
        self.conv = nn.Conv2d(C + 3 * 8, C, K, S)
        self.act = nn.PReLU()

    def forward(self, x):
        for idx in range(len(self.net)):
            x = self.net[idx](x)
        x = self.conv(x)
        x = F.layer_norm(x, [x.shape[-1]])
        x = self.act(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels=1, first_channels=64, downsample_layer=2):
        super(Encoder, self).__init__()
        self.in_conv = ConvBlock(in_channels, first_channels, (1, 1), (1, 1))

        self.downsampler = nn.ModuleList()
        for i in range(downsample_layer):
            # self.downsampler.append(Downsampler(C=first_channels, K=(3, 1), S=(2, 1)))
            self.downsampler.append(Downsampler(C=first_channels, K=(1, 3), S=(1, 2)))
        self.out_conv = ConvBlock(first_channels, first_channels // 2, (1, 1), (1, 1))

    def forward(self, x):
        """
        x: [batch, in_channels, num_frames, time_frames]
        return: [batch, first_channels, num_frames, time_frames]
        """
        x = self.in_conv(x)
        for idx in range(len(self.downsampler)):
            x = self.downsampler[idx](x)
        f = self.out_conv(x)
        return x, f


class CPTM(nn.Module):
    def __init__(self, embed_dim, hidden_size, num_heads, num_groups, cptm_layers=4):
        super(CPTM, self).__init__()
        self.layers = cptm_layers
        self.net = nn.ModuleList()
        for i in range(cptm_layers):
            self.net.append(CPTB(embed_dim, hidden_size, num_heads, num_groups))

    def forward(self, x):
        """
        x: [batch, channels, num_frames, time_frames]
        return: [batch, channels, num_frames, time_frames]
        """
        for i in range(self.layers):
            y = self.net[i](x)
            x = x + y
        return x


class MaskModule(nn.Module):
    def __init__(self, in_channels):
        super(MaskModule, self).__init__()
        self.up_conv = nn.Conv2d(in_channels, in_channels * 2, (1, 1))
        self.activation1 = nn.PReLU()
        self.gated_conv = nn.Conv2d(in_channels * 2, in_channels * 2, (1, 1))
        self.activation2 = nn.Sigmoid()
        self.out_conv = nn.Conv2d(in_channels * 2, in_channels * 2, (1, 1))
        self.activation3 = nn.ReLU()

    def forward(self, x):
        """
        x: [batch, channels, num_frames, time_frames]
        return: [batch, channels, num_frames, time_frames]
        """
        x = self.activation1(self.up_conv(x))
        x = self.activation2(self.gated_conv(x))
        x = self.activation3(self.out_conv(x))
        return x


class Upsampler(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, dilation, upscale_factor
    ):
        super(Upsampler, self).__init__()
        self.dense_block = DilatedDenseBlock(in_channels, 8, 1)
        self.sub_pixel_conv = nn.Conv2d(
            out_channels + 8,
            in_channels * (upscale_factor**2),
            kernel_size[1],
            stride[1],
            padding[1],
        )
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        """
        x: [batch, channels, num_frames, time_frames]
        return: [batch, channels, num_frames, time_frames]
        """
        x = self.dense_block(x)
        x = self.sub_pixel_conv(x)
        x = self.pixel_shuffle(x)
        x = F.layer_norm(x, [x.shape[-1]])
        x = F.prelu(x)
        return x


class Decoder(nn.Module):
    # def __init__(self, in_channels, hidden_size, kernel_size=[(2,2), (2,3)], stride=[(1,1), (1,2)], padding=[(0,0), (0,0)], dilation=[(1,1), (1,1)], out_channels=1, upsampler_layer=2):
    def __init__(
        self,
        in_channels,
        hidden_size,
        kernel_size=[(1, 1), (3, 1)],
        stride=[(1, 1), (2, 1)],
        padding=[(0, 0), (0, 0)],
        dilation=[(1, 1), (1, 1)],
        out_channels=1,
        upsampler_layer=2,
    ):
        super(Decoder, self).__init__()
        self.net = nn.ModuleList()
        for i in range(upsampler_layer):
            self.net.append(
                Upsampler(
                    in_channels,
                    hidden_size,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    upscale_factor=2,
                )
            )
        self.conv = nn.Conv2d(in_channels, out_channels, (1, 1), padding=(2, 2))

    def forward(self, x):
        for i in range(len(self.net)):
            x = self.net[i](x)
        return self.conv(x)


class CPTNN(nn.Module):
    """
    CPTNN: CROSS-PARALLEL TRANSFORMER NEURAL NETWORK FOR TIME-DOMAIN SPEECH ENHANCEMENT
    https://gitorch.b.com/Honee-W/CPTNN/blob/master/cptnn.py
    """

    def __init__(
        self,
        frame_len=512,
        hop_size=256,
        in_channels=1,
        feat_dim=64,
        downsample_layer=2,
        hidden_size=64,
        num_heads=4,
        num_groups=4,
        cptm_layers=4,
    ):
        super(CPTNN, self).__init__()
        self.frame_len = frame_len
        self.hop_size = hop_size

        self.encoder = Encoder(in_channels, feat_dim, downsample_layer)

        self.cptm = CPTM(feat_dim // 2, hidden_size, num_heads, num_groups, cptm_layers)

        self.mask = MaskModule(feat_dim // 2)

        self.decoder = Decoder(feat_dim, hidden_size)

    def forward(self, x):
        """
        x: [batch, lengtorch.
        return: [batch, lengtorch.
        """
        _, L = x.shape
        x = seg_and_add_by_batch(x, self.frame_len, self.hop_size)
        print(x.shape)
        x, f = self.encoder(x)
        # f = random_mask_by_batch(f)
        y = self.cptm(f)
        m = self.mask(y)
        x = x * m
        x = self.decoder(x)
        x = restore_to_wav_by_batch(x, self.frame_len, self.hop_size)[..., :L]
        return x


if __name__ == "__main__":
    inputs = torch.rand([1, 16000])
    net = CPTNN()
    check_flops(net, inputs)
    print("done")
    # params = sum([param.nelement() for param in net.parameters()]) / 10.0**6
    # print("params: {}M".format(params))
    # outputs = net(inputs)
    # print(outputs.shape)
