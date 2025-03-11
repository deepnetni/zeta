import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import einsum, nn

from JointNSHModel import expand_HT
from models.conv_stft import STFT
from utils.check_flops import check_flops
from utils.register import tables

# source: https://github.com/lucidrains/conformer/blob/master/conformer/conformer.py
# helper functions


def l2_norm(s1, s2):
    # norm = torch.sqrt(torch.sum(s1*s2, 1, keepdim=True))
    # norm = torch.norm(s1*s2, 1, keepdim=True)

    norm = torch.sum(s1 * s2, -1, keepdim=True)
    return norm


def si_snr(s1, s2, eps=1e-8):
    # s1 = remove_dc(s1)
    # s2 = remove_dc(s2)
    s1_s2_norm = l2_norm(s1, s2)
    s2_s2_norm = l2_norm(s2, s2)
    s_target = s1_s2_norm / (s2_s2_norm + eps) * s2
    e_nosie = s1 - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_nosie, e_nosie)
    snr = 10 * torch.log10((target_norm) / (noise_norm + eps) + eps)
    return torch.mean(snr)


def compute_mask_loss(noisy_spec, clean_spec, cmask):
    b, c, d, t = noisy_spec.size()
    # noisy_spec: torch.Size([1, 2, 201, 321])
    Sr = clean_spec[:, 0, :, :]
    Si = clean_spec[:, 1, ::, :]
    # Y = self.stft(noisy)
    Yr = noisy_spec[:, 0, :, :]
    Yi = noisy_spec[:, 1, :, :]
    Y_pow = Yr**2 + Yi**2
    Y_mag = torch.sqrt(Y_pow)
    gth_mask_r = (Sr * Yr + Si * Yi) / (Y_pow + 1e-8)
    gth_mask_i = (Si * Yr - Sr * Yi) / (Y_pow + 1e-8)
    gth_mask_r[gth_mask_r > 2] = 1
    gth_mask_r[gth_mask_r < -2] = -1
    gth_mask_i[gth_mask_i > 2] = 1
    gth_mask_i[gth_mask_i < -2] = -1

    # print('gth_mask_r: {}'.format(gth_mask_r.size()))
    # print('cmask: {}'.format(cmask.size()))
    cmask = cmask.permute(0, 2, 1, 3)
    mask_loss = F.mse_loss(gth_mask_r, cmask[..., 0]) + F.mse_loss(gth_mask_i, cmask[..., 1])
    # phase_loss = F.mse_loss(gth_mask_i, cmp_mask_i) * d #[:,self.feat_dim:, :], cmp_mask[:,self.feat_dim:, :]) * d
    # all_loss = amp_loss + phase_loss
    return mask_loss


def kaiming_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    if isinstance(m, nn.Conv1d):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


def power_compress(x):
    real = x[..., 0]
    imag = x[..., 1]
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = mag**0.3
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return torch.stack([real_compress, imag_compress], 1)


def power_uncompress(real, imag):
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = mag ** (1.0 / 0.3)
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return torch.stack([real_compress, imag_compress], -1)


class LearnableSigmoid(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)


def Csigmoid(x):
    a, b = x[..., 0], x[..., 1]
    denominator = 1 + 2 * torch.exp(-a) * torch.cos(b) + torch.exp(-2 * a)
    real = 1 + torch.exp(-a) * torch.cos(b) / denominator
    imag = torch.exp(-a) * torch.sin(b) / denominator
    return torch.stack((real, imag), dim=-1)


class CSwish(nn.Module):
    def forward(self, x):
        a, b = x[..., 0], x[..., 1]
        c = a.sigmoid()
        d = b.sigmoid()
        # y = Csigmoid(x)
        # c, d = y[...,0], y[...,1]
        return torch.stack((a * c - b * d, a * d + b * c), dim=-1)


class CGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        a, b = out[..., 0], out[..., 1]
        # gate = Csigmoid(gate)
        c, d = gate[..., 0], gate[..., 1]
        c = c.sigmoid()
        d = d.sigmoid()
        # return out * gate.sigmoid()
        return torch.stack((a * c - b * d, a * d + b * c), dim=-1)


class CDepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv_r = nn.Conv1d(chan_in, chan_out, kernel_size, groups=chan_in)
        self.conv_i = nn.Conv1d(chan_in, chan_out, kernel_size, groups=chan_in)

    def forward(self, x):
        a, b = x[..., 0], x[..., 1]
        a = F.pad(a, self.padding)
        b = F.pad(b, self.padding)
        return torch.stack(
            (self.conv_r(a) - self.conv_i(b), self.conv_r(b) + self.conv_i(a)), dim=-1
        )


class CConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size):
        super().__init__()
        self.conv_r = nn.Conv1d(chan_in, chan_out, kernel_size)
        self.conv_i = nn.Conv1d(chan_in, chan_out, kernel_size)

    def forward(self, x):
        a, b = x[..., 0], x[..., 1]
        return torch.stack(
            (self.conv_r(a) - self.conv_i(b), self.conv_r(b) + self.conv_i(a)), dim=-1
        )


# attention, feedforward, and conv module


class Scale(nn.Module):
    def __init__(self, scale, fn):
        super().__init__()
        self.fn = fn
        self.scale = scale

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale


class CPreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm_r = nn.LayerNorm(dim)
        self.norm_i = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        a, b = x[..., 0], x[..., 1]
        a = self.norm_r(a)
        b = self.norm_i(b)
        return self.fn(torch.stack((a, b), dim=-1), **kwargs)


class CLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm_r = nn.LayerNorm(dim)
        self.norm_i = nn.LayerNorm(dim)

    def forward(self, x):
        a, b = x[..., 0], x[..., 1]
        a = self.norm_r(a)
        b = self.norm_i(b)
        return torch.stack((a, b), dim=-1)


class CLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(CLinear, self).__init__()
        self.fc_r = nn.Linear(in_features, out_features, bias=bias)
        self.fc_i = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        a, b = x[..., 0], x[..., 1]
        return torch.stack((self.fc_r(a) - self.fc_i(b), self.fc_r(b) + self.fc_i(a)), dim=-1)


def Csoftmax(x, dim):
    a, b = x[..., 0], x[..., 1]
    return torch.softmax(torch.sqrt(a**2 + b**2), dim=dim)


class CDropout(nn.Module):
    def __init__(self, prob):
        super().__init__()
        self.dropout = nn.Dropout(prob)

    def forward(self, x):
        a, b = x[..., 0:1], x[..., 1:]
        c = torch.ones(x[..., 0:1].size()).to(a)
        c = self.dropout(c)
        return torch.cat((a * c, b * c), dim=-1)


class CBatchNorm1d(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bn_r = nn.BatchNorm1d(num_features=num_features)
        self.bn_i = nn.BatchNorm1d(num_features=num_features)

    def forward(self, x):
        a = self.bn_r(x[..., 0])
        b = self.bn_i(x[..., 1])
        return torch.stack((a, b), dim=-1)


class CAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, max_pos_emb=512):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.to_q = CLinear(dim, inner_dim, bias=False)
        self.to_kv = CLinear(dim, inner_dim * 2, bias=False)
        self.to_out = CLinear(inner_dim, dim)

        self.max_pos_emb = max_pos_emb
        self.rel_pos_emb = nn.Embedding(2 * max_pos_emb + 1, dim_head)

        self.dropout = CDropout(dropout)

    def forward(self, x, context=None, mask=None, context_mask=None):
        n, device, h, max_pos_emb, has_context = (
            x.shape[-3],
            x.device,
            self.heads,
            self.max_pos_emb,
            exists(context),
        )
        context = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-2))
        q, k, v = map(lambda t: rearrange(t, "b n (h d) c -> b h n d c", h=h), (q, k, v))

        q_r, q_i = q[..., 0], q[..., 1]
        k_r, k_i = k[..., 0], k[..., 1]
        dots_r = (
            einsum("b h i d, b h j d -> b h i j", q_r, k_r) * self.scale
            - einsum("b h i d, b h j d -> b h i j", q_i, k_i) * self.scale
        )
        dots_i = (
            einsum("b h i d, b h j d -> b h i j", q_r, k_i) * self.scale
            + einsum("b h i d, b h j d -> b h i j", q_i, k_r) * self.scale
        )

        # shaw's relative positional embedding
        seq = torch.arange(n, device=device)
        dist = rearrange(seq, "i -> i ()") - rearrange(seq, "j -> () j")
        dist = dist.clamp(-max_pos_emb, max_pos_emb) + max_pos_emb
        rel_pos_emb = self.rel_pos_emb(dist).to(q_r)
        pos_attn_r = einsum("b h n d, n r d -> b h n r", q_r, rel_pos_emb) * self.scale
        pos_attn_i = einsum("b h n d, n r d -> b h n r", q_i, rel_pos_emb) * self.scale

        dots_r = dots_r + pos_attn_r
        dots_i = dots_i + pos_attn_i

        if exists(mask) or exists(context_mask):
            mask = default(mask, lambda: torch.ones(*x.shape[:2], device=device))
            context_mask = (
                default(context_mask, mask)
                if not has_context
                else default(context_mask, lambda: torch.ones(*context.shape[:2], device=device))
            )
            mask_value = -torch.finfo(dots.dtype).max
            mask = rearrange(mask, "b i -> b () i ()") * rearrange(context_mask, "b j -> b () () j")
            dots_r.masked_fill_(~mask, mask_value)
            dots_i.masked_fill_(~mask, mask_value)

        attn = Csoftmax(torch.stack((dots_r, dots_i), dim=-1), dim=-1)
        v_r, v_i = v[..., 0], v[..., 1]

        out_r = einsum("b h i j, b h j d -> b h i d", attn, v_r)
        out_i = einsum("b h i j, b h j d -> b h i d", attn, v_i)
        out_r = rearrange(out_r, "b h n d -> b n (h d)")
        out_i = rearrange(out_i, "b h n d -> b n (h d)")
        out = self.to_out(torch.stack((out_r, out_i), dim=-1))
        return self.dropout(out)


class CFeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            CLinear(dim, dim * mult),
            CSwish(),
            CDropout(dropout),
            CLinear(dim * mult, dim),
            CDropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class CConformerConvModule(nn.Module):
    def __init__(self, dim, causal=False, expansion_factor=2, kernel_size=31, dropout=0.0):
        super().__init__()

        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        self.net = nn.Sequential(
            CLayerNorm(dim),
            Rearrange("b n c d -> b c n d"),
            CConv1d(dim, inner_dim * 2, 1),
            CGLU(dim=1),
            CDepthWiseConv1d(inner_dim, inner_dim, kernel_size=kernel_size, padding=padding),
            CBatchNorm1d(inner_dim) if not causal else nn.Identity(),
            CSwish(),
            CConv1d(inner_dim, dim, 1),
            Rearrange("b c n d -> b n c d"),
            CDropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# Conformer Block


class CConformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head=64,
        heads=8,
        ff_mult=4,
        conv_expansion_factor=2,
        conv_kernel_size=31,
        attn_dropout=0.0,
        ff_dropout=0.0,
        conv_dropout=0.0,
    ):
        super().__init__()
        self.ff1 = CFeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        self.attn = CAttention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout)
        self.conv = CConformerConvModule(
            dim=dim,
            causal=False,
            expansion_factor=conv_expansion_factor,
            kernel_size=conv_kernel_size,
            dropout=conv_dropout,
        )
        self.ff2 = CFeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)

        self.attn = CPreNorm(dim, self.attn)
        self.ff1 = Scale(0.5, CPreNorm(dim, self.ff1))
        self.ff2 = Scale(0.5, CPreNorm(dim, self.ff2))

        self.post_norm = CLayerNorm(dim)

    def forward(self, x, mask=None):
        x = self.ff1(x) + x
        x = self.attn(x, mask=mask) + x
        x = self.conv(x) + x
        x = self.ff2(x) + x
        x = self.post_norm(x)
        return x


class UniDeepFsmn(nn.Module):
    def __init__(self, input_dim, output_dim, lorder=None, hidden_size=None):
        super(UniDeepFsmn, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        if lorder is None:
            return

        self.lorder = lorder
        self.hidden_size = hidden_size

        self.linear = nn.Linear(input_dim, hidden_size)

        self.project = nn.Linear(hidden_size, output_dim, bias=False)

        self.conv1 = nn.Conv2d(
            output_dim, output_dim, [lorder, 1], [1, 1], groups=output_dim, bias=False
        )

    def forward(self, input):
        ## input: batch (b) x sequence(T) x feature (h)
        f1 = F.relu(self.linear(input))

        p1 = self.project(f1)

        x = torch.unsqueeze(p1, 1)
        # x: batch (b) x channel (c) x sequence(T) x feature (h)
        x_per = x.permute(0, 3, 2, 1)
        # x_per: batch (b) x feature (h) x sequence(T) x channel (c)
        y = F.pad(x_per, [0, 0, self.lorder - 1, 0])

        out = x_per + self.conv1(y)

        out1 = out.permute(0, 3, 2, 1)
        # out1: batch (b) x channel (c) x sequence(T) x feature (h)
        return input + out1.squeeze(1)


class CFsmn(nn.Module):
    def __init__(self, nIn, nHidden=128, nOut=128):
        super(CFsmn, self).__init__()

        self.fsmn_re_L1 = UniDeepFsmn(nIn, nHidden, 20, nHidden)
        self.fsmn_im_L1 = UniDeepFsmn(nIn, nHidden, 20, nHidden)

    def forward(self, x):
        # # shpae of input x : [b,c,h,T,2], [6, 256, 1, 106, 2]
        b, c, T, h, d = x.size()
        # x : [b,T,h,c,2]
        x = x.permute(0, 2, 3, 1, 4)
        x = torch.reshape(x, (b * T, h, c, d))

        real = self.fsmn_re_L1(x[..., 0]) - self.fsmn_im_L1(x[..., 1])
        imaginary = self.fsmn_re_L1(x[..., 1]) + self.fsmn_im_L1(x[..., 0])
        # output: [b*T,h,c,2], [6*106, h, 256, 2]
        output = torch.stack((real, imaginary), dim=-1)

        output = torch.reshape(output, (b, T, h, c, d))
        # output: [b,c,h,T,2], [6, 99, 1024, 2]
        # output = torch.transpose(output, 1, 3)

        return output.permute(0, 3, 1, 2, 4)


class CConstantPad2d(nn.Module):
    def __init__(self, padding, value):
        super(CConstantPad2d, self).__init__()
        self.padding = padding
        self.value = value
        self.pad_r = nn.ConstantPad2d(self.padding, self.value)
        self.pad_i = nn.ConstantPad2d(self.padding, self.value)

    def forward(self, x):
        a, b = x[..., 0], x[..., 1]
        return torch.stack((self.pad_r(a), self.pad_i(b)), dim=-1)


class CConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        use_fsmn=True,
        **kwargs,
    ):
        super().__init__()
        self.use_fsmn = use_fsmn
        if use_fsmn:
            self.fsmn = CFsmn(nIn=out_channel, nHidden=out_channel, nOut=out_channel)
        ## Model components
        self.conv_r = nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            **kwargs,
        )
        self.conv_i = nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            **kwargs,
        )

    def forward(self, x):
        a, b = x[..., 0], x[..., 1]
        real = self.conv_r(a) - self.conv_i(b)
        imag = self.conv_r(b) + self.conv_i(a)
        out = torch.stack((real, imag), dim=-1)
        if self.use_fsmn:
            out = self.fsmn(out)
        return out


class CInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=False):
        super().__init__()
        self.in_r = nn.InstanceNorm2d(
            num_features=num_features, eps=eps, momentum=momentum, affine=affine
        )
        self.in_i = nn.InstanceNorm2d(
            num_features=num_features, eps=eps, momentum=momentum, affine=affine
        )

    def forward(self, x):
        a = self.in_r(x[..., 0])
        b = self.in_i(x[..., 1])
        return torch.stack((a, b), dim=-1)


class CDilatedDenseNet(nn.Module):
    def __init__(self, depth=4, in_channels=64):
        super(CDilatedDenseNet, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.twidth = 2
        self.kernel_size = (self.twidth, 3)
        for i in range(self.depth):
            dil = 2**i
            pad_length = self.twidth + (dil - 1) * (self.twidth - 1) - 1
            setattr(self, "pad{}".format(i + 1), CConstantPad2d((1, 1, pad_length, 0), value=0.0))
            setattr(
                self,
                "conv{}".format(i + 1),
                CConv2d(
                    self.in_channels * (i + 1),
                    self.in_channels,
                    kernel_size=self.kernel_size,
                    dilation=(dil, 1),
                    use_fsmn=False,
                ),
            )
            setattr(self, "norm{}".format(i + 1), CInstanceNorm2d(in_channels, affine=True))
            setattr(self, "prelu{}".format(i + 1), nn.PReLU(self.in_channels))
            setattr(
                self,
                "cfsmn{}".format(i + 1),
                CFsmn(nIn=self.in_channels, nHidden=self.in_channels, nOut=self.in_channels),
            )

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, "pad{}".format(i + 1))(skip)
            out = getattr(self, "conv{}".format(i + 1))(out)
            out = getattr(self, "norm{}".format(i + 1))(out)
            out = getattr(self, "prelu{}".format(i + 1))(out)
            out = getattr(self, "cfsmn{}".format(i + 1))(out)
            skip = torch.cat([out, skip], dim=1)
        return out


class CDenseEncoder(nn.Module):
    def __init__(self, in_channel, channels=64):
        super(CDenseEncoder, self).__init__()
        self.conv_1 = nn.Sequential(
            CConv2d(in_channel, channels, (1, 1), (1, 1), use_fsmn=False),
            CInstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )
        self.dilated_dense = CDilatedDenseNet(depth=4, in_channels=channels)
        self.conv_2 = nn.Sequential(
            CConv2d(channels, channels, (1, 3), (1, 2), padding=(0, 1), use_fsmn=False),
            CInstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.dilated_dense(x)
        x = self.conv_2(x)
        return x


class TSCB(nn.Module):
    def __init__(self, num_channel=64):
        super(TSCB, self).__init__()
        self.time_conformer = CConformerBlock(
            dim=num_channel,
            dim_head=num_channel // 4,
            heads=4,
            conv_kernel_size=31,
            attn_dropout=0.2,
            ff_dropout=0.2,
        )
        self.freq_conformer = CConformerBlock(
            dim=num_channel,
            dim_head=num_channel // 4,
            heads=4,
            conv_kernel_size=31,
            attn_dropout=0.2,
            ff_dropout=0.2,
        )

    def forward(self, x_in):
        # x_in: torch.Size([1, 64, 321, 101, 2])
        b, c, t, f, d = x_in.size()
        x_t = x_in.permute(0, 3, 2, 1, 4).contiguous().view(b * f, t, c, d)
        x_t = self.time_conformer(x_t) + x_t
        x_f = x_t.view(b, f, t, c, d).permute(0, 2, 1, 3, 4).contiguous().view(b * t, f, c, d)
        x_f = self.freq_conformer(x_f) + x_f
        x_f = x_f.view(b, t, f, c, d).permute(0, 3, 1, 2, 4)
        return x_f


class CSPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        super(CSPConvTranspose2d, self).__init__()
        self.pad1 = CConstantPad2d((1, 1, 0, 0), value=0.0)
        self.out_channels = out_channels
        self.conv = CConv2d(
            in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1), use_fsmn=False
        )
        self.r = r

    def forward(self, x):
        x = self.pad1(x)
        out = self.conv(x)
        batch_size, nchannels, H, W, C = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W, C))
        out = out.permute(0, 2, 3, 4, 1, 5)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1, C))
        return out


class CMaskDecoder(nn.Module):
    def __init__(self, num_features, num_channel=64, out_channel=1):
        super(CMaskDecoder, self).__init__()
        self.dense_block = CDilatedDenseNet(depth=4, in_channels=num_channel)
        self.sub_pixel = CSPConvTranspose2d(num_channel, num_channel, (1, 3), 2)
        self.conv_1 = CConv2d(num_channel, out_channel, (1, 2), use_fsmn=False)
        self.norm = CInstanceNorm2d(out_channel, affine=True)
        self.relu = nn.LeakyReLU(inplace=True)
        self.final_conv = CConv2d(out_channel, out_channel, (1, 1), use_fsmn=False)

    def forward(self, x):
        x = self.dense_block(x)
        x = self.sub_pixel(x)
        x = self.conv_1(x)
        x = self.relu(self.norm(x))
        x = self.final_conv(x)
        return torch.tanh(x)


class ComplexDecoder(nn.Module):
    def __init__(self, num_channel=64):
        super(ComplexDecoder, self).__init__()
        self.dense_block = CDilatedDenseNet(depth=4, in_channels=num_channel)
        self.sub_pixel = CSPConvTranspose2d(num_channel, num_channel, (1, 3), 2)
        self.prelu = nn.PReLU(num_channel)
        self.norm = CInstanceNorm2d(num_channel, affine=True)
        self.conv = CConv2d(num_channel, 1, (1, 2), use_fsmn=False)

    def forward(self, x):
        x = self.dense_block(x)
        x = self.sub_pixel(x)
        x = self.prelu(self.norm(x))
        x = self.conv(x)
        return x


class TSCNet(nn.Module):
    def __init__(self, num_channel=64, nframe=512, nhop=256):
        num_features = nhop + 1
        super(TSCNet, self).__init__()
        self.alpha = 0.75
        self.beta = 0.25
        self.dense_encoder = CDenseEncoder(in_channel=1, channels=num_channel)

        self.TSCB_1 = TSCB(num_channel=num_channel)
        self.TSCB_2 = TSCB(num_channel=num_channel)
        self.TSCB_3 = TSCB(num_channel=num_channel)

        self.mask_decoder = CMaskDecoder(num_features, num_channel=num_channel, out_channel=1)
        self.complex_decoder = ComplexDecoder(num_channel=num_channel)

        self.stft = STFT(nframe, nhop)

    def forward(self, inp):
        x = self.stft.transform(inp)  # b,2,t,f
        # noisy_phase = torch.angle(torch.complex(x[:, 0, :, :], x[:, 1, :, :])).unsqueeze(1)
        x_in = x.permute(0, 2, 3, 1).unsqueeze(1)  # b,1,t,f,c
        a, b = x_in[..., 0], x_in[..., 1]  # b,1,t,f

        out_1 = self.dense_encoder(x_in)
        out_2 = self.TSCB_1(out_1)
        out_3 = self.TSCB_2(out_2)
        out_4 = self.TSCB_3(out_3)

        cmask = self.mask_decoder(out_4)
        c, d = cmask[..., 0], cmask[..., 1]

        masked_r = a * c - b * d
        masked_i = a * d + b * c
        complex_out = self.complex_decoder(out_4)

        final_real = self.alpha * masked_r + self.beta * complex_out[..., 0]
        final_imag = self.alpha * masked_i + self.beta * complex_out[..., 1]

        out_spec = torch.concat([final_real, final_imag], dim=1)
        out = self.stft.inverse(out_spec)

        return out


@tables.register("models", "D2FormerFIG")
class TSCNetFIG(nn.Module):
    def __init__(self, num_channel=64, nframe=512, nhop=256):
        num_features = nhop + 1
        super().__init__()
        self.alpha = 0.75
        self.beta = 0.25
        self.dense_encoder = CDenseEncoder(in_channel=2, channels=num_channel)

        self.TSCB_1 = TSCB(num_channel=num_channel)
        self.TSCB_2 = TSCB(num_channel=num_channel)
        # self.TSCB_3 = TSCB(num_channel=num_channel)

        self.mask_decoder = CMaskDecoder(num_features, num_channel=num_channel, out_channel=1)
        self.complex_decoder = ComplexDecoder(num_channel=num_channel)

        self.stft = STFT(nframe, nhop)
        self.reso = 16000 / nframe

    def forward(self, inp, HL):
        x = self.stft.transform(inp)  # b,2,t,f
        hl = expand_HT(HL, x.shape[-2], self.reso)  # B,C(1),T,F
        x_in = x.permute(0, 2, 3, 1).unsqueeze(1)  # b,1,t,f,c
        hl = hl.permute(0, 2, 3, 1).unsqueeze(1)  # b,1,t,f,c
        hl = hl.repeat(1, 1, 1, 1, 2)
        a, b = x_in[..., 0], x_in[..., 1]  # b,1,t,f

        x_in = torch.concat([x_in, hl], dim=1)

        out_1 = self.dense_encoder(x_in)
        out_2 = self.TSCB_1(out_1)
        out_3 = self.TSCB_2(out_2)
        # out_4 = self.TSCB_3(out_3)

        cmask = self.mask_decoder(out_3)
        c, d = cmask[..., 0], cmask[..., 1]

        masked_r = a * c - b * d
        masked_i = a * d + b * c
        complex_out = self.complex_decoder(out_3)

        final_real = self.alpha * masked_r + self.beta * complex_out[..., 0]
        final_imag = self.alpha * masked_i + self.beta * complex_out[..., 1]

        out_spec = torch.concat([final_real, final_imag], dim=1)
        out = self.stft.inverse(out_spec)

        return out


if __name__ == "__main__":
    inp = torch.randn(1, 16000)
    hl = torch.randn(1, 6)
    net = TSCNetFIG(32)
    check_flops(net, inp, hl)

    # net = TSCNet(32)
    # check_flops(net, inp)
