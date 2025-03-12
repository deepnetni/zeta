import torch
from torch import Tensor, nn

from einops.layers.torch import Rearrange
from .models.conv_stft import STFT
from .utils.check_flops import check_flops
from .utils.register import tables
from einops.layers.torch import Rearrange
from core.JointNSHModel import HLModule


def expand_HT(ht: torch.Tensor, T: int, reso):
    """
    ht: B,6
    output: B,c(1),T,nbin
    """
    # batch_size = ht.shape[0]
    # Freq_size = self.nbin

    m = int(250 / reso)
    bandarray = torch.tensor([0] + [(2**i) * m for i in range(ht.shape[1])]).to(ht.device)

    repeat_n = bandarray[1:] - bandarray[:-1]
    repeat_n[0] += 1

    expand_ht = ht.repeat_interleave(repeat_n, dim=-1).unsqueeze(1).unsqueeze(1)  # B,1,1,nbin
    expand_ht = expand_ht.repeat(1, 1, T, 1) / 100.0

    return expand_ht


def expand_HT_norepeat(ht: torch.Tensor, T: int, reso):
    """
    ht: B,6
    output: B,c(1),T,nbin
    """
    # batch_size = ht.shape[0]
    # Freq_size = self.nbin

    m = int(250 / reso)
    bandarray = torch.tensor([0] + [(2**i) * m for i in range(ht.shape[1])]).to(ht.device)

    repeat_n = bandarray[1:] - bandarray[:-1]
    repeat_n[0] += 1

    expand_ht = ht.repeat_interleave(repeat_n, dim=-1).unsqueeze(1).unsqueeze(1)  # B,1,1,nbin
    expand_ht = expand_ht / 100.0

    return expand_ht


class LearnableSigmoid(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


class Discriminator(nn.Module):
    def __init__(self, ndf, nframe=512, nhop=256, in_channel=3):
        super().__init__()
        # self.fcs_in = nn.Linear(6, 257)
        self.stft = STFT(nframe, nhop)
        self.nbin = nframe // 2 + 1
        self.reso = 16000 / nframe
        assert self.reso < 250 and 250 % self.reso == 0

        self.layers = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channel, ndf, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(ndf, affine=True),
            nn.PReLU(ndf),
            nn.utils.spectral_norm(nn.Conv2d(ndf, ndf * 2, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(ndf * 2, affine=True),
            nn.PReLU(2 * ndf),
            nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(ndf * 4, affine=True),
            nn.PReLU(4 * ndf),
            nn.utils.spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(ndf * 8, affine=True),
            nn.PReLU(8 * ndf),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(ndf * 8, ndf * 4)),
            nn.Dropout(0.3),
            nn.PReLU(4 * ndf),
            nn.utils.spectral_norm(nn.Linear(ndf * 4, 1)),
            LearnableSigmoid(1),
        )

    def forward(self, x, y, ht):
        """
        x: clean, y: enh, with shape B,T
        ht: B,6
        """
        x_spec = self.stft.transform(x)  # b,2,t,f
        y_spec = self.stft.transform(y)  # b,2,t,f

        x_mag = torch.sum(x_spec**2, dim=1)  # b,t,f
        y_mag = torch.sum(y_spec**2, dim=1)
        ht = expand_HT(ht, x_mag.shape[-2], self.reso)  # b,1,t,f

        x = x_mag.unsqueeze(1)  # b,t,f -> b,1,t,f
        y = y_mag.unsqueeze(1)
        # ht_embed = self.fcs_in(ht).unsqueeze(1)
        xy = torch.cat([x, y], dim=1)
        xy_ht = torch.cat([xy, ht], dim=1)
        return self.layers(xy_ht)


class DPRNN_Block(nn.Module):
    def __init__(self, numUnits: int, width: int):
        super().__init__()
        self.numUnits = numUnits

        self.width = width

        self.intra_rnn = nn.LSTM(
            input_size=self.numUnits,
            hidden_size=self.numUnits // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.intra_fc = nn.Linear(self.numUnits, self.numUnits)

        self.intra_ln = nn.LayerNorm(normalized_shape=[self.width, self.numUnits])

        self.inter_rnn = nn.LSTM(
            input_size=self.numUnits, hidden_size=self.numUnits, num_layers=1, batch_first=True
        )
        self.inter_fc = nn.Linear(self.numUnits, self.numUnits)

        self.inter_ln = nn.LayerNorm(normalized_shape=[self.width, self.numUnits])

    def forward(self, input: Tensor) -> Tensor:
        # input shape: [B, C, T, F]

        # Intra-Chunk Processing

        intra_RNN_input = input.permute(0, 2, 3, 1)  ## [B, T, F, C]
        intra_RNN_input_rs = intra_RNN_input.reshape(
            intra_RNN_input.size()[0] * intra_RNN_input.size()[1],
            intra_RNN_input.size()[2],
            intra_RNN_input.size()[3],
        )

        intra_RNN_output, _ = self.intra_rnn(intra_RNN_input_rs)
        intra_dense_out = self.intra_fc(intra_RNN_output)

        intra_ln_input = intra_dense_out.reshape(
            intra_RNN_input.size()[0],
            intra_RNN_input.size()[1],
            intra_RNN_input.size()[2],
            intra_RNN_input.size()[3],
        )
        intra_ln_out = self.intra_ln(intra_ln_input)

        intra_out = intra_ln_out.permute(0, 3, 1, 2)

        intra_out = intra_out + input

        # Inter-Chunk Processing

        inter_RNN_input = intra_out.permute(0, 3, 2, 1)  ## [B, F, T, C]
        inter_RNN_input_rs = inter_RNN_input.reshape(
            inter_RNN_input.size()[0] * inter_RNN_input.size()[1],
            inter_RNN_input.size()[2],
            inter_RNN_input.size()[3],
        )
        inter_RNN_output, _ = self.inter_rnn(inter_RNN_input_rs)
        inter_dense_out = self.inter_fc(inter_RNN_output)
        inter_ln_input = inter_dense_out.reshape(
            inter_RNN_input.size()[0],
            inter_RNN_input.size()[1],
            inter_RNN_input.size()[2],
            inter_RNN_input.size()[3],
        )
        inter_ln_input = inter_ln_input.permute(0, 2, 1, 3)
        inter_ln_out = self.inter_ln(inter_ln_input)
        inter_out = inter_ln_out.permute(0, 3, 1, 2)

        output = inter_out + intra_out

        return output


class DPRNN_Block_Conditional(nn.Module):
    def __init__(self, numUnits: int, width: int, ff_mult=4):
        super().__init__()
        self.numUnits = numUnits

        self.width = width

        self.intra_rnn = nn.LSTM(
            input_size=self.numUnits,
            hidden_size=self.numUnits // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.intra_fc = nn.Linear(self.numUnits, self.numUnits)
        self.intra_ln = nn.LayerNorm(normalized_shape=[self.width, self.numUnits])

        self.inter_rnn = nn.LSTM(
            input_size=self.numUnits, hidden_size=self.numUnits, num_layers=1, batch_first=True
        )
        self.inter_fc = nn.Linear(self.numUnits, self.numUnits)
        self.inter_ln = nn.LayerNorm(normalized_shape=[self.width, self.numUnits])

        dim = numUnits
        self.adaLN_modulation_f = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        self.norm1_f = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2_f = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp_f = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * ff_mult, dim),
        )

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * ff_mult, dim),
        )

    @staticmethod
    def modulate(x, shift, scale):
        """

        :param x: b,t,c
        :param shift: b,c
        :param scale: b,c
        :returns:

        """
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    @staticmethod
    def modulate_f(x, shift, scale):
        return x * (1 + scale) + shift

    def forward(self, input: Tensor, c: Tensor) -> Tensor:
        # input shape: [B, C, T, F]
        nB = input.size(0)
        nT = input.size(-2)

        # Intra-Chunk Processing
        (
            shift_msa_f,
            scale_msa_f,
            gate_msa_f,
            shift_mlp_f,
            scale_mlp_f,
            gate_mlp_f,
        ) = self.adaLN_modulation_f(c).chunk(6, dim=-1)

        shift_msa_f = shift_msa_f.repeat_interleave(nT, dim=0)  # bt,f,c
        scale_msa_f = scale_msa_f.repeat_interleave(nT, dim=0)
        gate_msa_f = gate_msa_f.repeat_interleave(nT, dim=0)
        shift_mlp_f = shift_mlp_f.repeat_interleave(nT, dim=0)
        scale_mlp_f = scale_mlp_f.repeat_interleave(nT, dim=0)
        gate_mlp_f = gate_mlp_f.repeat_interleave(nT, dim=0)

        intra_RNN_input = input.permute(0, 2, 3, 1)  ## [B, T, F, C]
        # BT,F,C
        intra_RNN_input_rs = intra_RNN_input.reshape(
            -1, intra_RNN_input.size(2), intra_RNN_input.size(3)
        )
        x_ = self.modulate_f(self.norm1_f(intra_RNN_input_rs), shift_msa_f, scale_msa_f)
        intra_RNN_output, _ = self.intra_rnn(x_)
        intra_dense_out = self.intra_fc(intra_RNN_output)
        intra_dense_out = intra_dense_out * gate_msa_f + intra_RNN_input_rs
        intra_ln_input = intra_dense_out.reshape(
            nB, nT, intra_RNN_input.size(2), intra_RNN_input.size(3)
        )
        intra_ln_out = self.intra_ln(intra_ln_input)
        intra_out = intra_ln_out.permute(0, 3, 1, 2)  # B,C,T,F
        # intra_out = intra_out + input

        # Inter-Chunk Processing
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            c.view(-1, c.size(-1)).contiguous()
        ).chunk(6, dim=1)

        inter_RNN_input = intra_out.permute(0, 3, 2, 1)  ## [B, F, T, C]
        # BF,T,C
        inter_RNN_input_rs = inter_RNN_input.reshape(
            -1, inter_RNN_input.size(2), inter_RNN_input.size(3)
        )
        x_ = self.modulate(self.norm1(inter_RNN_input_rs), shift_msa, scale_msa)
        inter_RNN_output, _ = self.inter_rnn(x_)
        inter_dense_out = self.inter_fc(inter_RNN_output)
        inter_dense_out = inter_dense_out * gate_msa.unsqueeze(1) + inter_RNN_input_rs
        x_ = gate_mlp.unsqueeze(1) * self.mlp(
            self.modulate(self.norm2(inter_dense_out), shift_mlp, scale_mlp)
        )
        inter_dense_out = inter_dense_out + x_
        inter_ln_input = inter_dense_out.reshape(
            nB, inter_RNN_input.size(1), nT, inter_RNN_input.size(3)
        )
        inter_ln_input = inter_ln_input.permute(0, 2, 1, 3)
        inter_ln_out = self.inter_ln(inter_ln_input)
        output = inter_ln_out.permute(0, 3, 1, 2)
        # output = inter_out + intra_out

        return output


@tables.register("models", "FTCRN")
class MGAN_G(nn.Module):
    """
    Cheng, J., Liang, R., Zhao, L., Huang, C. and Schuller, B.W., 2023. Speech denoising and compensation for hearing aids using an FTCRN-based metric GAN. IEEE Signal Processing Letters, 30, pp.374-378.
    """

    def __init__(self, nframe=512, nhop=256):
        super().__init__()

        self.stft = STFT(nframe, nhop, nframe)
        self.nbin = nframe // 2 + 1
        self.reso = 16000 / nframe
        assert self.reso < 250 and 250 % self.reso == 0

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=32, kernel_size=(1, 5), stride=(1, 2), padding=(0, 1)
            ),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)
            ),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)
            ),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)
            ),
            nn.BatchNorm2d(64),
            nn.PReLU(),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)
            ),
            nn.BatchNorm2d(128),
            nn.PReLU(),
        )

        self.DPRNN_1 = DPRNN_Block(numUnits=128, width=64)
        self.DPRNN_2 = DPRNN_Block(numUnits=128, width=64)

        self.convT1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256, out_channels=64, kernel_size=(1, 3), stride=(1, 1), padding=(0, 0)
            ),
            nn.BatchNorm2d(64),
            nn.PReLU(),
        )

        self.convT2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128, out_channels=32, kernel_size=(1, 3), stride=(1, 1), padding=(0, 0)
            ),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )

        self.convT3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64, out_channels=32, kernel_size=(1, 3), stride=(1, 1), padding=(0, 0)
            ),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )

        self.convT4 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64, out_channels=32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 0)
            ),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )

        self.convT5 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64, out_channels=2, kernel_size=(1, 5), stride=(1, 2), padding=(0, 0)
            ),
        )

    def forward(self, inp, ht):
        """
        inp: B,T
        ht: B, 6
        """
        # spec: [B, 2, T, Fc]

        spec = self.stft.transform(inp)  # b,2,t,f
        ht = expand_HT(ht, spec.shape[-2], self.reso)

        cat_input = torch.cat([spec, ht], dim=1)
        conv_out1 = self.conv1(cat_input)
        conv_out2 = self.conv2(conv_out1)
        conv_out3 = self.conv3(conv_out2)
        conv_out4 = self.conv4(conv_out3)
        conv_out5 = self.conv5(conv_out4)

        DPRNN_out1 = self.DPRNN_1(conv_out5)
        DPRNN_out2 = self.DPRNN_2(DPRNN_out1)

        convT1_input = torch.cat((conv_out5, DPRNN_out2), 1)
        convT1_out = self.convT1(convT1_input)

        convT2_input = torch.cat((conv_out4, convT1_out[:, :, :, :-2]), 1)
        convT2_out = self.convT2(convT2_input)

        convT3_input = torch.cat((conv_out3, convT2_out[:, :, :, :-2]), 1)
        convT3_out = self.convT3(convT3_input)

        convT4_input = torch.cat((conv_out2, convT3_out[:, :, :, :-2]), 1)
        convT4_out = self.convT4(convT4_input)

        convT5_input = torch.cat((conv_out1, convT4_out[:, :, :, :-1]), 1)
        convT5_out = self.convT5(convT5_input)

        mask_out = convT5_out[:, :, :, :-2]

        mask_real = mask_out[:, 0, :, :]
        mask_imag = mask_out[:, 1, :, :]

        noisy_real = spec[:, 0, :, :]
        noisy_imag = spec[:, 1, :, :]

        ####### simple complex reconstruct

        # B,T,F
        enh_real = noisy_real * mask_real - noisy_imag * mask_imag
        enh_imag = noisy_real * mask_imag + noisy_imag * mask_real

        spec_out = torch.stack([enh_real, enh_imag], dim=1)
        out = self.stft.inverse(spec_out)
        return out


@tables.register("models", "FTCRN_LINEAR")
class MGAN_G_L(nn.Module):
    """
    Cheng, J., Liang, R., Zhao, L., Huang, C. and Schuller, B.W., 2023. Speech denoising and compensation for hearing aids using an FTCRN-based metric GAN. IEEE Signal Processing Letters, 30, pp.374-378.
    """

    def __init__(self, nframe=512, nhop=256):
        super().__init__()

        self.stft = STFT(nframe, nhop, nframe)
        self.nbin = nframe // 2 + 1
        self.reso = 16000 / nframe
        assert self.reso < 250 and 250 % self.reso == 0

        self.freqs = torch.linspace(0, 16000 // 2, self.nbin)  # []

        self.preprocess = HLModule(self.nbin, HL_freq_extend=self.freqs)

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=32, kernel_size=(1, 5), stride=(1, 2), padding=(0, 1)
            ),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)
            ),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)
            ),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)
            ),
            nn.BatchNorm2d(64),
            nn.PReLU(),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)
            ),
            nn.BatchNorm2d(128),
            nn.PReLU(),
        )

        self.DPRNN_1 = DPRNN_Block(numUnits=128, width=64)
        self.DPRNN_2 = DPRNN_Block(numUnits=128, width=64)

        self.convT1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256, out_channels=64, kernel_size=(1, 3), stride=(1, 1), padding=(0, 0)
            ),
            nn.BatchNorm2d(64),
            nn.PReLU(),
        )

        self.convT2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128, out_channels=32, kernel_size=(1, 3), stride=(1, 1), padding=(0, 0)
            ),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )

        self.convT3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64, out_channels=32, kernel_size=(1, 3), stride=(1, 1), padding=(0, 0)
            ),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )

        self.convT4 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64, out_channels=32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 0)
            ),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )

        self.convT5 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64, out_channels=2, kernel_size=(1, 5), stride=(1, 2), padding=(0, 0)
            ),
        )

    def forward(self, inp, ht):
        """
        inp: B,T
        ht: B, 6
        """
        # spec: [B, 2, T, Fc]

        spec = self.stft.transform(inp)  # b,2,t,f
        nT = spec.size(-2)
        # ht = expand_HT(ht, spec.shape[-2], self.reso)
        ht = self.preprocess.extend_with_linear(ht)
        ht = ht.repeat(1, 1, nT, 1)

        cat_input = torch.cat([spec, ht], dim=1)
        conv_out1 = self.conv1(cat_input)
        conv_out2 = self.conv2(conv_out1)
        conv_out3 = self.conv3(conv_out2)
        conv_out4 = self.conv4(conv_out3)
        conv_out5 = self.conv5(conv_out4)

        DPRNN_out1 = self.DPRNN_1(conv_out5)
        DPRNN_out2 = self.DPRNN_2(DPRNN_out1)

        convT1_input = torch.cat((conv_out5, DPRNN_out2), 1)
        convT1_out = self.convT1(convT1_input)

        convT2_input = torch.cat((conv_out4, convT1_out[:, :, :, :-2]), 1)
        convT2_out = self.convT2(convT2_input)

        convT3_input = torch.cat((conv_out3, convT2_out[:, :, :, :-2]), 1)
        convT3_out = self.convT3(convT3_input)

        convT4_input = torch.cat((conv_out2, convT3_out[:, :, :, :-2]), 1)
        convT4_out = self.convT4(convT4_input)

        convT5_input = torch.cat((conv_out1, convT4_out[:, :, :, :-1]), 1)
        convT5_out = self.convT5(convT5_input)

        mask_out = convT5_out[:, :, :, :-2]

        mask_real = mask_out[:, 0, :, :]
        mask_imag = mask_out[:, 1, :, :]

        noisy_real = spec[:, 0, :, :]
        noisy_imag = spec[:, 1, :, :]

        ####### simple complex reconstruct

        # B,T,F
        enh_real = noisy_real * mask_real - noisy_imag * mask_imag
        enh_imag = noisy_real * mask_imag + noisy_imag * mask_real

        spec_out = torch.stack([enh_real, enh_imag], dim=1)
        out = self.stft.inverse(spec_out)
        return out


@tables.register("models", "FTCRN_BASE_VAD")
class MGAN_G_BVAD(nn.Module):
    """
    Cheng, J., Liang, R., Zhao, L., Huang, C. and Schuller, B.W., 2023. Speech denoising and compensation for hearing aids using an FTCRN-based metric GAN. IEEE Signal Processing Letters, 30, pp.374-378.
    """

    def __init__(self, nframe=512, nhop=256):
        super().__init__()

        self.stft = STFT(nframe, nhop, nframe)
        self.nbin = nframe // 2 + 1
        self.reso = 16000 / nframe
        assert self.reso < 250 and 250 % self.reso == 0

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=32, kernel_size=(1, 5), stride=(1, 2), padding=(0, 1)
            ),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)
            ),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)
            ),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)
            ),
            nn.BatchNorm2d(64),
            nn.PReLU(),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)
            ),
            nn.BatchNorm2d(128),
            nn.PReLU(),
        )

        self.DPRNN_1 = DPRNN_Block(numUnits=128, width=64)
        self.DPRNN_2 = DPRNN_Block(numUnits=128, width=64)

        self.convT1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256, out_channels=64, kernel_size=(1, 3), stride=(1, 1), padding=(0, 0)
            ),
            nn.BatchNorm2d(64),
            nn.PReLU(),
        )

        self.convT2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128, out_channels=32, kernel_size=(1, 3), stride=(1, 1), padding=(0, 0)
            ),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )

        self.convT3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64, out_channels=32, kernel_size=(1, 3), stride=(1, 1), padding=(0, 0)
            ),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )

        self.convT4 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64, out_channels=32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 0)
            ),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )

        self.convT5 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64, out_channels=2, kernel_size=(1, 5), stride=(1, 2), padding=(0, 0)
            ),
        )

        self.vad_predictor = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1, 64), stride=(1, 64)),  # B,C,T,1
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=1,
                stride=1,
                padding=0,
            ),  # b,c,t,1
            Rearrange("b c t ()->b t c"),
            nn.LayerNorm(128),
            nn.PReLU(),
            nn.GRU(
                input_size=128,
                hidden_size=128,
                num_layers=2,
                batch_first=True,
            ),
        )

        self.vad_post = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, inp, ht):
        """
        inp: B,T
        ht: B, 6
        """
        # spec: [B, 2, T, Fc]

        spec = self.stft.transform(inp)  # b,2,t,f
        ht = expand_HT(ht, spec.shape[-2], self.reso)

        cat_input = torch.cat([spec, ht], dim=1)
        conv_out1 = self.conv1(cat_input)
        conv_out2 = self.conv2(conv_out1)
        conv_out3 = self.conv3(conv_out2)
        conv_out4 = self.conv4(conv_out3)
        conv_out5 = self.conv5(conv_out4)

        DPRNN_out1 = self.DPRNN_1(conv_out5)
        DPRNN_out2 = self.DPRNN_2(DPRNN_out1)

        vad_pred, _ = self.vad_predictor(DPRNN_out2)
        vad = self.vad_post(vad_pred)

        convT1_input = torch.cat((conv_out5, DPRNN_out2), 1)
        convT1_out = self.convT1(convT1_input)

        convT2_input = torch.cat((conv_out4, convT1_out[:, :, :, :-2]), 1)
        convT2_out = self.convT2(convT2_input)

        convT3_input = torch.cat((conv_out3, convT2_out[:, :, :, :-2]), 1)
        convT3_out = self.convT3(convT3_input)

        convT4_input = torch.cat((conv_out2, convT3_out[:, :, :, :-2]), 1)
        convT4_out = self.convT4(convT4_input)

        convT5_input = torch.cat((conv_out1, convT4_out[:, :, :, :-1]), 1)
        convT5_out = self.convT5(convT5_input)

        mask_out = convT5_out[:, :, :, :-2]

        mask_real = mask_out[:, 0, :, :]
        mask_imag = mask_out[:, 1, :, :]

        noisy_real = spec[:, 0, :, :]
        noisy_imag = spec[:, 1, :, :]

        ####### simple complex reconstruct

        # B,T,F
        enh_real = noisy_real * mask_real - noisy_imag * mask_imag
        enh_imag = noisy_real * mask_imag + noisy_imag * mask_real

        spec_out = torch.stack([enh_real, enh_imag], dim=1)
        out = self.stft.inverse(spec_out)
        return out, vad


@tables.register("models", "FTCRN_COND")
class MGAN_G_COND(nn.Module):
    """
    Cheng, J., Liang, R., Zhao, L., Huang, C. and Schuller, B.W., 2023. Speech denoising and compensation for hearing aids using an FTCRN-based metric GAN. IEEE Signal Processing Letters, 30, pp.374-378.
    """

    def __init__(self, nframe=512, nhop=256):
        super().__init__()

        self.stft = STFT(nframe, nhop, nframe)
        self.nbin = nframe // 2 + 1
        self.reso = 16000 / nframe
        assert self.reso < 250 and 250 % self.reso == 0

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=2, out_channels=32, kernel_size=(1, 5), stride=(1, 2), padding=(0, 1)
            ),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)
            ),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)
            ),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)
            ),
            nn.BatchNorm2d(64),
            nn.PReLU(),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)
            ),
            nn.BatchNorm2d(128),
            nn.PReLU(),
        )

        self.DPRNN_1 = DPRNN_Block_Conditional(numUnits=128, width=64)
        self.DPRNN_2 = DPRNN_Block_Conditional(numUnits=128, width=64)

        self.convT1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256, out_channels=64, kernel_size=(1, 3), stride=(1, 1), padding=(0, 0)
            ),
            nn.BatchNorm2d(64),
            nn.PReLU(),
        )

        self.convT2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128, out_channels=32, kernel_size=(1, 3), stride=(1, 1), padding=(0, 0)
            ),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )

        self.convT3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64, out_channels=32, kernel_size=(1, 3), stride=(1, 1), padding=(0, 0)
            ),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )

        self.convT4 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64, out_channels=32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 0)
            ),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )

        self.convT5 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64, out_channels=2, kernel_size=(1, 5), stride=(1, 2), padding=(0, 0)
            ),
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.nbin, self.nbin * 4),
            Rearrange("b c t (f n)-> b (c n) t f", n=4),
            nn.GELU(approximate="tanh"),
            nn.Conv2d(4, 16, (1, 5), (1, 2), (0, 1)),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.Conv2d(16, 64, (1, 3), (1, 2), (0, 1)),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            Rearrange("b c t f-> b f t c"),
            nn.Linear(64, 128),
        )

    def forward(self, inp, ht):
        """
        inp: B,T
        ht: B, 6
        """
        # spec: [B, 2, T, Fc]

        spec = self.stft.transform(inp)  # b,2,t,f
        ht = expand_HT_norepeat(ht, spec.shape[-2], self.reso)

        hl = self.mlp(ht)  # b,1,1,f
        hl = hl.squeeze(2)

        # cat_input = torch.cat([spec, ht], dim=1)
        conv_out1 = self.conv1(spec)
        conv_out2 = self.conv2(conv_out1)
        conv_out3 = self.conv3(conv_out2)
        conv_out4 = self.conv4(conv_out3)
        conv_out5 = self.conv5(conv_out4)

        DPRNN_out1 = self.DPRNN_1(conv_out5, hl)
        DPRNN_out2 = self.DPRNN_2(DPRNN_out1, hl)

        convT1_input = torch.cat((conv_out5, DPRNN_out2), 1)
        convT1_out = self.convT1(convT1_input)

        convT2_input = torch.cat((conv_out4, convT1_out[:, :, :, :-2]), 1)
        convT2_out = self.convT2(convT2_input)

        convT3_input = torch.cat((conv_out3, convT2_out[:, :, :, :-2]), 1)
        convT3_out = self.convT3(convT3_input)

        convT4_input = torch.cat((conv_out2, convT3_out[:, :, :, :-2]), 1)
        convT4_out = self.convT4(convT4_input)

        convT5_input = torch.cat((conv_out1, convT4_out[:, :, :, :-1]), 1)
        convT5_out = self.convT5(convT5_input)

        mask_out = convT5_out[:, :, :, :-2]

        mask_real = mask_out[:, 0, :, :]
        mask_imag = mask_out[:, 1, :, :]

        noisy_real = spec[:, 0, :, :]
        noisy_imag = spec[:, 1, :, :]

        ####### simple complex reconstruct

        # B,T,F
        enh_real = noisy_real * mask_real - noisy_imag * mask_imag
        enh_imag = noisy_real * mask_imag + noisy_imag * mask_real

        spec_out = torch.stack([enh_real, enh_imag], dim=1)
        out = self.stft.inverse(spec_out)
        return out


@tables.register("models", "FTCRN_VAD")
class MGAN_G_VAD(nn.Module):
    def __init__(self, nframe=512, nhop=256):
        super().__init__()

        self.stft = STFT(nframe, nhop, nframe)
        self.nbin = nframe // 2 + 1
        self.reso = 16000 / nframe
        assert self.reso < 250 and 250 % self.reso == 0

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=2, out_channels=32, kernel_size=(1, 5), stride=(1, 2), padding=(0, 1)
            ),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)
            ),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)
            ),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)
            ),
            nn.BatchNorm2d(64),
            nn.PReLU(),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)
            ),
            nn.BatchNorm2d(128),
            nn.PReLU(),
        )

        self.DPRNN_1 = DPRNN_Block_Conditional(numUnits=128, width=64)
        self.DPRNN_2 = DPRNN_Block_Conditional(numUnits=128, width=64)

        self.convT1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256, out_channels=64, kernel_size=(1, 3), stride=(1, 1), padding=(0, 0)
            ),
            nn.BatchNorm2d(64),
            nn.PReLU(),
        )

        self.convT2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128, out_channels=32, kernel_size=(1, 3), stride=(1, 1), padding=(0, 0)
            ),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )

        self.convT3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64, out_channels=32, kernel_size=(1, 3), stride=(1, 1), padding=(0, 0)
            ),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )

        self.convT4 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64, out_channels=32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 0)
            ),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )

        self.convT5 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64, out_channels=2, kernel_size=(1, 5), stride=(1, 2), padding=(0, 0)
            ),
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.nbin, self.nbin * 4),
            Rearrange("b c t (f n)-> b (c n) t f", n=4),
            nn.GELU(approximate="tanh"),
            nn.Conv2d(4, 16, (1, 5), (1, 2), (0, 1)),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.Conv2d(16, 64, (1, 3), (1, 2), (0, 1)),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            Rearrange("b c t f-> b f t c"),
            nn.Linear(64, 128),
        )

        self.vad_predictor = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1, 64), stride=(1, 64)),  # B,C,T,1
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=1,
                stride=1,
                padding=0,
            ),  # b,c,t,1
            Rearrange("b c t ()->b t c"),
            nn.LayerNorm(128),
            nn.PReLU(),
            nn.GRU(
                input_size=128,
                hidden_size=128,
                num_layers=2,
                batch_first=True,
            ),
        )

        self.vad_post = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, inp, ht):
        """
        inp: B,T
        ht: B, 6
        """
        # spec: [B, 2, T, Fc]

        spec = self.stft.transform(inp)  # b,2,t,f
        ht = expand_HT_norepeat(ht, spec.shape[-2], self.reso)

        hl = self.mlp(ht)  # b,1,1,f
        hl = hl.squeeze(2)

        # cat_input = torch.cat([spec, ht], dim=1)
        conv_out1 = self.conv1(spec)
        conv_out2 = self.conv2(conv_out1)
        conv_out3 = self.conv3(conv_out2)
        conv_out4 = self.conv4(conv_out3)
        conv_out5 = self.conv5(conv_out4)

        DPRNN_out1 = self.DPRNN_1(conv_out5, hl)
        DPRNN_out2 = self.DPRNN_2(DPRNN_out1, hl)

        vad_pred, _ = self.vad_predictor(DPRNN_out2)
        vad = self.vad_post(vad_pred)

        convT1_input = torch.cat((conv_out5, DPRNN_out2), 1)
        convT1_out = self.convT1(convT1_input)

        convT2_input = torch.cat((conv_out4, convT1_out[:, :, :, :-2]), 1)
        convT2_out = self.convT2(convT2_input)

        convT3_input = torch.cat((conv_out3, convT2_out[:, :, :, :-2]), 1)
        convT3_out = self.convT3(convT3_input)

        convT4_input = torch.cat((conv_out2, convT3_out[:, :, :, :-2]), 1)
        convT4_out = self.convT4(convT4_input)

        convT5_input = torch.cat((conv_out1, convT4_out[:, :, :, :-1]), 1)
        convT5_out = self.convT5(convT5_input)

        mask_out = convT5_out[:, :, :, :-2]

        mask_real = mask_out[:, 0, :, :]
        mask_imag = mask_out[:, 1, :, :]

        noisy_real = spec[:, 0, :, :]
        noisy_imag = spec[:, 1, :, :]

        ####### simple complex reconstruct

        # B,T,F
        enh_real = noisy_real * mask_real - noisy_imag * mask_imag
        enh_imag = noisy_real * mask_imag + noisy_imag * mask_real

        spec_out = torch.stack([enh_real, enh_imag], dim=1)
        out = self.stft.inverse(spec_out)
        return out, vad


if __name__ == "__main__":
    inputs = torch.randn(2, 16000)
    # inputs_ht = torch.randn(1, 1, 100, 257)
    inputs_ht = torch.tensor([[2, 4, 8, 16, 32, 64], [1, 2, 3, 4, 5, 6]])

    # net = MGAN_G(512, 256)
    net = MGAN_G_VAD(512, 256)
    # net = MGAN_G_COND(512, 256)

    # out = net.expand_ht(inputs_ht, 10)
    out, _ = net(inputs, inputs_ht)
    print(out.shape)

    net = Discriminator(16)
    out = net(inputs, inputs, inputs_ht)
    print(out.shape)

    # check_flops(Model, inputs, inputs_ht)

    # params_of_network = 0
    # for param in Model.parameters():
    #     params_of_network += param.numel()

    # print(f"\tNetwork: {params_of_network / 1e6} million.")

    # est_real, est_imag = Model(inputs, inputs_ht)

    # print(est_real.shape)
