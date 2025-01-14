import torch
from torch import Tensor, nn

from core.models.conv_stft import STFT
from core.utils.check_flops import check_flops
from core.utils.register import tables


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


if __name__ == "__main__":
    inputs = torch.randn(2, 16000)
    # inputs_ht = torch.randn(1, 1, 100, 257)
    inputs_ht = torch.tensor([[2, 4, 8, 16, 32, 64], [1, 2, 3, 4, 5, 6]])

    net = MGAN_G(512, 256)

    # out = net.expand_ht(inputs_ht, 10)
    out = net(inputs, inputs_ht)

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
