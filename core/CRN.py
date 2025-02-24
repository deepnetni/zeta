import torch
import torch.nn as nn
import torch.nn.functional as F

from JointNSHModel import expand_HT
from models.conv_stft import STFT
from utils.check_flops import check_flops
from utils.register import tables


class CausalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 2),
            stride=(2, 1),
            padding=(0, 1),
        )
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        self.activation = nn.ELU()

    def forward(self, x):
        """
        2D Causal convolution.
        Args:
            x: [B, C, F, T]

        Returns:
            [B, C, F, T]
        """
        x = self.conv(x)
        x = x[:, :, :, :-1]  # chomp size
        x = self.norm(x)
        x = self.activation(x)
        return x


class CausalTransConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_last=False, output_padding=(0, 0)):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 2),
            stride=(2, 1),
            output_padding=output_padding,
        )
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        if is_last:
            self.activation = nn.ReLU()
        else:
            self.activation = nn.ELU()

    def forward(self, x):
        """
        2D Causal convolution.
        Args:
            x: [B, C, F, T]

        Returns:
            [B, C, F, T]
        """
        x = self.conv(x)
        x = x[:, :, :, :-1]  # chomp size
        x = self.norm(x)
        x = self.activation(x)
        return x


class CRN(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self):
        super(CRN, self).__init__()
        # Encoder
        self.conv_block_1 = CausalConvBlock(1, 16)
        self.conv_block_2 = CausalConvBlock(16, 32)
        self.conv_block_3 = CausalConvBlock(32, 64)
        self.conv_block_4 = CausalConvBlock(64, 128)
        self.conv_block_5 = CausalConvBlock(128, 256)

        # LSTM
        self.lstm_layer = nn.LSTM(input_size=1024, hidden_size=1024, num_layers=2, batch_first=True)

        self.tran_conv_block_1 = CausalTransConvBlock(256 + 256, 128)
        self.tran_conv_block_2 = CausalTransConvBlock(128 + 128, 64)
        self.tran_conv_block_3 = CausalTransConvBlock(64 + 64, 32)
        self.tran_conv_block_4 = CausalTransConvBlock(32 + 32, 16, output_padding=(1, 0))
        self.tran_conv_block_5 = CausalTransConvBlock(16 + 16, 1, is_last=True)

    def forward(self, x):
        self.lstm_layer.flatten_parameters()

        e_1 = self.conv_block_1(x)
        e_2 = self.conv_block_2(e_1)
        e_3 = self.conv_block_3(e_2)
        e_4 = self.conv_block_4(e_3)
        e_5 = self.conv_block_5(e_4)  # [2, 256, 4, 200]

        batch_size, n_channels, n_f_bins, n_frame_size = e_5.shape

        # [2, 256, 4, 200] = [2, 1024, 200] => [2, 200, 1024]
        lstm_in = e_5.reshape(batch_size, n_channels * n_f_bins, n_frame_size).permute(0, 2, 1)
        lstm_out, _ = self.lstm_layer(lstm_in)  # [2, 200, 1024]
        lstm_out = lstm_out.permute(0, 2, 1).reshape(
            batch_size, n_channels, n_f_bins, n_frame_size
        )  # [2, 256, 4, 200]

        d_1 = self.tran_conv_block_1(torch.cat((lstm_out, e_5), 1))
        d_2 = self.tran_conv_block_2(torch.cat((d_1, e_4), 1))
        d_3 = self.tran_conv_block_3(torch.cat((d_2, e_3), 1))
        d_4 = self.tran_conv_block_4(torch.cat((d_3, e_2), 1))
        d_5 = self.tran_conv_block_5(torch.cat((d_4, e_1), 1))

        return d_5


@tables.register("models", "CRN_FIG6")
class CRN_FIG6(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self):
        super(CRN_FIG6, self).__init__()
        self.stft = STFT(512, 256, 512)
        # Encoder
        self.conv_block_1 = CausalConvBlock(2, 8)
        self.conv_block_2 = CausalConvBlock(8, 16)
        self.conv_block_3 = CausalConvBlock(16, 32)
        self.conv_block_4 = CausalConvBlock(32, 64)
        self.conv_block_5 = CausalConvBlock(64, 128)

        # LSTM
        self.lstm_layer = nn.LSTM(input_size=896, hidden_size=896, num_layers=2, batch_first=True)

        self.tran_conv_block_1 = CausalTransConvBlock(128 * 2, 64)
        self.tran_conv_block_2 = CausalTransConvBlock(64 * 2, 32)
        self.tran_conv_block_3 = CausalTransConvBlock(32 * 2, 16)
        self.tran_conv_block_4 = CausalTransConvBlock(16 * 2, 8, output_padding=(1, 0))
        self.tran_conv_block_5 = CausalTransConvBlock(8 * 2, 1, is_last=True)

        self.reso = 16000 / 512

    def forward(self, inputs, HL):
        self.lstm_layer.flatten_parameters()
        specs = self.stft.transform(inputs)  # b,c,t,f
        mag = specs.pow(2).sum(1, keepdim=True).sqrt()
        phase = torch.atan2(specs[:, 1, ...], specs[:, 0, ...]).unsqueeze(1)

        hl = expand_HT(HL, specs.shape[-2], self.reso)  # B,C(1),T,F
        x = torch.concat([mag, hl], dim=1)  # b,c,t,f
        x = x.permute(0, 1, 3, 2)
        e_1 = self.conv_block_1(x)
        e_2 = self.conv_block_2(e_1)
        e_3 = self.conv_block_3(e_2)
        e_4 = self.conv_block_4(e_3)
        e_5 = self.conv_block_5(e_4)  # [2, 256, 4, 200]

        batch_size, n_channels, n_f_bins, n_frame_size = e_5.shape

        # [2, 256, 4, 200] = [2, 1024, 200] => [2, 200, 1024]
        lstm_in = e_5.reshape(batch_size, n_channels * n_f_bins, n_frame_size).permute(0, 2, 1)
        lstm_out, _ = self.lstm_layer(lstm_in)  # [2, 200, 1024]
        lstm_out = lstm_out.permute(0, 2, 1).reshape(
            batch_size, n_channels, n_f_bins, n_frame_size
        )  # [2, 256, 4, 200]

        d_1 = self.tran_conv_block_1(torch.cat((lstm_out, e_5), 1))
        d_2 = self.tran_conv_block_2(torch.cat((d_1, e_4), 1))
        d_3 = self.tran_conv_block_3(torch.cat((d_2, e_3), 1))
        d_4 = self.tran_conv_block_4(torch.cat((d_3, e_2), 1))
        d_5 = self.tran_conv_block_5(torch.cat((d_4, e_1), 1))

        mag_ = d_5.permute(0, 1, 3, 2)
        r, i = mag_ * torch.cos(phase), mag_ * torch.sin(phase)

        spec = torch.concat([r, i], dim=1)

        wave = self.stft.inverse(spec)

        return wave

    def loss(self, target, enhanced):
        l1 = F.mse_loss(enhanced, target, reduction="mean")
        return dict(loss=l1)


if __name__ == "__main__":
    net = CRN_FIG6()
    # inp = torch.rand(2, 1, 161, 200)
    inp = torch.rand(1, 16000)
    hl = torch.rand(1, 6)

    check_flops(net, inp, hl)
