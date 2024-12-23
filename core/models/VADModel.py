from numpy import repeat
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from torch import Tensor, nn
from utils.register import tables


def vad_to_frames(vad: Tensor, nframe: int, nhop: int):
    """
    vad: B,T,1 or B,T

    return: B,T(#Frame),1
    """
    npad = int(nframe // 2)
    vad = vad.squeeze(-1)

    vad = F.pad(vad, (npad, npad), value=0.0)
    N = vad.shape[-1]
    idx = torch.arange(nframe).reshape(1, -1)
    step = torch.arange(0, N - nframe + 1, nhop).reshape(-1, 1)
    idx = step + idx

    frames = vad[:, idx]  # B,T(#frame),D
    frames = torch.mean(frames, dim=-1, keepdim=True)  # B,T,1
    ones_vec = torch.ones_like(frames)
    zeros_vec = torch.zeros_like(frames)

    vad_label = torch.where(frames >= 0.5, ones_vec, zeros_vec)

    return vad_label


def pack_frames_vad(vad: Tensor, nframe: int, nhop: int) -> torch.Tensor:
    """
    vad: B,#f,1

    return: B,T
    """
    assert nframe == 2 * nhop

    nF = vad.shape[1]
    npad = int(nframe // 2)
    vad = vad.repeat(1, 1, nframe)  # B,T,D
    vad = vad.permute(0, 2, 1)  # B,D,T(#f)
    nlen = nF * nhop + nframe - nhop
    out = F.fold(vad, output_size=(nlen, 1), kernel_size=(nframe, 1), stride=(nhop, 1))
    out = out.squeeze()[..., npad:-npad] / 2
    return out


@tables.register("models", "crnn_vad")
class CRNN_VAD_new(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self, feat_size):
        super(CRNN_VAD_new, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=2,
                out_channels=4,
                kernel_size=(1, 3),
                stride=(1, 2),
                padding=(0, 1),
            ),
            nn.BatchNorm2d(4),
            nn.PReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=4,
                out_channels=8,
                kernel_size=(1, 3),
                stride=(1, 2),
                padding=(0, 1),
            ),
            nn.BatchNorm2d(8),
            nn.PReLU(),
        )
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=4,
        #         out_channels=8,
        #         kernel_size=(1, 3),
        #         stride=(1, 2),
        #         padding=(0, 1),
        #     ),
        #     nn.BatchNorm2d(8),
        #     nn.LeakyReLU(negative_slope=0.3),
        # )
        # self.conv4 = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=8,
        #         out_channels=8,
        #         kernel_size=(1, 3),
        #         stride=(1, 2),
        #         padding=(0, 1),
        #     ),
        #     nn.BatchNorm2d(8),
        #     nn.LeakyReLU(negative_slope=0.3),
        # )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1, 3), stride=(1, 2)),
            nn.BatchNorm2d(16),
            nn.PReLU(),
        )

        # feat//8 * 16
        self.GRU = nn.GRU(
            input_size=feat_size // 8 * 16,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
        )

        self.output_dense = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())

    def forward(self, x):
        # conv
        # (B, in_c, T, F)

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        # x4 = self.conv4(x3)
        # x5 = self.conv5(x4)

        mid_in = x3

        mid_GRU_in = mid_in.permute(0, 2, 1, 3)  # bctf->btcf
        mid_GRU_in = mid_GRU_in.reshape(mid_GRU_in.size()[0], mid_GRU_in.size()[1], -1)

        gru_out, _ = self.GRU(mid_GRU_in)

        sig_out = self.output_dense(gru_out)

        return sig_out


@tables.register("models", "crnn_vad_origin")
class CRNN_VAD_new_origin(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self, feat_size):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=2,
                out_channels=4,
                kernel_size=(1, 3),
                stride=(1, 2),
                padding=(0, 1),
            ),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(negative_slope=0.3),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=4,
                out_channels=4,
                kernel_size=(1, 3),
                stride=(1, 2),
                padding=(0, 1),
            ),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(negative_slope=0.3),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=4,
                out_channels=8,
                kernel_size=(1, 3),
                stride=(1, 2),
                padding=(0, 1),
            ),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(negative_slope=0.3),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=8,
                out_channels=8,
                kernel_size=(1, 3),
                stride=(1, 2),
                padding=(0, 1),
            ),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(negative_slope=0.3),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1, 3), stride=(1, 2)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope=0.3),
        )

        # feat//32 * 16
        self.GRU = nn.GRU(
            input_size=feat_size // 2,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
        )

        self.output_dense = nn.Linear(128, 1)

    def forward(self, x):
        # conv
        # (B, in_c, T, F)

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        mid_in = x5

        mid_GRU_in = mid_in.permute(0, 2, 1, 3)  # bctf->btcf
        mid_GRU_in = mid_GRU_in.reshape(mid_GRU_in.size()[0], mid_GRU_in.size()[1], -1)

        gru_out, _ = self.GRU(mid_GRU_in)

        sig_out = torch.sigmoid(self.output_dense(gru_out))

        return sig_out


class CRNN_VAD_new_Lite(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self):
        super(CRNN_VAD_new_Lite, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=2,
                out_channels=4,
                kernel_size=(1, 3),
                stride=(1, 2),
                padding=(0, 1),
            ),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(negative_slope=0.3),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=4,
                out_channels=4,
                kernel_size=(1, 3),
                stride=(1, 2),
                padding=(0, 1),
            ),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(negative_slope=0.3),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=4,
                out_channels=8,
                kernel_size=(1, 3),
                stride=(1, 2),
                padding=(0, 1),
            ),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(negative_slope=0.3),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=8,
                out_channels=8,
                kernel_size=(1, 3),
                stride=(1, 2),
                padding=(0, 1),
            ),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(negative_slope=0.3),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(1, 3), stride=(1, 2)),
            nn.BatchNorm2d(12),
            nn.LeakyReLU(negative_slope=0.3),
        )

        self.GRU = nn.GRU(input_size=96, hidden_size=96, num_layers=1, batch_first=True)

        self.output_dense = nn.Linear(96, 1)

    def forward(self, x):
        # conv
        # (B, in_c, T, F)

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        mid_in = x5

        mid_GRU_in = mid_in.permute(0, 2, 1, 3)
        mid_GRU_in = mid_GRU_in.reshape(mid_GRU_in.size()[0], mid_GRU_in.size()[1], -1)

        gru_out, _ = self.GRU(mid_GRU_in)

        sig_out = torch.sigmoid(self.output_dense(gru_out))

        return sig_out


if __name__ == "__main__":
    from thop import profile
    import soundfile as sf
    import warnings
    import numpy as np

    inputs = torch.randn(1, 2, 250, 257)  # B,C,T,F

    net = CRNN_VAD_new(257)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="This API is being deprecated")
        flops, params = profile(
            net,
            inputs=(inputs,),
            verbose=False,
        )
    print(flops / 1e9, params / 1e6)

    data, fs = sf.read("/home/deepni/trunk/infant/train/3.wav")
    vad = data[:, 1]
    vad = vad[None, :, None]
    print(vad.shape)
    ret = vad_to_frames(torch.from_numpy(vad), 512, 256)
    out = pack_frames_vad(ret, 512, 256)
    out = out.cpu().numpy()  # T,
    N = out.shape[-1]
    print(out.shape)
    sf.write("test.wav", np.stack([data[:N, 0], out], axis=-1), fs)
    # print(torch.allclose(vad, out))
    # print(vad)
    # print(out)

    # vad_out = net(inputs)

    # print(vad_out.shape)
    #
    #
