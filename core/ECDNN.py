import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

from models.conv_stft import STFT
from utils.check_flops import check_flops
from utils.register import tables


class AGModule(nn.Module):
    def __init__(self, channel: int) -> None:
        super().__init__()

        self.conv_l = nn.Conv2d(2 * channel, channel, (1, 1), (1, 1))
        self.conv_k = nn.Conv2d(channel, channel, (1, 1), (1, 1))
        self.conv_r = nn.Sequential(
            nn.ReLU(), nn.Conv2d(channel, channel, (1, 1), (1, 1)), nn.Sigmoid()
        )

    def forward(self, enc, dec):
        print(enc.shape, dec.shape, "@")
        w = self.conv_k(self.conv_l(enc) + self.conv_k(dec))
        return dec * w


@tables.register("models", "ECDNN_FIG6")
class ECDNN_FIG6(nn.Module):
    def __init__(self, nframe=512, nhop=256, channels=[8, 16, 32, 64, 128]) -> None:
        super().__init__()
        self.stft = STFT(nframe, nhop)

        in_channels = [1] + channels

        self.encoder = nn.ModuleList()
        self.ag_r = nn.ModuleList()
        self.ag_i = nn.ModuleList()
        # self.decoder = nn.ModuleList()

        lr, li = [], []
        ag_r, ag_i = [], []
        nlayer = len(channels) - 1

        for i, (ci, co) in enumerate(zip(in_channels[:-1], in_channels[1:])):
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(ci, co, (1, 3), (1, 2), (0, 1)),
                    nn.BatchNorm2d(co),
                    nn.ELU(),
                )
            )
            ag_r.append(AGModule(ci))
            ag_i.append(AGModule(ci))

            if i == nlayer:
                lr.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(co, ci, (1, 3), (1, 2), (0, 1)),
                        nn.BatchNorm2d(ci),
                        nn.ELU(),
                    )
                )
                li.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(co, ci, (1, 3), (1, 2), (0, 1)),
                        nn.BatchNorm2d(ci),
                        nn.ELU(),
                    )
                )
            elif i == 0:
                lr.append(
                    nn.ConvTranspose2d(co, ci, (1, 3), (1, 2), (0, 1)),
                )
                li.append(
                    nn.ConvTranspose2d(co, ci, (1, 3), (1, 2), (0, 1)),
                )
            else:
                lr.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(co * 2, ci, (1, 3), (1, 2), (0, 1)),
                        nn.BatchNorm2d(ci),
                        nn.ELU(),
                    )
                )
                li.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(co * 2, ci, (1, 3), (1, 2), (0, 1)),
                        nn.BatchNorm2d(ci),
                        nn.ELU(),
                    )
                )

        self.decoder_r = nn.ModuleList(lr[::-1])
        self.decoder_r = nn.ModuleList(lr[::-1])
        self.ag_r = nn.ModuleList(ag_r[::-1])
        self.ag_i = nn.ModuleList(ag_i[::-1])

    def forward(self, inp, HL):
        x = self.stft.transform(inp)
        mag = x.pow(2).sum(1, keepdim=True).sqrt()

        x = mag
        x_enc = []
        for l in self.encoder:
            x = l(x)
            x_enc.append(x)

        x_enc = x_enc[::-1]
        x_r = x
        for i, l in enumerate(self.decoder_r):
            x_r = l(x_r)
            print(x_r.shape, x_enc[i].shape)
            x_ = self.ag_r[i](x_enc[i], x_r)
            x_r = torch.concat([x_r, x_], dim=1)

        # x_i = x
        # for l in self.decoder_i:
        #     x_i = l(x_i)

        # xk = torch.concat([x_r, x_i], dim=1)
        # out = self.stft.inverse(xk)
        return out


if __name__ == "__main__":
    inp = torch.rand(1, 16000)
    HL = torch.rand(1, 6)
    net = ECDNN_FIG6(512, 256)

    check_flops(net, inp, HL)
