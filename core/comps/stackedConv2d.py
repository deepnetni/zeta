from itertools import repeat
from typing import List, Tuple, Union

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange


class StackedConv2d(nn.Module):
    def __init__(
        self,
        channels: List,
        kernels: Union[List, Tuple],
        strides: Union[List, Tuple],
        norm: nn.Module = nn.BatchNorm2d,
    ) -> None:
        super().__init__()

        if isinstance(kernels, Tuple):
            kernels = repeat(kernels, times=len(channels) - 1)
        else:
            assert (
                len(kernels) >= len(channels) - 1
            ), f"kernel configure is not enough, required {len(channels)-1}, given {len(kernels)}"

        if isinstance(strides, Tuple):
            strides = repeat(strides, times=len(channels) - 1)
        else:
            assert (
                len(strides) >= len(channels) - 1
            ), f"strides configure is not enough, required {len(channels)-1}, given {len(kernels)}"

        self.layer = nn.ModuleList()
        for ci, co, kernel, stride in zip(channels[:-1], channels[1:], kernels, strides):
            nt, nf = kernel
            padf = (nf - 1) // 2
            padt = nt - 1
            self.layer.append(
                nn.Sequential(
                    nn.ConstantPad2d((padf, padf, padt, 0), value=0.0),
                    nn.Conv2d(ci, co, kernel, stride),
                    norm(co),
                    nn.PReLU(co),
                )
            )

    def forward(self, x):
        """
        x: b,c,t,f
        """
        state = []
        for l in self.layer:
            x = l(x)
            state.append(x)

        return x, state


class StackedTransposedConv2d(nn.Module):
    def __init__(
        self,
        channels: List,
        kernels: Union[List, Tuple],
        strides: Union[List, Tuple],
        norm: nn.Module = nn.BatchNorm2d,
        skip: bool = True,
    ) -> None:
        super().__init__()
        self.skip = skip

        if isinstance(kernels, Tuple):
            kernels = repeat(kernels, times=len(channels) - 1)
        else:
            assert (
                len(kernels) >= len(channels) - 1
            ), f"kernel configure is not enough, required {len(channels)-1}, given {len(kernels)}"

        if isinstance(strides, Tuple):
            strides = repeat(strides, times=len(channels) - 1)
        else:
            assert (
                len(strides) >= len(channels) - 1
            ), f"strides configure is not enough, required {len(channels)-1}, given {len(kernels)}"

        self.layer = nn.ModuleList()
        self.padt = []
        for ci, co, kernel, stride in zip(channels[:-1], channels[1:], kernels, strides):
            ci = ci * 2 if self.skip else ci
            nt, nf = kernel
            padf = (nf - 1) // 2
            self.padt.append(nt - 1)
            self.layer.append(
                nn.Sequential(
                    # nn.ConstantPad2d((padf, padf, padt, 0), value=0.0),
                    nn.ConvTranspose2d(ci, co, kernel, stride, padding=(0, padf)),
                    norm(co),
                    nn.PReLU(co),
                )
            )

    def forward(self, x, y=None):
        """
        x: b,c,t,f
        """
        for i, l in enumerate(self.layer):
            if self.skip:
                assert y is not None
                x = torch.concat([x, y[i]], dim=1)

            x = l(x)
            x = x[..., self.padt[i] :, :]

        return x


if __name__ == "__main__":
    inp = torch.randn(1, 2, 10, 33)
    net = StackedConv2d(
        [2, 3, 4],
        [(2, 3), (2, 5)],
        (1, 2),
    )
    out, sta = net(inp)
    print(out.shape)

    net_ = StackedTransposedConv2d(
        [2, 3, 4][::-1],
        [(2, 3), (2, 5)][::-1],
        (1, 2),
    )

    out = net_(out, sta[::-1])
    print(out.shape)
