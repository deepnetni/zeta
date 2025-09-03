import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import get_window

from einops import rearrange


class OverlapAdd(nn.Module):
    def __init__(self, nblk) -> None:
        """Frame2Frame Time <--> Frequency convertion."""
        super().__init__()

        self.register_buffer("win", torch.tensor(get_window("hann", 2 * nblk)).sqrt().float())
        self.nblk = nblk

    def reset_buff(self, inp):
        nB = inp.size(0)

        self.buff_frame = torch.zeros(nB, 2 * self.nblk, dtype=inp.dtype).to(inp.device)
        self.buff_ola = torch.zeros(nB, self.nblk, dtype=inp.dtype).to(inp.device)

    def inverse(self, xk):
        """
        xk: B,2,1,F
        """
        assert xk.shape[2] == 1

        xk = rearrange(xk, "b c 1 f -> b f c").contiguous()
        x = torch.fft.irfft(torch.view_as_complex(xk), n=self.nblk * 2, dim=-1)
        x = x * self.win

        out = self.buff_ola + x[..., : self.nblk]

        self.buff_ola = x[..., -self.nblk :]

        return out

    def transform(self, inp):
        """
        inp: B,nblk
        return: B,2,1,F
        """
        self.buff_frame[:, : self.nblk] = self.buff_frame[:, self.nblk :]
        self.buff_frame[:, self.nblk :] = inp

        # B,F,2
        xk = torch.fft.rfft(self.buff_frame * self.win, n=self.nblk * 2, dim=-1)
        # B,2,1,F
        xk = torch.view_as_real(xk).permute(0, 2, 1).unsqueeze(2)

        return xk


def check():
    inp = torch.randn(1, 64 * 3)
    inp_p = F.pad(inp, (0, 64))
    net = OverlapAdd(64)
    net.reset_buff(inp)

    l = []
    for i in range(4):
        st = i * 64
        ed = st + 64
        out = net.transform(inp_p[:, st:ed])
        out = net.inverse(out)
        l.append(out)

    inp_ = torch.concat(l[1:], dim=-1)
    diff = (inp[:, :] - inp_[:, :]).abs().sum()
    print(diff)


if __name__ == "__main__":
    check()
