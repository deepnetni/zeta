import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from typing import Optional


class AcousticFeedbackSim(nn.Module):
    def __init__(self, rir, nblk) -> None:
        """perform convolution using block-wise frequency domain.
        e.g.,
        out = scipy.signal.convolve(inp, rir, mode="full")
        out = out[:nblk]

        :param rir: RIR
        :param nblk: split RIR to blocks with `nblk` size.
        :returns:

        """
        super().__init__()
        assert rir.shape[-1] % nblk == 0

        self.N = nblk

        # K, B, N+1
        self.register_buffer("rir_fft_blk_weight", self._compute_rir_fft_blk(rir))

        self.init_cache = False

    def compute_MSG(self, margin_dB=2):
        """marginally stable gain"""
        xk_rir = self.rir_fft_blk_weight  # K,B,N+1
        # K,B,N+1 -> B,K,N+1
        rir_mag = torch.abs(xk_rir).permute(1, 0, 2)
        rir_pha = torch.angle(xk_rir).permute(1, 0, 2)

        # phase ~ 0 window
        mask = (rir_pha > -0.1) & (rir_pha < 0.1)
        pha_zero_mag = torch.where(mask, rir_mag, torch.zeros_like(rir_mag))

        # (B,) peak over freq & block
        peak_gain = pha_zero_mag.amax(dim=(1, 2))

        MSG_dB = 20 * torch.log10(torch.ones_like(peak_gain) / (peak_gain + 1e-7)) - margin_dB

        MSG = 10 ** (MSG_dB / 20)

        return MSG

    def reset_cache(self, inp: torch.Tensor, rir: Optional[torch.Tensor] = None):
        """
        inp: only uses its batch num count.
        """
        nB = inp.size(0)
        # K,nB,N+1
        K = self.rir_fft_blk_weight.shape[0]
        self.buff_inp_blk_fft = torch.zeros(K, nB, self.N + 1, dtype=torch.complex64).to(inp.device)
        # B,N
        self.buff_frames = torch.zeros(nB, self.N, dtype=torch.float32).to(inp.device)
        self.init_cache = True

        if rir is not None:
            self.update_rir(rir)

    def apply_full(self, inp):
        """end2end operation.
        inp: B,T
        """
        L = inp.shape[-1]

        nf = L // self.N

        out = []
        for i in range(nf):
            st = i * self.N
            d = inp[:, st : st + self.N]
            d_ = self.forward(d)
            out.append(d_)

        return torch.concat(out, dim=-1).float()

    def update_rir(self, rir):
        # self.rir_fft_blk_weight = self._compute_rir_fft_blk(rir)
        with torch.no_grad():
            new = self._compute_rir_fft_blk(rir).to(self.rir_fft_blk_weight.device)
            self.rir_fft_blk_weight.copy_(new)

    def _compute_rir_fft_blk(self, rir: torch.Tensor):
        """
        rir: (B,2N) or (2N,)
        return: (K, 1, N+1) complex if rir is (1, 2N)
        return: (K, B, N+1) complex if rir is (B, 2N)
        """
        if rir.ndim == 1:
            rir = rir[None, ...]

        idx = torch.arange(0, rir.shape[-1], self.N).reshape(-1, 1) + torch.arange(self.N)
        rir_blk = rir[..., idx]  # B, K, N

        # K, 2N
        rir_blk_pad = torch.concat([rir_blk, torch.zeros_like(rir_blk)], dim=-1)
        # complex (B, K, N+1)->(K, B, N+1)
        rir_blk_pad_k = torch.fft.rfft(rir_blk_pad, n=2 * self.N, dim=-1).permute(1, 0, 2)

        return torch.flip(rir_blk_pad_k, dims=[0])

    def update_buff(self, inp_blk):
        dframe = torch.concat([self.buff_frames, inp_blk], dim=-1)
        dframe_k = torch.fft.rfft(dframe, n=2 * self.N, dim=-1)  # B,N+1
        self.buff_frames = inp_blk
        self.buff_inp_blk_fft = torch.roll(self.buff_inp_blk_fft, shifts=-1, dims=0)
        self.buff_inp_blk_fft[-1] = dframe_k

        return self.buff_inp_blk_fft

    def forward(self, inp_blk: torch.Tensor):
        """block-wise operation.
        inp_blk: each block input, with shape (B,T).
        """
        assert inp_blk.ndim > 1
        if not self.init_cache:
            self.reset_cache(inp_blk)

        # K,B,N+1
        cached_frames = self.update_buff(inp_blk)

        # B, N+1
        out = cached_frames * self.rir_fft_blk_weight
        out = torch.real(torch.fft.irfft(out.sum(0)))

        return out[..., self.N :]


def check_test():
    import numpy as np
    import scipy.signal as sps

    # from audiolib import AcousticFeedbackSim as afc_np

    rir = torch.randn(2, 4096)
    inp = torch.randn(2, 512)
    obj = AcousticFeedbackSim(rir, 64)

    out = obj.apply_full(inp)
    print(out.shape)
    out = out.numpy()
    out_ = sps.fftconvolve(inp, rir, "full", axes=-1)  # N1+N2-1
    print(out_.shape)

    # for i in range(2):
    #     obj_ = afc_np(rir[i].numpy(), 64)
    #     # obj_ = obj(rir[i].numpy(), 64)
    #     out_ = obj_.apply_full(inp[i].numpy())

    #     out__ = out[i].numpy()
    print(
        np.allclose(out_[:, :512], out, 1e-5, 1e-4), out.shape, (np.abs(out_[:, :512] - out)).sum()
    )
    print(out[0, :40].round(3))
    print(out_[0, :40].round(3))


if __name__ == "__main__":
    check_test()

    # rir = torch.randn(2, 4096)
    # inp = torch.randn(2, 512)
    # obj = AcousticFeedbackSim(rir, 64).cuda()
    # obj.compute_MSG()

    # out = obj.apply(inp)
