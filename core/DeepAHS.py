import torch
import torch.nn as nn
import torch.nn.functional as F

import einops
from einops import rearrange
from einops.layers.torch import Rearrange

from comps.overlapAdd import OverlapAdd
from comps.conv_stft import STFT, SpecFeat


class DeepAHS(nn.Module):
    def __init__(self, nblk=64, winLen=9, **kwargs):
        super().__init__()

        self.nblk = nblk
        self.nwin = winLen
        self.ola = OverlapAdd(nblk)
        self.ola_ref = OverlapAdd(nblk)
        self.stft = STFT(nblk * 2, nblk, win="hann sqrt")

        self.fc1 = nn.Linear(368, 1)
        self.gru1 = nn.GRU(nblk + 1, 257, batch_first=True)

    def reset_buff(self, inp):
        self.ola.reset_buff(inp)
        self.ola_ref.reset_buff(inp)

    def forward(self, inp, ref, h=None, online=True):
        if not online:
            xk = self.stft.transform(inp)  # b,2,t,f
            xkr = self.stft.transform(ref)  # b,2,t,f
        else:
            xk = self.ola.transform(inp)  # b,2,1,f
            xkr = self.ola_ref.transform(ref)

        # b,1,t,f
        xk_mag = xk.pow(2).sum(1, keepdim=True).sqrt()
        xk_pha = torch.atan2(xk[:, (1,), ...], xk[:, (0,), ...])

        # ! log-power spectra
        lps = SpecFeat.logPowerSpectra(xk)
        lps_ref = SpecFeat.logPowerSpectra(xkr)
        # b,c,t,f->b,t,f,c
        lps_mix = torch.concat([lps, lps_ref], dim=1).permute(0, 2, 3, 1)

        xk_cpx = torch.complex(xk[:, 0, ...], xk[:, 1, ...])
        xkr_cpx = torch.complex(xkr[:, 0, ...], xk[:, 1, ...])
        # b,2,t,f,
        xk_mix_cpx = torch.stack([xk_cpx, xkr_cpx], dim=1)

        # ! Channel convariance
        xk_mix_cpx = xk_mix_cpx - xk_mix_cpx.mean(dim=1, keepdim=True)
        xk_mix_cov = torch.einsum("bctf,bvtf->btfcv", xk_mix_cpx, xk_mix_cpx.conj())
        mask = torch.tril(torch.ones(2, 2, dtype=torch.bool, device=inp.device))
        xk_ch_cov = xk_mix_cov[..., mask]  # b,t,f, W*(W+1)//2
        xk_ch_cov = torch.concat([xk_ch_cov.real, xk_ch_cov.imag], dim=-1)

        # ! Temporal Corr
        if online == False:
            xk_t_corr = SpecFeat.corr(xk, dim=2, winLen=9, tril=True)
            xk_t_corr = einops.rearrange(xk_t_corr, "b c t f n->b t f (c n)")

            xkr_t_corr = SpecFeat.corr(xkr, dim=2, winLen=9, tril=True)
            xkr_t_corr = einops.rearrange(xkr_t_corr, "b c t f n->b t f (c n)")
        else:
            pass

        # ! Frequency Corr
        if online == False:
            xk_f_corr = SpecFeat.corr(xk, dim=3, winLen=9, tril=True)
            xk_f_corr = einops.rearrange(xk_f_corr, "b c t f n->b t f (c n)")

            xkr_f_corr = SpecFeat.corr(xkr, dim=3, winLen=9, tril=True)
            xkr_f_corr = einops.rearrange(xkr_f_corr, "b c t f n->b t f (c n)")
        else:
            pass

        # b,t,f,c
        xk = torch.concat(
            [lps_mix, xk_ch_cov, xk_t_corr, xk_t_corr, xkr_f_corr, xkr_f_corr], dim=-1
        )
        xk = self.fc1(xk).squeeze(-1)
        print(xk.shape)
        x, h = self.gru1(xk, h)
        print(x.shape)


if __name__ == "__main__":
    inp = torch.randn(3, 128)
    ref = torch.randn(3, 128)
    net = DeepAHS()
    net.reset_buff(inp)
    net(inp, ref, online=False)
