import torch
import torch.nn as nn

from core.models.conv_stft import STFT
from core.models.PMSQE.pmsqe_asteroid import SingleSrcPMSQE
from einops import rearrange


def power(spec, freq_dim=-1):
    # B,T,2F -> ((B,T,F), (B,T,F))(chunk) -> B,T,F,2(stack) -> B,T,F(sum)
    return torch.stack(torch.chunk(spec, 2, dim=freq_dim), dim=-1).pow(2).sum(dim=-1)


class PMSQELoss(nn.Module):
    """Perceptally-Motivated Speech Quality"""

    def __init__(self) -> None:
        super().__init__()
        self.stft = STFT(512, 256)

    def forward(self, sph, enh, pad_mask=None):
        """
        sph: B,T; or B,C,T,F
        enh: B,T; or B,C,T,F

        pad_mask: indicate the padding frames with shape B, T, 1 where 1 for valid frame.
        """
        if sph.ndim == 2:
            sph_spec = self.stft.transform(sph)
            enh_spec = self.stft.transform(enh)
        elif sph.ndim == 4:
            sph_spec = sph
            enh_spec = enh
        else:
            raise RuntimeError("shape not supported.")
        assert sph_spec.shape[-1] == 257

        if pad_mask is None:
            pad_mask = torch.ones(enh_spec.shape[0], enh_spec.shape[2], 1, device=enh_spec.device)

        # b,2,t,f -> b,t,2f
        sph_spec = rearrange(sph_spec, "b c t f->b t (c f)")
        enh_spec = rearrange(enh_spec, "b c t f->b t (c f)")

        power_sph_spec = power(sph_spec)
        power_enh_spec = power(enh_spec)

        # wD: Mean symmetric distortion.
        # wDA: Mean asymmetric distortion.
        # pmsq = PMSQE().cuda()
        # wD, wDA = pmsq(power_enh_spec, power_sph_spec, pad_mask)
        # alpha = 0.1
        # score = alpha * (wD + 0.309 * wDA)

        pmsq = SingleSrcPMSQE(sample_rate=16000).cuda()
        score = pmsq(power_enh_spec, power_sph_spec, pad_mask)
        score = score.mean()

        return score
