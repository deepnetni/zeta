import torch
import torch.nn as nn
import torch.nn.functional as F


def l2_norm(s: Tensor, keepdim=False):
    """
    sum(x^ord) ^ 1/ord
    """
    return torch.linalg.norm(s, dim=-1, keepdim=keepdim)


class SISNRLoss(nn.Module):
    def __init__(self, zero_mean: bool = True) -> None:
        super().__init__()
        self.zero_mean = zero_mean

    def forward(self, sph, enh):
        """
        sph: B,T

        Example:
            >>> a = torch.tensor([1,2,3,4]).float()
            >>> b = torch.tensor([1,2,3,4]).float()
            >>> score = loss_sisnr(a, b)

        Algo:
            s_target = <sph, enh> * sph / sph^2, where <> means inner dot
            e_noise = enh - s_target
            sisnr = 10 * log_10(|s_target|^2 / |e_noise|^2)
        """
        eps = torch.finfo(sph.dtype).eps

        if self.zero_mean is True:
            s = sph - torch.mean(sph, dim=-1, keepdim=True)
            s_hat = enh - torch.mean(enh, dim=-1, keepdim=True)
        else:
            s = sph
            s_hat = enh

        s_target = (
            (torch.sum(s_hat * s, dim=-1, keepdim=True) + eps)
            * s
            / (l2_norm(s, keepdim=True) ** 2 + eps)
        )
        e_noise = s_hat - s_target
        # sisnr_ = 10 * torch.log10((l2_norm(s_target) ** 2 + eps) / (l2_norm(e_noise) ** 2 + eps))
        sisnr = 10 * torch.log10(
            (torch.sum(s_target**2, dim=-1) + eps) / (torch.sum(e_noise**2, dim=-1) + eps)
        )
        return -torch.mean(sisnr)
