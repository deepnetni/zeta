#!/usr/bin/env python3
import torch
import torch.nn as nn


def anti_wrapping_function(x):
    return torch.abs(x - torch.round(x / (2 * torch.pi)) * 2 * torch.pi)


def phase_losses(phase_r, phase_g):
    """
    phase_r: target
    """
    ip_loss = torch.mean(anti_wrapping_function(phase_r - phase_g))
    gd_loss = torch.mean(
        anti_wrapping_function(torch.diff(phase_r, dim=1) - torch.diff(phase_g, dim=1))
    )
    iaf_loss = torch.mean(
        anti_wrapping_function(torch.diff(phase_r, dim=2) - torch.diff(phase_g, dim=2))
    )

    phase_loss = ip_loss + gd_loss + iaf_loss
    return phase_loss, dict(
        ip_loss=ip_loss.detach(), gd_loss=gd_loss.detach(), iaf_loss=iaf_loss.detach()
    )


class PhaseLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sph: torch.Tensor, enh: torch.Tensor):
        """
        enh: (B,2,T,F)
        sph: (B,2,T,F)
        """
        eps = torch.finfo(sph.dtype).eps
        phase_sph = torch.atan2(sph[:, 1, ...] + eps, sph[:, 0, ...] + eps).transpose(-1, -2)
        phase_enh = torch.atan2(enh[:, 1, ...] + eps, enh[:, 0, ...] + eps).transpose(-1, -2)

        return phase_losses(phase_sph, phase_enh)

    @property
    def domain(self):
        return "spectrum"
