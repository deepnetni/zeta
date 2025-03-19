import torch
import torch.nn as nn


class MagEuclideanLoss(nn.Module):
    def __init__(self, l_type="L2"):
        super().__init__()
        self.l_type = l_type

    def forward(self, sph: torch.Tensor, enh: torch.Tensor):
        """
        enh: (B,T,F)
        sph: (B,T,F)
        frame_list: list
        """
        if self.l_type == "L1" or self.l_type == "l1":
            loss_mag = torch.abs(enh - sph).mean()
        elif self.l_type == "L2" or self.l_type == "l2":
            loss_mag = torch.square(enh - sph).mean()
        else:
            raise RuntimeError("only L1 and L2 are supported")
        return loss_mag

    @property
    def domain(self):
        return "spectrum"


class ComMagEuclideanLoss(nn.Module):
    def __init__(self, alpha=0.5, l_type="L2"):
        super().__init__()
        self.alpha = alpha
        self.l_type = l_type

    def forward(self, sph: torch.Tensor, enh: torch.Tensor):
        """
        enh: (B,2,T,F)
        sph: (B,2,T,F)
        frame_list: list
        alpha: scalar
        l_type: str, L1 or L2
        """
        enh_mag, sph_mag = torch.norm(enh, dim=1), torch.norm(sph, dim=1)

        if self.l_type == "L1" or self.l_type == "l1":
            loss_com = torch.abs(enh - sph).mean()
            loss_mag = torch.abs(enh_mag - sph_mag).mean()
        elif self.l_type == "L2" or self.l_type == "l2":
            # F.mse
            loss_com = torch.square(enh - sph).mean()
            loss_mag = torch.square(enh_mag - sph_mag).mean()
        else:
            raise RuntimeError("only L1 and L2 are supported!")
        return self.alpha * loss_com + (1 - self.alpha) * loss_mag

    @property
    def domain(self):
        return "spectrum"


class ComMagSpectrumLoss(nn.Module):
    def __init__(self, factor=0.3, alpha=0.5, l_type="L2"):
        super().__init__()
        self.factor = factor
        self.alpha = alpha
        self.l_type = l_type

    def forward(self, sph: torch.Tensor, enh: torch.Tensor):
        """
        enh: (B,2,T,F)
        sph: (B,2,T,F)
        """
        eps = torch.finfo(sph.dtype).eps
        # B,T,F
        enh_mag = torch.norm(enh, dim=1, keepdim=True) + eps
        sph_mag = torch.norm(sph, dim=1, keepdim=True) + eps

        enh_mag_comp = enh_mag**self.factor
        sph_mag_comp = sph_mag**self.factor
        sph_comp = sph_mag ** (self.factor - 1) * sph
        enh_comp = enh_mag ** (self.factor - 1) * enh

        if self.l_type == "L1" or self.l_type == "l1":
            loss_spec = torch.abs(enh_comp - sph_comp).mean()
            loss_mag = torch.abs(enh_mag_comp - sph_mag_comp).mean()
        elif self.l_type == "L2" or self.l_type == "l2":
            loss_spec = torch.square(enh_comp - sph_comp).mean()
            loss_mag = torch.square(enh_mag_comp - sph_mag_comp).mean()
        else:
            raise RuntimeError("only L1 and L2 are supported!")
        spec_lv = self.alpha * loss_spec
        mag_lv = (1 - self.alpha) * loss_mag
        meta = dict(spec_lv=spec_lv.detach(), mag_lv=mag_lv.detach())
        return spec_lv + mag_lv, meta

    @property
    def domain(self):
        return "spectrum"
