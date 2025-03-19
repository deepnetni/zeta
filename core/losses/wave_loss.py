import torch
import torch.nn as nn


class WaveLoss(nn.Module):
    def __init__(self, l_type="L2"):
        super().__init__()
        self.l_type = l_type

    def forward(self, sph: torch.Tensor, enh: torch.Tensor):
        """
        enh: (B,T)
        sph: (B,T)
        """
        if self.l_type == "L1" or self.l_type == "l1":
            # F.l1_loss
            loss_v = torch.abs(enh - sph).mean()
        elif self.l_type == "L2" or self.l_type == "l2":
            # F.mse_loss
            loss_v = torch.square(enh - sph).mean()
        else:
            raise RuntimeError("only L1 and L2 are supported")
        return loss_v

    @property
    def domain(self):
        return "time"
