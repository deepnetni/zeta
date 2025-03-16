import torch
import torch.nn as nn
import torch.nn.functional as F
from core.models.pase.models.frontend import wf_builder


class PASELoss(nn.Module):
    def __init__(self, cfg_path="core/config/frontend/PASE+.cfg") -> None:
        super().__init__()

        self.pase = wf_builder(cfg_path)
        assert self.pase is not None
        self.pase.cuda()
        self.pase.eval()
        self.pase.load_pretrained("core/pretrained/pase_e199.ckpt", load_last=True, verbose=False)

    def forward(self, sph: torch.Tensor, enh: torch.Tensor):
        """
        sph: B,T
        """
        assert self.pase is not None

        sph_pase = self.pase(sph.unsqueeze(1)).flatten(0)
        enh_pase = self.pase(enh.unsqueeze(1)).flatten(0)
        pase_lv = F.mse_loss(sph_pase, enh_pase)
        return pase_lv
