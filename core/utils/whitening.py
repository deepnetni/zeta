import torch
import torch.nn as nn
import torch.nn.functional as F

import einops
from einops import rearrange
from einops.layers.torch import Rearrange


class Whitening(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def _unfold_temporal(self, x, N):
        # B, T, D = x.shape
        x_pad = F.pad(x, pad=(0, 0, N - 1, 0))  # (DP,DF,TP,TF)
        # unfold along time dimension, (B, T, F, N)
        x_unfolded = x_pad.unfold(dimension=1, size=N, step=1)

        return x_unfolded

    def _PCA(self, x: torch.Tensor):
        """Performs PCA whitening on the input data x.
        x: [B, T, F], magnitude spectrum.
        Returns whitened data of same shape.
        """

        # * Step 1: Zero-center
        # * BTF->B,F,T
        x = x.transpose(-1, -2)
        x_center = x - x.mean(dim=-1, keepdim=True)

        # * Step 2: Covariance matrix: shape (B, F, F)
        cov = x_center @ x_center.transpose(-1, -2) / (x.shape[-1] - 1)

        # * Step 3: Eigen-decomposition, `eigvecs` arranged by columns.
        # * eigvals, shape (B,F); eigvecs, shape (B,F,F)
        # Ax = ax
        # * cov @ eigvecs[:, 0] = eigvals[:, 0] * eigvecs[:, 0]
        eigvals, eigvecs = torch.linalg.eigh(cov)  # ascending order
        # * Step 3.2: Sort in descending order
        idx = torch.argsort(eigvals, descending=True)
        eigvals = torch.gather(eigvals, dim=-1, index=idx)
        # * idx, B,T,F->B,T,1,F->B,T,F,F
        idx = idx.unsqueeze(-2).expand(-1, eigvecs.size(-2), -1)
        eigvecs = torch.gather(eigvecs, dim=-1, index=idx)

        # * Step 5: Whitening transform
        eps = 1e-5
        D_inv_sqrt = torch.diag_embed(1.0 / torch.sqrt(eigvals + eps))  # (F, F)
        whitening_matrix = D_inv_sqrt @ eigvecs.transpose(-1, -2)  # (F, F)
        x_whitened = whitening_matrix @ x_center

        # BFT
        return x_whitened, eigvecs

    def PCA(self, x, dim_out=None):
        x, _ = self._PCA(x)  # BFT
        if dim_out is not None:
            x = x[:, :dim_out, :]
        return x.transpose(-1, -2)

    def ZCA(self, x, dim_out=None):
        """Zero-phase Component Analysis Whitening."""
        x, eigvecs = self._PCA(x)
        x = eigvecs @ x
        if dim_out is not None:
            x = x[:, :dim_out, :]
        return x.transpose(-1, -2)


if __name__ == "__main__":
    inp = torch.randn(2, 12, 5)
    opt = Whitening()
    # out = opt.PCA(inp)
    out = opt.ZCA(inp, 2)
    out = out.transpose(-1, -2)
    cor = out[0, ...] @ out[0, ...].T / 11
    print(out.shape, cor)
