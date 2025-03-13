import os
import sys
import torch
import torch.nn as nn
from pathlib import Path


class PredHandler(object):
    def __init__(self, base_directory: str, net: nn.Module, ckpt: str) -> None:
        base_dir = Path(base_directory)
        if not base_dir.exists():
            raise RuntimeError(f"{base_dir} not exists.")
        self.out_dir: Path = base_dir / "output"

        if os.path.isabs(ckpt) and os.path.isfile(ckpt):
            ckpt_file = ckpt
        else:
            ckpt_file = str(base_dir / "checkpoints" / f"epoch_{ckpt:0>4}.pth")

        self.net = net
        self.net.load_state_dict(torch.load(ckpt_file)["net"])
        self.net.cuda()
        self.net.eval()

    @torch.no_grad()
    def __call__(self, fname: str):
        pass
