from typing import Dict
from omegaconf import DictConfig, OmegaConf


def show_cfg(cfg: DictConfig):
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    print(OmegaConf.to_yaml(resolved_cfg))
