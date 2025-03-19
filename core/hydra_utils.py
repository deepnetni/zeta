import sys
from typing import List, Any
import re
from omegaconf import DictConfig, OmegaConf


def filter(key, patterns):
    return any(p.search(key) for p in patterns)


def filter_nested_dict(cfg: Any, p: Any):
    return {
        k: filter_nested_dict(v, p) if isinstance(v, dict) else v
        for k, v in cfg.items()
        if not filter(k, p)
    }


def show_cfg(cfg: DictConfig, exclude: List = [".*dset", ".*loss*"]):
    # dict
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    patterns = [re.compile(x) for x in exclude]
    filter_cfg = filter_nested_dict(resolved_cfg, patterns)

    # print(OmegaConf.to_yaml(resolved_cfg))
    print(OmegaConf.to_yaml(filter_cfg))
