import argparse
import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf


# @hydra.main(config_path="conf", config_name="test_hydra", version_base=None)
@hydra.main(config_path="conf", config_name="train", version_base=None)
def test(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    md_name = cfg.model.md_name
    dset_name = cfg.dset.dset_name
    print(md_name, dset_name, "@")


if __name__ == "__main__":
    test()
    pass
