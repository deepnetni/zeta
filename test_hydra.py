import argparse
import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate


# @hydra.main(config_path="conf", config_name="test_hydra", version_base=None)
@hydra.main(config_path="conf", config_name="train_fig6", version_base=None)
def test(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # md_name = cfg.model.md_name
    # dset_name = cfg.dset.dset_name
    # print(md_name, dset_name, "@")

    # net = instantiate(cfg.model.net)
    # print(net)


if __name__ == "__main__":
    test()
    pass
