import argparse
import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="train mode", action="store_true", default=True)
    parser.add_argument("--pred", help="predict mode", action="store_true")

    args = parser.parse_args()
    return args


@hydra.main(config_path="conf", config_name="train", version_base=None)
def fit(cfg: DictConfig):
    cfg.model.engine.info_dir = os.path.expanduser(cfg.model.engine.info_dir)

    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    args = parse()
    fit()
