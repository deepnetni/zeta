import argparse
import os
import sys
import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from core.hydra_utils import show_cfg

# from core.rebuild.FTCRN import Discriminator
# from load_class import load_class
# from core.datasets_manager import get_datasets
# from core.utils.check_flops import check_flops


warnings.filterwarnings("ignore", category=UserWarning, module="torch")


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="train mode", action="store_true", default=True)
    parser.add_argument("--pred", help="predict mode", action="store_true")
    parser.add_argument("--valid_first", help="valid first", action="store_true")
    parser.add_argument("--vtest_first", help="valid first", action="store_true")
    parser.add_argument(
        "--root_save_dir", help="root directory of all results", type=str
    )

    args = parser.parse_args()
    return args


@hydra.main(config_path="conf", config_name="train_fig6", version_base=None)
def fit(cfg: DictConfig):
    # cfg.model.engine.info_dir = os.path.expanduser(cfg.model.engine.info_dir)
    cfg_eng = cfg.model.engine
    cfg_eng.info_dir = cfg_eng.info_dir + "_" + cfg.dset.dset_name

    show_cfg(cfg.model)

    # net = instantiate(cfg.model.net)
    # train_dset = instantiate(cfg.dset.train)
    # valid_dset = instantiate(cfg.dset.valid)
    # vtest_dset = instantiate(cfg.dset.vtest)
    # eng = instantiate(
    #     cfg.model.engine,
    #     train_dset,
    #     valid_dset,
    #     vtest_dset,
    #     net=net,
    #     net_D=Discriminator(ndf=16),
    # )

    eng = instantiate(cfg.model.engine)

    print(eng)
    eng.fit()


if __name__ == "__main__":
    fit()
