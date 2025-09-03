import os
import sys
import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from core.hydra_utils import show_cfg

# from load_class import load_class
# from core.datasets_manager import get_datasets
# from core.utils.check_flops import check_flops


warnings.filterwarnings("ignore", category=UserWarning, module="torch")


@hydra.main(config_path="conf", config_name="train_fig6", version_base=None)
def fit(cfg: DictConfig):
    # cfg.model.engine.info_dir = os.path.expanduser(cfg.model.engine.info_dir)
    cfg_eng = cfg.model.engine
    if cfg.dset.dset_name:
        cfg_eng.info_dir = cfg_eng.info_dir + "_" + cfg.dset.dset_name

    show_cfg(cfg.model)
    # sys.exit()

    if cfg.pred is False:
        eng = instantiate(cfg.model.engine)
        print(eng)
        eng.fit()
    else:
        eng = instantiate(cfg.model.predictor)
        eng.run()


if __name__ == "__main__":
    fit()
