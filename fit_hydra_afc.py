import os
import sys
import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from core.hydra_utils import show_cfg
import ast


warnings.filterwarnings("ignore", category=UserWarning, module="torch")


@hydra.main(config_path="conf_afc", config_name="train_afc", version_base=None)
def fit(cfg: DictConfig):
    # cfg.model.engine.info_dir = os.path.expanduser(cfg.model.engine.info_dir)
    cfg_eng = cfg.model.engine
    if cfg.dset.dset_name:
        cfg_eng.info_dir = cfg_eng.info_dir + "_" + cfg.dset.dset_name

    show_cfg(cfg.model)
    # sys.exit()

    if cfg.vep:
        eng = instantiate(cfg.model.engine)
        # epochs = ast.literal_eval(cfg.vep)
        eng.eval_epoch(list(cfg.vep))
    elif cfg.pred is False:
        eng = instantiate(cfg.model.engine)
        print(eng)
        eng.fit()
    else:
        eng = instantiate(cfg.model.predictor)
        eng.run()


if __name__ == "__main__":
    fit()
