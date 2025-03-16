import argparse
import os
import sys

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate, call


# @hydra.main(config_path="conf", config_name="test_hydra", version_base=None)
@hydra.main(config_path="conf", config_name="train_fig6", version_base=None)
def test(cfg: DictConfig):
    # print(OmegaConf.to_yaml(cfg))

    inp = torch.randn(1, 16000)
    enh = torch.randn(1, 16000)
    loss_l = {}
    print(cfg.loss)
    for l in cfg.loss:
        print(l.name)
        if l.name == "pase_lv":
            loss_l[l.name] = dict(w=l.weight, func=call(l.func))
        else:
            loss_l[l.name] = dict(w=l.weight, func=instantiate(l.func))

    # print(loss_l)
    for k, i in loss_l.items():
        if k == "pase_lv":
            v = i["func"](inp.unsqueeze(1))
        else:
            v = i["func"](inp, enh)
        print(k, v)
    # self.loss_l[k] = instantiate()
    # md_name = cfg.model.md_name
    # dset_name = cfg.dset.dset_name
    # print(md_name, dset_name, "@")

    # net = instantiate(cfg.model.net)
    # print(net)


if __name__ == "__main__":
    test()
    pass
