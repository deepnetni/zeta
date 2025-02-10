import sys
import argparse
import os
from dataclasses import asdict, dataclass, field

import torch
import torch.nn.functional as F
import yaml


from tqdm import tqdm

from core.datasets_manager import get_datasets
from core.rebuild.fig6_baselines import *
from core.utils.audiolib import audioread, audiowrite
from core.utils.ini_opts import read_ini
from core.utils.logger import cprint
from core.utils.register import tables
from core.Trainer_wGAN_for_fig6 import Trainer


@dataclass
class Eng_conf:
    name: str = "FTCRN"
    epochs: int = 50
    desc: str = ""
    info_dir: str = r"/home/deepni/model_results_trunk/FIG6/trained_FTCRN_GAN"
    resume: bool = True
    optimizer_name: str = "adam"
    scheduler_name: str = "stepLR"
    valid_per_epoch: int = 1
    vtest_per_epoch: int = 5  # 0 for disabled
    ## the output dir to store the predict files of `vtest_dset` during testing
    vtest_outdir: str = "vtest"
    dsets_raw_metrics: str = "dset_metrics.json"

    train_batch_sz: int = 14
    train_num_workers: int = 16
    valid_batch_sz: int = 24
    valid_num_workers: int = 16
    vtest_batch_sz: int = 24
    vtest_num_workers: int = 16


@dataclass
class Model_conf:
    nframe: int = 512
    nhop: int = 256


@dataclass
class Conf:
    config: Eng_conf = Eng_conf()
    md_conf: Model_conf = Model_conf()


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", help="test mode", action="store_true")
    parser.add_argument("--train", help="train mode", action="store_true", default=True)
    parser.add_argument("--pred", help="predict mode", action="store_true")

    parser.add_argument("--ckpt", help="ckpt path", type=str)
    parser.add_argument("--epoch", help="epoch", type=int)
    parser.add_argument("--src", help="input directory", type=str)
    parser.add_argument("--out", help="predicting output directory", type=str)
    parser.add_argument(
        "--root_save_dir", help="root directory of all results", type=str
    )
    parser.add_argument("--valid_first", help="valid first", action="store_true")

    parser.add_argument("--valid", help="input directory", action="store_true")
    parser.add_argument("--vtest", help="input directory", action="store_true")

    parser.add_argument("--conf", help="config file")
    parser.add_argument("--name", help="name of the model")

    args = parser.parse_args()
    args.train = False if args.test or args.pred else True
    return args


def fetch_config(cfg_fname):
    print("##", cfg_fname)
    if os.path.splitext(cfg_fname)[-1] == ".ini":
        cfg = read_ini(cfg_fname)
    elif os.path.splitext(cfg_fname)[-1] == ".yaml":
        with open(cfg_fname, "r") as fp:
            cfg = yaml.load(fp, Loader=yaml.FullLoader)
    else:
        raise RuntimeError("File not supported.")

    return cfg


def overrides(conf, args):
    def value(v, default_v):
        return v if v is not None else default_v

    conf["config"]["name"] = value(args.name, conf["config"]["name"])
    conf["config"]["epochs"] = value(args.epoch, conf["config"]["epochs"])

    return conf


if __name__ == "__main__":
    args = parse()

    if args.conf is not None and args.conf != "":
        cfg = fetch_config(args.conf)
    else:
        cfg = asdict(Conf())

    cfg = overrides(cfg, args)

    md_conf = cfg["md_conf"]
    md_name = cfg["config"]["name"]

    tables.print() if args.train else None

    cprint.r(f"current: {md_name}")
    model = tables.models.get(md_name)
    assert model is not None
    net = model(**md_conf)

    train_dset, valid_dset, vtest_dset = get_datasets("FIG6_SIG")

    if args.train:
        init = cfg["config"]

        init = cfg["config"]
        eng = Trainer(
            train_dset,
            valid_dset,
            vtest_dset,
            net=net,
            net_D=Discriminator(ndf=16),
            valid_first=args.valid_first,
            root_save_dir=args.root_save_dir,
            **init,
        )
        print(eng)

        eng.fit()

    elif args.pred:  # pred
        assert args.ckpt is not None
        assert args.out is not None
        net.load_state_dict(torch.load(args.ckpt))
        net.cuda()
        net.eval()

        if args.valid:
            dset = valid_dset
        elif args.vtest:
            dset = vtest_dset
        else:
            raise RuntimeError("not supported.")

        for d, HL, fname in tqdm(dset):
            d = d.cuda()
            HL = HL.cuda()

            with torch.no_grad():
                out = net(d, HL)
            out = out.cpu().detach().squeeze().numpy()

            fout = os.path.join(args.out, fname)
            outd = os.path.dirname(fout)
            os.makedirs(outd) if not os.path.exists(outd) else None
            audiowrite(fout, out, sample_rate=16000)
