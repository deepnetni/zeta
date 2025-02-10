import argparse
import os
import sys
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from core.datasets_manager import get_datasets
from core.Trainer_wGAN_for_fig6 import (
    Trainer,
    TrainerGumbelCodebook,
    TrainerMultiOutputs,
)
from core.Trainer_wGAN_VAD_for_fig6 import TrainerVAD
from core.utils.audiolib import audioread, audiowrite
from core.utils.ini_opts import read_ini
from core.utils.logger import cprint
from core.utils.register import tables

warnings.filterwarnings("ignore", category=UserWarning, module="torch")
from core.fig6_baselines import *
from core.JointNSHModel import *


@dataclass
class Eng_conf:
    name: str = "baseline_fig6"
    epochs: int = 25
    desc: str = ""
    info_dir: str = f"{Path.home()}/model_results_trunk/FIG6/fig6_GAN"
    resume: bool = True
    optimizer_name: str = "adam"
    scheduler_name: str = "stepLR"
    valid_per_epoch: int = 1
    vtest_per_epoch: int = 1  # 0 for disabled
    ## the output dir to store the predict files of `vtest_dset` during testing
    vtest_outdir: str = "vtest"
    dsets_raw_metrics: str = "dset_metrics.json"

    train_batch_sz: int = 10  # 6(48)
    train_num_workers: int = 16
    valid_batch_sz: int = 12  # 12
    valid_num_workers: int = 16
    vtest_batch_sz: int = 8  # 12
    vtest_num_workers: int = 16


@dataclass
class Model_conf:
    nframe: int = 512
    nhop: int = 256
    mid_channel: int = 36
    conformer_num: int = 2


@dataclass
class FTCRN_conf:
    nframe: int = 512
    nhop: int = 256


@dataclass
class Conf:
    config: Eng_conf = Eng_conf()
    md_conf: Model_conf = Model_conf()
    ftcrn_conf: FTCRN_conf = FTCRN_conf()


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
    parser.add_argument("--vtest_first", help="valid first", action="store_true")

    parser.add_argument("--valid", help="input directory", action="store_true")
    parser.add_argument("--vtest", help="input directory", action="store_true")

    parser.add_argument("--conf", help="config file")
    parser.add_argument("--name", help="name of the model")

    parser.add_argument("--small", help="small dataset", action="store_true")
    parser.add_argument("--vad", help="vad dataset", action="store_true")

    parser.add_argument("--print", help="print models", action="store_true")

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
    if args.small:
        conf["config"]["info_dir"] = conf["config"]["info_dir"] + "_small"

    return conf


if __name__ == "__main__":
    # import inspect

    args = parse()

    if args.print:
        tables.print()
        sys.exit()

    if args.conf is not None and args.conf != "":
        cfg = fetch_config(args.conf)
    else:
        cfg = asdict(Conf())

    cfg = overrides(cfg, args)

    md_conf = cfg["md_conf"]
    md_name = cfg["config"]["name"]

    if args.small:
        if args.vad:
            train_dset, valid_dset, vtest_dset = get_datasets("FIG6smallVad_SIG")
        else:
            train_dset, valid_dset, vtest_dset = get_datasets("FIG6small_SIG")
    else:
        if args.vad:
            train_dset, valid_dset, vtest_dset = get_datasets("FIG6Vad_SIG")
        else:
            train_dset, valid_dset, vtest_dset = get_datasets("FIG6_SIG")

    if md_name in [
        "baseline_fig6",
        "condConformer",
        "baseline_fig6_linear",
        "xkcConformer",
        "DiTConformer",
    ]:
        Trainer = Trainer
        # Trainer = TrainerPhase
    elif md_name == "GumbelCodebook":
        Trainer = TrainerGumbelCodebook
    elif md_name in [
        "baseline_fig6_vad",
        "condConformerVAD",
        "FTCRN_VAD",
        "FTCRN_BASE_VAD",
    ]:
        Trainer = TrainerVAD
    else:
        Trainer = Trainer

    print("Trainer Class: ", Trainer.__name__)

    cprint.r(f"current: {md_name}")
    model = tables.models.get(md_name)
    assert model is not None
    if "FTCRN" in md_name:
        net = model(**cfg["ftcrn_conf"])
    else:
        net = model(**md_conf)

    if args.train:
        init = cfg["config"]
        eng = Trainer(
            train_dset,
            valid_dset,
            vtest_dset,
            net=net,
            net_D=Discriminator(ndf=16),
            valid_first=args.valid_first,
            vtest_first=args.vtest_first,
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
