import argparse
import os
import sys
import shutil
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
    TrainerHAMGAN,
    TrainerMGAN,
    TrainerMultiOutputs,
    TrainerforBaselines,
    TrainerforMPSENET,
)
from core.Trainer_wGAN_VAD_for_fig6 import TrainerSEVAD, TrainerVAD
from core.utils.audiolib import audioread, audiowrite
from core.utils.ini_opts import read_ini
from core.utils.logger import cprint
from core.utils.register import tables
from core.rebuild.FTCRN import *


warnings.filterwarnings("ignore", category=UserWarning, module="torch")
from core.fig6_baselines import *
from core.JointNSHModel import *
import core.NUNet_TLS
import core.DCCRN
import core.CRN
import core.aia_trans_official
import core.MP_SENet
import core.cmgan_generator


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
    vtest_per_epoch: int = 0  # 0 for disabled
    ## the output dir to store the predict files of `vtest_dset` during testing
    vtest_outdir: str = "vtest"
    dsets_raw_metrics: str = "dset_metrics.json"

    train_batch_sz: int = 3  # 6(48), 10 for ftcrn, 12
    train_num_workers: int = 16
    valid_batch_sz: int = 6  # 12

    valid_num_workers: int = 16
    vtest_batch_sz: int = 8  # 12
    vtest_num_workers: int = 16


@dataclass
class Model_conf:
    nframe: int = 512
    nhop: int = 256
    mid_channel: int = 48  # 48, 36, 60
    conformer_num: int = 2  # 2


@dataclass
class FTCRN_conf:
    nframe: int = 512
    nhop: int = 256


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

    parser.add_argument("--ckpt", help="ckpt path", type=str, default="25")
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
    parser.add_argument("--dset", help="name of the dataset")

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
    if args.dset:
        conf["config"]["info_dir"] = conf["config"]["info_dir"] + "_" + args.dset

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

    if args.dset is not None:
        dset_name = args.dset
    else:
        if args.vad:
            dset_name = "FIG6Vad"
        else:
            dset_name = "FIG6"

    if md_name == "baseVADSE":
        dset_name = "FIG6SE"
    train_dset, valid_dset, vtest_dset = get_datasets(dset_name)

    if md_name in [
        "baseline_fig6",
        "baseline_fig6_linear",
        "condConformer",
        "IterCondConformer",
        "FTCRN",
        "FTCRN_COND",
        "FTCRN_COND_Iter",
    ]:
        Trainer = Trainer
    elif md_name == "GumbelCodebook":
        Trainer = TrainerGumbelCodebook
    elif md_name == "CMGAN_FIG6":
        Trainer = TrainerMGAN
    elif md_name in [
        "baseline_fig6_vad",
        "condConformerVAD",
        "condConformerVAD8",
        "FTCRN_VAD",
        "FTCRN_BASE_VAD",
    ]:
        Trainer = TrainerVAD
    elif md_name == "baseVADSE":
        Trainer = TrainerSEVAD
    elif md_name == "MP_SENetFIG6":
        Trainer = TrainerforMPSENET
    elif md_name == "HAMGAN":
        Trainer = TrainerHAMGAN
    else:
        # Trainer = Trainer
        Trainer = TrainerforBaselines

    print("Trainer Class: ", Trainer.__name__)

    cprint.r(f"current: {md_name}")
    model = tables.models.get(md_name)
    assert model is not None
    if "FTCRN" in md_name:
        net = model(**cfg["ftcrn_conf"])
    elif "baseline" in md_name or "Conformer" in md_name or "baseVADSE" in md_name:
        net = model(**md_conf)
    else:
        net = model()

    if args.train:
        assert (
            train_dset is not None and valid_dset is not None and vtest_dset is not None
        )

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
        # assert args.ckpt is not None
        # assert args.out is not None

        if args.root_save_dir is None:
            base_dir = Path(cfg["config"]["info_dir"]) / md_name
        else:
            base_dir = Path(cfg["config"]["info_dir"]) / args.root_save_dir

        if not base_dir.exists():
            raise RuntimeError(f"{base_dir} not exists.")

        out_dir: Path = base_dir / "output"
        if args.valid:
            dset = valid_dset
        elif args.vtest:
            dset = vtest_dset
        elif args.dset:
            dset = vtest_dset
        else:
            raise RuntimeError("not supported.")

        cprint.y(f"output:{out_dir}")

        if os.path.isabs(args.ckpt) and os.path.isfile(args.ckpt):
            ckpt_file = args.ckpt
        else:
            ckpt_file = base_dir / "checkpoints" / f"epoch_{args.ckpt:0>4}.pth"

        net.load_state_dict(torch.load(ckpt_file)["net"])
        net.cuda()
        net.eval()

        check = 0
        for d, HL, fname in tqdm(dset):
            fout = os.path.join(str(out_dir), fname)
            outd = os.path.dirname(fout)
            if check == 0:
                if not os.path.exists(outd):
                    os.makedirs(outd)
                else:
                    break
                check = 1

            d = d.cuda()
            HL = HL.cuda()

            if args.vad:
                with torch.no_grad():
                    out, vad = net(d, HL)
            else:
                with torch.no_grad():
                    out = net(d, HL)

            out = out.cpu().detach().squeeze().numpy()

            audiowrite(fout, out, sample_rate=16000)
