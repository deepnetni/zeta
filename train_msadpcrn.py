import argparse
import os
import warnings
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml

# from matplotlib import pyplot as plt
from torchmetrics.functional.audio import signal_distortion_ratio as SDR

# from utils.conv_stft_loss import MultiResolutionSTFTLoss
from tqdm import tqdm

from core.datasets_manager import get_datasets
from core.utils.audiolib import audioread, audiowrite
from core.utils.gcc_phat import gcc_phat
from core.utils.ini_opts import read_ini
from core.utils.logger import cprint
from core.MSA_DPCRN_FAST import *
from core.Trainer_for_AEC import Trainer


@dataclass
class Eng_conf:
    name: str = "msa_dpcrn"
    epochs: int = 100
    desc: str = ""
    info_dir: str = r"E:\model_results_trunk\AEC\trained_msadpcrn_align"
    resume: bool = True
    optimizer_name: str = "adam"
    scheduler_name: str = "stepLR"
    valid_per_epoch: int = 1
    vtest_per_epoch: int = 5  # 0 for disabled
    ## the output dir to store the predict files of `vtest_dset` during testing
    vtest_outdir: str = "vtest"
    dsets_raw_metrics: str = "dset_metrics.json"
    train_batch_sz: int = 12
    train_num_workers: int = 16
    valid_batch_sz: int = 4
    valid_num_workers: int = 8
    vtest_batch_sz: int = 4
    vtest_num_workers: int = 8


@dataclass
class Model_conf:
    nframe: int = 512
    nhop: int = 256
    nfft: int = 512
    cnn_num: List[int] = field(default_factory=lambda: [16, 32, 64, 64])
    stride: List[int] = field(default_factory=lambda: [2, 2, 1, 1])  # 65
    rnn_hidden_num: int = 64


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
    parser.add_argument("--draw", help="input directory", action="store_true")

    parser.add_argument("--conf", help="config file")
    parser.add_argument("--name", help="name of the model")
    parser.add_argument("--online", help="frame 2 frame mode", action="store_true")

    args = parser.parse_args()
    args.train = False if args.test or args.pred else True
    return args


def split_to_frames(x, nframe, nhop):
    """
    input: B,T
    """
    pad = nframe // 2

    x = F.pad(x, (pad, pad))

    N = (x.size(-1) // nhop) * nhop
    print(N)

    idx = torch.arange(nframe)
    idx = torch.arange(0, N - nhop, nhop).unsqueeze(-1) + idx
    print(idx.shape)


def synthe_to_waves(x, nframe, nhop):
    pass


def fetch_config(cfg_fname=None):
    if cfg_fname is None:
        return asdict(Conf())

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

    cfg = fetch_config(args.conf)
    cfg = overrides(cfg, args)

    md_conf = cfg["md_conf"]
    md_name = cfg["config"]["name"]

    tables.print() if args.train else None
    cprint.r(f"current: {md_name}")

    model = tables.models.get(md_name)
    assert model is not None
    net = model(**md_conf)

    if args.train:
        init = cfg["config"]
        train_dset, valid_dset, vtest_dset = get_datasets("AECChallenge")

        init = cfg["config"]
        eng = Trainer(
            train_dset,
            valid_dset,
            vtest_dset,
            # vpred_dset=vpred_dset,
            net=net,
            valid_first=args.valid_first,
            nframe=512,
            nhop=256,
            root_save_dir=args.root_save_dir,
            **init,
        )

        print(eng)

        eng.fit()

    elif args.online:
        assert args.ckpt is not None
        assert args.out is not None
        net = MSA_DPCRN_SPEC_ALIGN_online(**cfg["md_conf"])
        net.load_state_dict(torch.load(args.ckpt)["net"])
        net.cuda()
        net.eval()

        if len(args.src) == 2:
            mic, _ = audioread(args.src[0])
            ref, _ = audioread(args.src[1])
        elif len(args.src) == 1:
            data, _ = audioread(args.src[0])
            mic, ref = data[..., 0], data[..., 1]
        else:
            raise RuntimeError("input format error.")

        align = True
        if align:
            fs = 16000
            tau, _ = gcc_phat(mic, ref, fs=fs, interp=1)
            tau = max(0, int((tau - 0.001) * fs))
            ref = np.concatenate([np.zeros(tau), ref], axis=-1, dtype=np.float32)[
                : mic.shape[-1]
            ]
        else:
            N = min(len(mic), len(ref))
            N = 16000 * 5
            mic = mic[:N]
            ref = ref[:N]

        stft = STFT(nframe=512, nhop=256, win="hann sqrt").cuda()

        d_mic = mic.cuda()
        d_ref = ref.cuda()
        d_mic = stft.transform(d_mic)  # b,2,t,f
        d_ref = stft.transform(d_ref)

        out_list = []
        state = None
        for nt in tqdm(range(d_mic.size(2))):
            mic_frame = d_mic[..., nt, :].unsqueeze(2)  # B,2,1(T),F
            ref_frame = d_ref[..., nt, :].unsqueeze(2)

            with torch.no_grad():
                # * w: b,2,t,f; out_frame: b,2,1,f
                out_frame, w, state = net(mic_frame, ref_frame, state)
                out_list.append(out_frame)  # B,

        out = torch.concat(out_list, dim=2)  # B,2,t,f
        out = stft.inverse(out).cpu().squeeze().numpy()  # B,T

        outd = os.path.dirname(__file__) if args.out is None else args.out
        fout = os.path.join(outd, "enh.wav")
        os.makedirs(outd) if not os.path.exists(outd) else None
        audiowrite(fout, out, sample_rate=16000)

    else:  # pred
        assert args.ckpt is not None
        assert args.src is not None
        net.load_state_dict(torch.load(args.ckpt)["net"])
        net.cuda()
        net.eval()

        if len(args.src) == 2:
            mic, _ = audioread(args.src[0])
            ref, _ = audioread(args.src[1])
        elif len(args.src) == 1:
            data, _ = audioread(args.src[0])
            mic, ref = data[..., 0], data[..., 1]
        else:
            raise RuntimeError("input format error.")

        align = True
        if align:
            fs = 16000
            tau, _ = gcc_phat(mic, ref, fs=fs, interp=1)
            tau = max(0, int((tau - 0.001) * fs))
            ref = np.concatenate([np.zeros(tau), ref], axis=-1, dtype=np.float32)[
                : mic.shape[-1]
            ]
        else:
            N = min(len(mic), len(ref))
            mic = mic[:N]
            ref = ref[:N]

        d_mic = mic.cuda()
        d_ref = ref.cuda()

        with torch.no_grad():
            # * w: b,2,t,f
            out, w = net(d_mic, d_ref)
        out = out.cpu().detach().squeeze().numpy()

        outd = os.path.dirname(__file__) if args.out is None else args.out
        fout = os.path.join(outd, "enh.wav")
        os.makedirs(outd) if not os.path.exists(outd) else None
        audiowrite(fout, out, sample_rate=16000)
