import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import soundfile as sf

from utils.audiolib import activitydetector, audioread, audiowrite, check_power
from utils.HAids.PyFIG6.pyFIG6 import FIG6_compensation_vad
from utils.mp_decoder import mpStarMap
from utils.trunk import CHiMe3
from utils.vad import VAD


@mpStarMap(10)
def work(filenum: int, dset, hl, outdir, fs=16000):
    # out = re.search(r"(\w*)-\d+_(fileid_\d+\.wav)", fname)
    # src_fname = re.sub(r".*_(fileid_\d+\.wav)", r"clean_\1", fname)
    # print(hl.shape, src_path, fpath)

    dnsy, dsph = dset[filenum]
    dnsy = dnsy.numpy()
    dsph = dsph.numpy()

    ndata = dnsy.mean(axis=-1)
    ht = hl[np.random.choice(len(hl))]

    sdata_fig6 = FIG6_compensation_vad(ht, dsph, fs, 128, 64)
    ndata_fig6 = FIG6_compensation_vad(ht, ndata, fs, 128, 64)

    if not check_power(sdata_fig6, -35):
        # power = 10 * np.log10((dsph**2).mean() + 1e-5)
        # power_ = 10 * np.log10((dsph**2).sum() / np.count_nonzero(d_vad) + 1e-5)
        # act = activitydetector(dsph)
        # print(f"skip {filenum}", power, act, power_)
        return 0

    vad_detect = VAD(10, fs, level=2)
    vad_detect.reset()
    x_vad = np.ones_like(dsph)
    d_vad = vad_detect.vad_waves(dsph)  # T,
    x_vad[: len(d_vad)] = d_vad

    meta = dict(HL=json.dumps(ht.tolist()))

    N = len(sdata_fig6)
    dsph = dsph[:N]
    dnsy = dnsy[:N, ...]
    x_vad = x_vad[:N]
    assert len(sdata_fig6) == len(dsph) == len(dnsy) == len(ndata_fig6)

    sf.write(f"{outdir}/{filenum}_src.wav", dsph, fs)
    sf.write(f"{outdir}/{filenum}_nearend.wav", dnsy, fs)
    sf.write(f"{outdir}/{filenum}_nearend_fig6.wav", ndata_fig6, fs)
    target = np.stack([sdata_fig6, x_vad], axis=-1)  # T,2
    sf.write(f"{outdir}/{filenum}_target.wav", target, fs)

    with open(f"{outdir}/{filenum}.json", "w+") as fp:
        json.dump(meta, fp, indent=2)

    return len(dsph) / fs


def parser():
    parser = argparse.ArgumentParser(description="--src --out /yy/zz")
    parser.add_argument("--train", help="", action="store_true")
    parser.add_argument("--valid", help="", action="store_true")
    parser.add_argument("--test", help="", action="store_true")
    parser.add_argument("--out", help="out dirname")
    parser.add_argument(
        "--src",
        help="out dirname",
        default="/home/deepni/disk/DNS-Challenge/datasets/test_set/synthetic/no_reverb",
    )

    return parser.parse_args()


if __name__ == "__main__":
    """
    Example:
        >>> python augment_wdrc.py --yaml xx.yaml --outdir /home/deepni/datasets/dns_p09_50h/ --time 50
    """
    args = parser()

    if not os.path.exists(args.out):
        os.makedirs(args.out)
    else:
        shutil.rmtree(args.out)
        os.makedirs(args.out)

    out = {}

    dset_p = "/home/deepni/datasets/CHiME3"
    if args.train:
        dset = CHiMe3(dset_p, subdir="train", nlen=3.0, min_len=1.0)
    elif args.valid:
        dset = CHiMe3(dset_p, subdir="dev", nlen=3.0, min_len=1.0)
    elif args.test:
        dset = CHiMe3(dset_p, subdir="test", nlen=3.0, min_len=1.0)
    else:
        raise RuntimeError()

    audiogram_f = "./template/audiogram.txt"
    with open(audiogram_f, "r", encoding="utf-8") as fp:
        ctx = [l.strip().replace("\t", ",").replace(" ", ",") for l in fp.readlines()]
        ctx = [list(map(float, l.split(","))) for l in ctx]
        ctx = np.array(ctx)  # N, 6

    # work(range(len(train)), dset=train, hl=ctx, outdir=args.out)
    out = work(range(len(dset)), dset=dset, hl=ctx, outdir=args.out)
    # work(range(len(vtest)), dset=vtest, hl=ctx, outdir=args.out)

    # dnsy = Path(args.src) / "noisy"
    # dsrc = Path(args.src) / "clean"
    # nsy_list = list(map(str, dnsy.rglob("[!.]*.wav")))
    # nfile = len(nsy_list)

    # hl_pick = np.random.choice(len(ctx), size=(nfile))
    # out["aug"] = work(nsy_list, ctx[hl_pick], range(nfile), outdir=args.out, srcdir=str(dsrc))

    num = np.array(out)
    print(f"Generating {num.sum() / 3600:.2f} hours, {np.count_nonzero(num)}/{len(dset)}.")
