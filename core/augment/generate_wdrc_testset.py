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

from utils.audiolib import audioread, audiowrite
from utils.HAids.PyFIG6.pyFIG6 import FIG6_compensation_vad
from utils.mp_decoder import mpStarMap
from utils.vad import VAD


@mpStarMap(10)
def work(fpath: str, hl, filenum: int, srcdir, outdir):
    _, fname = os.path.split(fpath)

    # out = re.search(r"(\w*)-\d+_(fileid_\d+\.wav)", fname)
    src_fname = re.sub(r".*_(fileid_\d+\.wav)", r"clean_\1", fname)
    src_path = os.path.join(srcdir, src_fname)
    # print(hl.shape, src_path, fpath)

    meta = dict(clean=src_fname, noisy=fname, HL=json.dumps(hl.tolist()))
    sdata, fs = audioread(src_path)
    ndata, fs2 = audioread(fpath)

    assert fs == fs2

    vad_detect = VAD(10, fs, level=2)
    vad_detect.reset()
    # x_vad = np.ones_like(sdata)
    x_vad = vad_detect.vad_waves(sdata)  # T,
    # x_vad[: len(d_vad)] = d_vad

    sdata_fig6 = FIG6_compensation_vad(hl, sdata, fs, 128, 64)
    ndata_fig6 = FIG6_compensation_vad(hl, ndata, fs, 128, 64)

    sf.write(f"{outdir}/{filenum}_src.wav", sdata, fs)
    sf.write(f"{outdir}/{filenum}_nearend.wav", ndata, fs)
    sf.write(f"{outdir}/{filenum}_nearend_fig6.wav", ndata_fig6, fs)
    target = np.stack([sdata_fig6, x_vad], axis=-1)  # T,2
    sf.write(f"{outdir}/{filenum}_target.wav", target, fs)

    # sf.write(
    #     f"{outdir}/{filenum}_{mode}_comp.wav",
    #     np.stack([audio["src"], audio["transform"], audio["target"]], axis=-1),
    #     fs,
    # )

    with open(f"{outdir}/{filenum}.json", "w+") as fp:
        json.dump(meta, fp, indent=2)

    return len(sdata) / fs


def parser():
    parser = argparse.ArgumentParser(description="--src --out /yy/zz")
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

    dnsy = Path(args.src) / "noisy"
    dsrc = Path(args.src) / "clean"
    nsy_list = list(map(str, dnsy.rglob("[!.]*.wav")))
    nfile = len(nsy_list)

    audiogram_f = "./template/audiogram.txt"
    with open(audiogram_f, "r", encoding="utf-8") as fp:
        ctx = [l.strip().replace("\t", ",").replace(" ", ",") for l in fp.readlines()]
        ctx = [list(map(float, l.split(","))) for l in ctx]
        ctx = np.array(ctx)  # N, 6

    hl_pick = np.random.choice(len(ctx), size=(nfile))
    out["aug"] = work(nsy_list, ctx[hl_pick], range(nfile), outdir=args.out, srcdir=str(dsrc))

    num = np.array(out["aug"])
    print(f"Generating {num.sum() / 3600:.2f} hours.")
