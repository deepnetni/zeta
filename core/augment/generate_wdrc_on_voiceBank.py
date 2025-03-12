import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict

import librosa
import numpy as np
import soundfile as sf

from utils.audiolib import audioread, audiowrite, normalize, rms
from utils.HAids.PyFIG6.pyFIG6 import FIG6_compensation_vad
from utils.mp_decoder import mpStarMap
from utils.vad import VAD


@mpStarMap(10)
def work(fpath: str, hl, outdir, nsydir, fs_tgt=16000):
    _, fname = os.path.split(fpath)
    nfpath = os.path.join(nsydir, fname)

    sdata, fs = audioread(fpath)
    if fs != fs_tgt:
        sdata = librosa.resample(sdata, orig_sr=fs, target_sr=fs_tgt)
    ndata, fs2 = audioread(nfpath)
    if fs2 != fs_tgt:
        ndata = librosa.resample(ndata, orig_sr=fs2, target_sr=fs_tgt)

    # vad_detect = VAD(10, fs, level=2)
    # vad_detect.reset()
    # x_vad = np.ones_like(sdata)
    # x_vad = vad_detect.vad_waves(sdata)  # T,
    # x_vad[: len(d_vad)] = d_vad

    pow_s = rms(sdata, True)
    pow = pow_s
    if pow_s < -30:
        pow = np.random.uniform(-25, -30)
        sdata, scalar = normalize(sdata, pow)
        ndata = ndata * scalar

    sdata_fig6, x_vad, m_fig = FIG6_compensation_vad(hl, sdata, fs, 128, 64, ret_vad=True)
    # spl = m_fig["spl_in"]  # B,T,C
    ndata_fig6 = FIG6_compensation_vad(hl, ndata, fs, 128, 64)

    target = np.stack([sdata_fig6, x_vad], axis=-1)  # T,2

    fname = fname.removesuffix(".wav")

    sf.write(f"{outdir}/{fname}_nearend.wav", ndata, fs_tgt)
    sf.write(f"{outdir}/{fname}_src.wav", sdata, fs_tgt)
    sf.write(f"{outdir}/{fname}_target.wav", target, fs_tgt)
    sf.write(f"{outdir}/{fname}_nearend_fig6.wav", ndata_fig6, fs_tgt)

    with open(f"{outdir}/{fname}.json", "w+") as fp:
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
    parser.add_argument(
        "--nsy",
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

    dsrc = Path(args.src)
    src_list = list(map(str, dsrc.rglob("[!.]*.wav")))
    nfile = len(src_list)
    # src_list = []

    audiogram_f = "./template/audiogram.txt"
    with open(audiogram_f, "r", encoding="utf-8") as fp:
        ctx = [l.strip().replace("\t", ",").replace(" ", ",") for l in fp.readlines()]
        ctx = [list(map(float, l.split(","))) for l in ctx]
        ctx = np.array(ctx)  # N, 6

    hl_pick = np.random.choice(len(ctx), size=(nfile))
    out["aug"] = work(src_list, ctx[hl_pick], outdir=args.out, nsydir=args.nsy)

    num = np.array(out["aug"])
    print(f"Generating {num.sum() / 3600:.2f} hours.")
