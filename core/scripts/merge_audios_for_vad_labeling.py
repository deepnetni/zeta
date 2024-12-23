import argparse
import os
from pathlib import Path
import json

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm
from utils.mp_decoder import mpMap
from utils.audiolib import audioread, audiowrite


@mpMap(10)
def worker(filenum: int, flist, args, last_n):
    if filenum != last_n:
        st = filenum * args.num
        ed = st + args.num
        files = flist[st:ed]
    else:
        st = filenum * args.num
        files = flist[st:None]

    data_l = []
    meta = {}
    N = 0
    for f in files:
        data, fs = audioread(f)  # T,
        data_l.append(data)  # [T,T,T]
        meta.update({f.name: {"st": N, "ed": N + len(data)}})
        N += len(data)

    data_c = np.concatenate(data_l, axis=-1)
    data = np.stack([data_c, np.ones_like(data_c) * 0.9], axis=-1)

    audiowrite(f"{args.out}/{filenum}.wav", data, args.fs)
    with open(f"{args.out}/{filenum}.json", "w+") as fp:
        json.dump(meta, fp, indent=2)


def parse():
    parser = argparse.ArgumentParser(
        description="Merge `num` audio files into a single file and add a channel filled with `0.9` for subsequent VAD labeling."
        "\n\nExample: python concat_speech.py --src xx --out yy --num 60",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--src",
        help="src file or directory",
        type=str,
        default="/home/deepni/disk/DNS-Challenge/datasets/clean",
    )
    parser.add_argument(
        "--out",
        help="out file or directory",
        type=str,
        default="/home/deepni/disk/DNS-Challenge-vad/clean",
    )
    parser.add_argument("--num", help="files to concat", type=int, default=60)
    parser.add_argument("--fs", help="sample rate", type=int, default=16000)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse()
    assert args.out is not None

    flists = list(Path(args.src).rglob("**/*.wav"))
    N = np.ceil(len(flists) / args.num).astype(int)

    worker(range(N), flists, args, N)
