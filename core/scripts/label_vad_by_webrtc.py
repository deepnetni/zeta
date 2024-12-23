import argparse
import os
from pathlib import Path
import json

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm
from utils.mp_decoder import mpMap, mpStarMap
from utils.audiolib import audioread, audiowrite
from utils.vad import VAD
import librosa


@mpStarMap(1)
def worker(src, dst, args):
    d, fs = audioread(src)

    if d.ndim > 1:
        d = d[:, 0]

    if fs != args.fs:
        d = librosa.resample(d, orig_sr=fs, target_sr=args.fs)
        fs = args.fs

    vad_detect = VAD(10, fs, 3)

    d_vad = vad_detect.vad_waves(d)
    N = d_vad.shape[-1]
    vad = np.stack([d[:N], d_vad], axis=-1)
    audiowrite(dst, vad, fs)


def parse():
    parser = argparse.ArgumentParser(
        description="label wav file using the webrtc method."
        "\n\nExample: python concat_speech.py --src xx --out yy",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--src",
        help="src file or directory",
        type=str,
        default="",
    )
    parser.add_argument(
        "--out",
        help="out file or directory",
        type=str,
        default="",
    )
    parser.add_argument("--fs", help="sample rate", type=int, default=16000)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse()
    print(args)
    assert args.out is not None

    if os.path.isfile(args.src):
        worker([args.src], [args.out], args=args)
    elif os.path.isdir(args.src):
        flist = list(map(str, Path(args.src).rglob("**/*.wav")))
        olist = list(map(lambda f: f.replace(args.src, args.out), flist))
        # print(flist[0], olist[0])
        worker(flist, olist, args=args)
    else:
        raise RuntimeError("")
