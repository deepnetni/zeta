import argparse
import json
import multiprocessing as mp
import os
import re
import sys
from itertools import repeat
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio as si_sdr
from torchmetrics.functional.audio import signal_distortion_ratio as SDR
from tqdm import tqdm

from utils.audiolib import audioread
from utils.composite_metrics import eval_composite
from utils.metrics import *
from utils.mp_decoder import mpStarMap
from scripts.l3das.metrics import task1_metric
from utils.logger import cprint


def parse():
    parser = argparse.ArgumentParser(
        description="compute the metrics score with input file or directory."
        "\n\nExample: python compute_metrics.py --src xx --out yy --sisnr ",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--src", help="src file or directory", type=str, default=None)
    parser.add_argument("--src_spec", help="src file or directory", type=str, default=None)
    parser.add_argument("--same", help="", default=False)
    parser.add_argument("--pattern", help="noisy files", default=r"^(?!\.).*_mic\.wav$")

    parser.add_argument("--out", help="dst file or directory", type=str)
    parser.add_argument("--fs", help="dst file or directory", type=int, default=16000)

    # metrics
    parser.add_argument("--sisnr", help="si-snr", action="store_true")
    parser.add_argument("--snr", help="snr", action="store_true")
    parser.add_argument("--sdr", help="sdr", action="store_true")
    parser.add_argument("--pesqw", help="wide band pesq", action="store_true")
    parser.add_argument("--pesqn", help="narrow band pesq", action="store_true")
    parser.add_argument("--stoi", help="stoi", action="store_true")
    parser.add_argument("--asr", help="csig, cbak, covl", action="store_true")
    parser.add_argument("--l3das", help="l3das", action="store_true")
    parser.add_argument("--metrics", help="compute multi-metrics", nargs="+")

    # output
    parser.add_argument("--excel", help="excel name", default=None)

    parser.add_argument(
        "--map", help="pattern between mic and sph, --map mic.wav sph.wav", nargs="+"
    )

    # the output folder contain multi-types
    args = parser.parse_args()

    if args.src is not None:
        args.src = os.path.abspath(args.src)
    elif args.src_spec is not None:
        args.src_spec = os.path.abspath(args.src_spec)
        args.same = True
    else:
        raise RuntimeError(f"src or src_spec must configure one.")

    args.out = os.path.abspath(args.out)

    return args


def save_excel(fname: str, data: dict):
    """
    data:{'caf': {'pesq':np.array, 'stoi':np.array, ...}, 'bus':{...}, ...}
    """

    with pd.ExcelWriter(fname) as writer:
        for sheet_name, metrics in data.items():
            df = pd.DataFrame(metrics)
            # df.to_excel(writer, sheet_name=m, index=False, header=None)
            df.to_excel(writer, sheet_name=sheet_name, index=False)


def to_excel(data: dict, excel_f: str):
    """
    data:{'caf': [{'pesq':v, 'stoi':v,...}, {...}], 'bus':[], 'ped':[]}
    """
    if excel_f is None:
        return

    tag = excel_f.split("/")[-1].split(".")[0]

    save = {}
    for env, metrics in data.items():
        # env: caf, bus, ...
        # metrics: [{'pesq':v, ...}, {...}]

        dic = {}  # {'pesq': [...], 'stoi': [...]}
        for d in metrics:
            for k, v in d.items():
                dic.setdefault(k, []).append(v)

        for k, v in dic.items():
            # k: pesq, stoi, ...
            # v: list[...]
            save.setdefault(k, {"type": [tag] * len(v)}).update({env: v})
            save.setdefault(f"all_{k}", []).extend(v)

    # save: {'pesq':{'caf':[..], 'bus':[..]}, "stoi":{..}}
    with pd.ExcelWriter(excel_f) as writer:
        for m, d in save.items():
            df = pd.DataFrame(d)
            # df.to_excel(writer, sheet_name=m, index=False, header=None)
            df.to_excel(writer, sheet_name=m, index=False)


@mpStarMap(5)
def compute_score(spath, epath, args):
    """Wrapped by the multiprocessing decoder.
    Input: spath, epath is a list.
    return: [{'m1':v,'m2'..}, {...}]


    Example:
        compute_score([files], [files], args=args)
    """
    # print(spath, epath)
    edata, _ = audioread(epath)
    sdata, _ = audioread(spath)
    # print(edata.shape) if edata.ndim > 1 else None
    # edata = edata.mean(-1) if edata.ndim > 1 else edata
    edata = edata[:, 0] if edata.ndim > 1 else edata
    sdata = sdata[: len(edata)]
    # edata = np.concatenate([np.zeros(15), sdata[: len(edata) - 15]])

    metrics = {}

    if args.sisnr or "sisnr" in args.metrics:
        metrics["sisnr"] = compute_si_snr(sdata, edata)

    if args.snr or "snr" in args.metrics:
        metrics["snr"] = compute_snr(sdata, edata)

    if args.pesqw or "pesqw" in args.metrics:
        metrics["pesqw"] = compute_pesq(sdata, edata, fs=args.fs, mode="wb")
        # print(spath, epath, metrics)

    if args.pesqn or "pesqn" in args.metrics:
        metrics["pesqn"] = compute_pesq(sdata, edata, fs=args.fs, mode="nb")

    if args.stoi or "stoi" in args.metrics:
        metrics["stoi"] = compute_stoi(sdata, edata, fs=args.fs)

    if args.sdr or "sdr" in args.metrics:
        edata = torch.from_numpy(edata)
        sdata = torch.from_numpy(sdata)
        sdr = SDR(preds=edata, target=sdata)
        # sisdr = si_sdr(preds=edata, target=sdata)
        metrics["sdr"] = sdr.cpu().numpy()

    if args.asr or "asr" in args.metrics:
        metrics.update(eval_composite(sdata, edata, sample_rate=args.fs))

    if args.l3das or "l3das" in args.metrics:
        score, wer, stoi_sc = task1_metric(sdata, edata)
        metrics.update({"l3das_score": score, "l3das_wer": wer * 100, "l3das_stoi": stoi_sc})

    return metrics


def compute_box(result: List) -> Dict:
    """
    result: [{'v1':[v, ...],'v2': [...]}, {...}]
    return: {'m1':v, 'm2':v2,...}
    """
    if len(result) == 0:
        raise RuntimeWarning("empty result.")

    if isinstance(result[0], float):
        sc = np.array(result)
        sc = [
            np.percentile(sc, 25).round(4),
            np.percentile(sc, 50).round(4),
            np.percentile(sc, 75).round(4),
        ]
    elif isinstance(result[0], dict):
        dic = {}
        for d in result:
            for k, v in d.items():
                dic.setdefault(k, []).append(v)

        sc = {}
        for k, v in dic.items():
            d = np.array(v)
            d = [
                np.percentile(d, 25).round(4),
                np.percentile(d, 50).round(4),
                np.percentile(d, 75).round(4),
            ]
            sc[k] = d
    else:
        raise RuntimeError("not supported.")

    return sc


def pack_metrics(result: List) -> Dict:
    """
    result: [{'m1':v,'m2'..}, {...}]
    return: {'m1':np.array, 'm2':...}
    """
    if len(result) == 0:
        raise RuntimeWarning("empty result.")

    dic = {}
    for d in result:
        for k, v in d.items():
            dic.setdefault(k, []).append(v)

    return {k: np.array(v) for k, v in dic.items()}


if __name__ == "__main__":
    """
    python compute_metrics.py --out ../trained_mcse_spdns/pred_mcse_50/test --src ~/datasets/spatialReverbNoise/test --map target.wav mic.wav --pesq

    Note: the `out` and `src` given the top different path, which means the ungiven path is the same.
        e,g,. `--out /a/b --src /c/d` means /a/b/e/f/..wav corresponds to /c/d/e/f/..wav
    """
    args = parse()

    score = {}
    if os.path.isfile(args.out):
        score_l = compute_score([args.src], [args.out], args=args)
        score = pack_metrics(score_l)  # {'m1': np.array, 'm2':..}

        for k, v in score.items():
            print(k)
            print(v.mean().round(3))

    elif os.path.isdir(args.out):
        # out/sub1->src, src/sub1->src
        items = {}

        for root, dirs, files in os.walk(args.out):
            files = list(filter(lambda f: re.search(args.pattern, f), files))
            # files = list(filter(lambda f: f.endswith(".wav") and not f.startswith("."), files))
            if root.startswith(".") or len(files) == 0:
                continue

            enh_l = list(map(lambda f: os.path.join(root, f), files))
            if args.same is False:
                src_l = [f.replace(args.out, args.src) for f in enh_l]
            else:
                src_l = [f.replace(root, args.src_spec) for f in enh_l]
            subd = str(Path(root).relative_to(args.out))

            # print("@", enh_l[0] if len(files) != 0 else [], src_l[0])

            if args.map is not None:
                src_l = [f.replace(*args.map) for f in src_l]

            score_l = compute_score(src_l, enh_l, args=args)
            score = pack_metrics(score_l)
            items[subd] = score  # {"subd":{"m1":np.array, "m2":np.ndarry, ...}, ...}

        items = dict(sorted(items.items()))
        save_excel(args.excel, items) if args.excel is not None and len(items) != 0 else None

        for subd, scores in items.items():
            cprint.b(f"\n{subd} ---------\n")
            result = []
            col_w = []
            for k, v in scores.items():
                print(f"{ subd }, {k}, {len(v)}")
                result.append([k, v.mean().round(3), v.std().round(3)])

            col_w = [max(len(str(d)) for d in col) for col in zip(*result)]
            for ele in result:
                print("| " + " | ".join(str(d).ljust(w) for d, w in zip(ele, col_w)) + " |")
    else:
        raise RuntimeError("")
