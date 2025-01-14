import argparse
import json

# import multiprocessing as mp
import os
import shutil

# from itertools import repeat
from typing import Dict

import numpy as np
import soundfile as sf

# import yaml
# from tqdm import tqdm

from synthesizer.synthesizer_wdrc import Synthesizer
from utils.mp_decoder import mpMap


@mpMap(10)
def work(filenum: int, yaml, outdir, fs):
    generator = Synthesizer(yaml)
    audio = generator.generate()

    meta: Dict
    meta = audio["info"]
    mode = meta.get("apply_gain", {})["mode"]

    sf.write(f"{outdir}/{filenum}_{mode}_src.wav", audio["src"], fs)
    sf.write(f"{outdir}/{filenum}_{mode}_transform.wav", audio["transform"], fs)
    sf.write(f"{outdir}/{filenum}_{mode}_nearend.wav", audio["nearend"], fs)
    sf.write(f"{outdir}/{filenum}_{mode}_nearend_fig6.wav", audio["nearend_fig6"], fs)
    target = audio["target"] - audio["target"].mean()
    target = np.stack([target, audio["vad"]])  # T,2
    sf.write(f"{outdir}/{filenum}_{mode}_target.wav", target, fs)

    # sf.write(
    #     f"{outdir}/{filenum}_{mode}_comp.wav",
    #     np.stack([audio["src"], audio["transform"], audio["target"]], axis=-1),
    #     fs,
    # )
    with open(f"{outdir}/{filenum}_{mode}.json", "w+") as fp:
        json.dump(meta, fp, indent=2)

    return 1


def parser():
    parser = argparse.ArgumentParser(
        description="python augment_sig.py --yaml xx.yaml --outdir /yy/zz --time 50"
    )
    parser.add_argument(
        "--time", help="augment audio length based on hours", type=float, default=100.0
    )
    parser.add_argument(
        "--yaml",
        help="path to the configure yaml file",
        type=str,
        default="./template/synthesizer_config_wdrc.yaml",
    )
    parser.add_argument("--outdir", help="out dirname")

    return parser.parse_args()


if __name__ == "__main__":
    """
    Example:
        >>> python augment_wdrc.py --yaml xx.yaml --outdir /home/deepni/datasets/dns_p09_50h/ --time 50
    """
    args = parser()

    # with open(args.yaml) as f:
    #     cfg = yaml.load(f, Loader=yaml.FullLoader)

    worker = Synthesizer(args.yaml)
    fs = worker.cfg["onlinesynth_sampling_rate"]
    nlen = worker.cfg["onlinesynth_duration"]
    nfile = int(args.time * 3600 // (nlen))

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    else:
        shutil.rmtree(args.outdir)
        os.makedirs(args.outdir)

    out = {}
    # mp.freeze_support()
    # p = mp.Pool(processes=30)
    # worker_l = [Synthesizer(args.yaml) for _ in range(30)] # not work, still on the main thread
    # out["aug"] = list(
    #     p.starmap(
    #         work,
    #         tqdm(
    #             zip(repeat(args.yaml), range(nfile), repeat(args.outdir), repeat(fs)),
    #             ncols=80,
    #             total=nfile,
    #         ),
    #     )
    # )

    out["aug"] = work(range(nfile), args.yaml, args.outdir, fs)
    num = np.array(out["aug"])
    print(f"Generating {num.sum() * nlen / 3600:.2f} hours.")
