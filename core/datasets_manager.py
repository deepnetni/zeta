import os
import sys
from typing import Dict, Optional

import yaml

from core.utils.logger import cprint
from core.utils.trunk import *
from core.utils.trunk_v2 import *


def get_datasets(name: str, conf: Optional[Dict] = None):
    """
    name: format `dataset_fucntion` different funsions split with the `_` character.
    return: train, valid, test
    """
    data_yaml_f = os.path.join(os.path.dirname(__file__), "datasets.yaml")
    with open(data_yaml_f, "r") as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)

    conf = cfg[name.split("_")[0]] if conf is None else conf
    cprint.y(f"using {name} dataset")

    train, valid, vtest = None, None, None
    if name == "FIG6_SIG":
        train = FIG6Trunk(**conf["train"])
        valid = FIG6Trunk(**conf["valid"])
        vtest = FIG6Trunk(**conf["vtest"])
    elif name == "chime3":  # MC
        train = CHiMe3(conf["train"]["path"], subdir="train", nlen=3.0, min_len=1.0)
        valid = CHiMe3(conf["valid"]["path"], subdir="dev")
        vtest = CHiMe3(conf["vtest"]["path"], subdir="test")
    elif name == "l3das22_4ch":
        train = L3DAS22(
            dirname=conf["train"]["path"],
            patten="**/*_A.wav",
            flist="L3das22.csv",
            clean_dirname=conf["train"]["label_path"],
            min_len=1.0,
            nlen=3.0,
        )
        valid = L3DAS22(
            dirname=conf["valid"]["path"],
            patten="**/*_A.wav",
            flist="L3das22_val.csv",
            clean_dirname=conf["valid"]["label_path"],
        )
        vtest = L3DAS22(
            dirname=conf["vtest"]["path"],
            patten="**/*_A.wav",
            flist="L3das22_vtest.csv",
            clean_dirname=conf["vtest"]["label_path"],
        )
    elif name == "DNSChallenge20":
        train = NSTrunk(**conf["train"])
        valid = NSTrunk(**conf["valid"])
        vtest = NSTrunk(**conf["vtest"])

    elif name == "AECChallenge":
        train = AECTrunk(
            dirname=conf["train"]["path"],
            flist="aec-100-30.csv",
            patten="**/*mic.wav",
            keymap=("mic", "ref", "sph"),
            align=True,
        )
        valid = AECTrunk(
            dirname=conf["valid"]["path"],
            flist="aec-4-1.csv",
            patten="**/*mic.wav",
            keymap=("mic", "ref", "sph"),
            align=True,
        )
        vtest = AECTrunk(
            dirname=conf["vtest"]["path"],
            flist="aec-test-set.csv",
            patten="**/*mic.wav",
            keymap=("mic", "lpb", "sph"),
            align=True,
        )

    elif name == "AECChallenge8k":
        train = AECTrunk(
            dirname=conf["train"]["path"],
            flist="aec-100-30-l8k.csv",
            patten="**/*mic.wav",
            keymap=("mic", "ref", "sph"),
            align=True,
        )
        valid = AECTrunk(
            dirname=conf["valid"]["path"],
            flist="aec-4-1-l8k.csv",
            patten="**/*mic.wav",
            keymap=("mic", "ref", "sph"),
            align=True,
        )
        vtest = AECTrunk(
            dirname=conf["vtest"]["path"],
            flist="aec-test-set-l8k.csv",
            patten="**/*mic.wav",
            keymap=("mic", "lpb", "sph"),
            align=True,
        )
    elif name == "l3das22_all":
        raise NotImplementedError(name)
    elif name == "whamr_2chse":
        train = WHAMR_2CH(
            conf["train"]["path"],
            patten="*.wav",
            flist="whamr_tr.csv",
            clean_dirname=conf["train"]["label_path"],
            nlen=2.5,
            min_len=1.0,
        )
        valid = WHAMR_2CH(
            conf["valid"]["path"],
            patten="*.wav",
            flist="whamr_cv.csv",
            clean_dirname=conf["valid"]["label_path"],
        )
        vtest = WHAMR_2CH(
            conf["vtest"]["path"],
            patten="*.wav",
            flist="whamr_tt.csv",
            clean_dirname=conf["vtest"]["label_path"],
        )
    else:
        raise NotImplementedError(name)

    out = (train, valid, vtest)
    return out


if __name__ == "__main__":
    # tr, cv, tt = get_datasets("l3das22_4ch")
    # tr, cv, tt = get_datasets("whamr_2chse")
    # tr, cv, tt = get_datasets("AECChallenge")
    tr, cv, tt = get_datasets("FIG6_SIG")

    a, b, c = tr[0]
    print(c, isinstance(tr, torch.utils.data.Dataset))
