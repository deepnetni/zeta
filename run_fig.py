import sys
import numpy as np
from subprocess import call
from itertools import product


cmd = f"python fit_fig6_mgan.py --pred"
item = [
    "--valid",
    "--vtest",
    "--dset FIG6_noise92",
    # "--dset FIG6_github",
]

md_name = [
    # "baseline_fig6",
    # "baseline_fig6_linear",
    # "baseline_fig6_vad --vad",
    # "condConformer",
    "IterCondConformer",
    # "condConformerVAD --vad",
    # "FTCRN",
    # "FTCRN_BASE_VAD --vad",
    # "FTCRN_LINEAR",
    # "FTCRN_COND",
    # "FTCRN_VAD --vad",
    # "FTCRN_COND_Iter",
]


for m, i in product(md_name, item):
    try:
        call(f"{cmd} --name {m} {i}", shell=True)
    except KeyboardInterrupt:
        print("Caught Keyboard Interrupt.", file=sys.stderr)
        sys.exit()
    except OSError as err:
        print(f"Execution failed:, {err}", file=sys.stderr)
