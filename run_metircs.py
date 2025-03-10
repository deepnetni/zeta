import os
import sys
import numpy as np
from subprocess import call
from itertools import product
from core.utils.logger import cprint

#  python core/compute_metrics.py --src ~/datasets/dns_wdrc/test_noise92 --out ~/datasets/dns_wdrc/test_noise92/ --pattern ".*nearend_fig6\.wav" --map nearend_fig6.wav target.wav --metrics hasqi pesqw pesqn sdr sisnr stoi

cmd = f"python core/compute_metrics.py"
# src = "~/trunk/dns_wdrc"
# src = "~/datasets/dns_wdrc"
src = "~/datasets/dns_wdrc/test_noise92"
# src = "~/datasets/dns_wdrc/dns_testset_noverb"
_, dname = os.path.split(src)

metrics = "stoi pesqw pesqn sisnr sdr hasqi"
# metrics = "l3das"
md_name = [
    # "baseline_fig6",
    # "baseline_fig6_linear",
    # "baseline_fig6_vad",
    # "condConformer",
    # "IterCondConformer",
    # "condConformerVAD_mc36"
    "condConformerVAD8_mc36"
    # "condConformerVAD",
    # "condConformerVAD8_mc48"
    # "condConformerVAD_mc60"
    # "condConformerVAD8_mc60"
    # "FTCRN",
    # "FTCRN_BASE_VAD",
    # "FTCRN_LINEAR",
    # "FTCRN_COND",
    # "FTCRN_VAD",
    # "FTCRN_COND_Iter",
]


for m in md_name:
    cprint.r(m)
    # out = f"~/model_results_trunk/FIG6/fig6_GAN/{m}/output"
    out = f"~/model_results_trunk/FIG6/fig6_GAN/{m}/output/{dname}"
    # out = f"~/model_results_trunk/FIG6/fig6_GAN/{m}/output/dns_testset_noverb"
    try:
        call(
            f"{cmd} --src {src} --out {out} --metrics {metrics} --excel {m}_{dname}.xlsx",
            shell=True,
        )
    except KeyboardInterrupt:
        print("Caught Keyboard Interrupt.", file=sys.stderr)
        sys.exit()
    except OSError as err:
        print(f"Execution failed:, {err}", file=sys.stderr)
