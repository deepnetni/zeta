import sys
import numpy as np
from subprocess import call
from itertools import product


cmd = f"python core/compute_metrics.py"
src = "~/trunk/dns_wdrc/"
# metrics = "stoi pesqw pesqn sisnr sdr hasqi"
metrics = "hasqi"
md_name = [
    # "baseline_fig6",
    "baseline_fig6_linear",
    "baseline_fig6_vad",
    "condConformer",
    "condConformerVAD",
    "FTCRN",
    "FTCRN_BASE_VAD",
    "FTCRN_LINEAR",
    "FTCRN_COND",
]


for m in md_name:
    print(m)
    out = f"~/model_results_trunk/FIG6/fig6_GAN/{m}/output/"
    try:
        call(
            f"{cmd} --src {src} --out {out} --metrics {metrics} --excel {m}_hasqi.xlsx",
            shell=True,
        )
    except KeyboardInterrupt:
        print("Caught Keyboard Interrupt.", file=sys.stderr)
        sys.exit()
    except OSError as err:
        print(f"Execution failed:, {err}", file=sys.stderr)
