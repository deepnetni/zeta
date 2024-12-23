import sys
import os
import numpy as np
from utils.logger import cprint
from subprocess import call


# fmt: off
# cmd_l = [
#     "python", "train_mcse_spdns.py",
#     "--conf", "config/yconf_mcse_sp.yaml",
#     "--name", "mcse_3en_wdefu_wfafu_c72_B2",
#     "--epoch", "10",
# ]
# ret = call(cmd_l)

for e in [1,2,3,4, *np.arange(5, 100, 5)]:
    try:
        call(f"python train_mcse_whamr.py --conf config/yconf_mcse_whamr.yaml --name mcse_3en_wdefu_wfafu_c72_B2 --epoch {e}", shell=True)
        call(f"python train_mcnet_whamr.py --conf config/yconf_mcnet_whamr.yaml --epoch {e}", shell=True)
        call(f"python train_deftan_whamr.py --conf config/yconf_deftan_whamr.yaml --epoch {e}", shell=True)
    except KeyboardInterrupt:
        cprint.r("Caught Keyboard Interrupt.", file=sys.stderr)
        sys.exit()
    except OSError as err:
        cprint.r(f"Execution failed:, {err}", file=sys.stderr)
