import os
import sys
from subprocess import call
from utils.logger import cprint

# fmt: off
try:
    call("python train_mcse_chime.py --conf config/yconf_mcse_chime.yaml --name mcse_conformer", shell=True)

    call("python train_mcse_chime.py --conf config/yconf_mcse_chime.yaml --name mcse_3en_wdefu_wfafu_cau_c72_B2", shell=True)
    call("python train_spatial_chime.py")
    call("python train_mcse_chime.py --conf config/yconf_mcse_chime.yaml --name mcse_baseline+TC+ED", shell=True)

    call("python train_mcse_chime.py --conf config/yconf_mcse_chime.yaml --name mcse_baseline", shell=True)
    call("python train_mcse_chime.py --conf config/yconf_mcse_chime.yaml --name mcse_uc_wdefu_wfafu_c72_B2", shell=True)
    call("python train_mcse_chime.py --conf config/yconf_mcse_chime.yaml --name mcse_baseline+UC+ED", shell=True)
    call("python train_mcse_chime.py --conf config/yconf_mcse_chime.yaml --name mcse_baseline+3en+ED", shell=True)
    call("python train_mcse_chime.py --conf config/yconf_mcse_chime.yaml --name mcse_3en_all_aff_wdefu_wfafu_c72_B2", shell=True)
    call("python train_mcse_chime.py --conf config/yconf_mcse_chime.yaml --name mcse_3en_all_att_wdefu_wfafu_c72_B2", shell=True)
    # call("python train_mcse_chime.py --conf config/yconf_mcse_chime.yaml --name mcse_baseline+3en+ED+FA", shell=True)
    call("python train_mcse_chime.py --conf config/yconf_mcse_chime.yaml --name mcse_baseline+3en", shell=True)
    call("python train_mcse_chime.py --conf config/yconf_mcse_chime.yaml --name mcse_baseline+3aff", shell=True)
    call("python train_mcse_chime.py --conf config/yconf_mcse_chime.yaml --name mcse_baseline+3att", shell=True)

    # call("python train_mcse_chime.py --conf config/yconf_mcse_chime.yaml --name mcse_3en_wdefu_wfafu_c72_B2", shell=True)
    call("python train_mcse_chime.py --conf config/yconf_mcse_chime.yaml --name mcse_3en_wdefu_wfafu_inv_c72_B2", shell=True)
except KeyboardInterrupt:
    cprint.r("Caught Keyboard Interrupt.", file=sys.stderr)
    sys.exit()
except OSError as err:
    cprint.r(f"Execution failed:, {err}", file=sys.stderr)
