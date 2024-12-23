import os
from utils.logger import CPrint

cp = CPrint()

epoch = 70
vt = "valid"
# vt = "vtest"


cmd = f"python train_mcse_l3das.py --{vt} --pred"
cmd_mcnet = f"python train_mcnet_l3das.py --{vt} --pred"
cmd_deftan = f"python train_deftan_l3das.py --{vt} --pred"
cmd_seu = f"python train_seuspeech_l3das.py --{vt} --pred"
cmd_beam_unet = (
    f"python l3das_baseline.py --model_path rebuild/L3DASBaselines/baseline_task1_checkpoint --{vt}"
)

cmd_valid = "python scripts/compute_metrics.py"


src_test = "/home/deepni/datasets/l3das/L3DAS22_Task1_test/labels"
src_valid = "/home/deepni/datasets/l3das/L3DAS22_Task1_dev/labels"
src_dir = src_test if vt == "vtest" else src_valid

ckpath = "trained_mcse_l3das_win"

ckpt_file = f"epoch_0{epoch}.pth" if epoch == 100 else f"epoch_00{epoch}.pth"


elements = [
    # ("mcse_3en_wdefu_wfafu_c72_B2", f"{ckpath}/pred_mcse_{vt}_{epoch}"),
    # ("mcse_3en_wdefu_wfafu_cau_c72_B2", f"{ckpath}/pred_mcse_cau_{vt}_{epoch}"),
    # ("mcse_baseline", f"{ckpath}/pred_baseline_{vt}_{epoch}"),
    #
    # ("mcse_baseline+3en", f"{ckpath}/pred_baseline_3en_{vt}_{epoch}"),
    # ("mcse_baseline+UC+ED", f"{ckpath}/pred_baseline_UC+ED_{vt}_{epoch}"),
    # ("mcse_baseline+3aff", f"{ckpath}/pred_baseline_3aff_{vt}_{epoch}"),
    # ("mcse_baseline+3att", f"{ckpath}/pred_baseline_3att_{vt}_{epoch}"),
    # ("mcse_baseline+3en+ED", f"{ckpath}/pred_baseline+3en+ED_{vt}_{epoch}"),
    # ("mcse_uc_wdefu_wfafu_c72_B2", f"{ckpath}/pred_UC+all_{vt}_{epoch}"),
    # ("mcse_3en_all_aff_wdefu_wfafu_c72_B2", f"{ckpath}/pred_3AFF+all_{vt}_{epoch}"),
    ("mcse_3en_all_att_wdefu_wfafu_c72_B2", f"{ckpath}/pred_3ATT+all_{vt}_{epoch}"),
    # ("mcse_3en_wdefu_wfafu_inv_c72_B2", f"{ckpath}/pred_inv_all_{vt}_{epoch}"),
    #
    # ("mcnet_raw", f"{ckpath}/pred_mcnet_{vt}_{epoch}"),
    # ("deftan", f"{ckpath}/pred_deftan_{vt}_{epoch}"),
    # ("seu_speech", f"{ckpath}/pred_seuspeech_{vt}_{epoch}"),
    # ("beamforming_unet", f"{ckpath}/pred_beamforming_unet_{vt}"),
]

for name, outdir in elements:
    # name = "mcse_3en_wdefu_wfafu_c72_B2"
    # outdir = f"{ckpath}/pred_mcse_vtest_{epoch}"
    cp.r(name)

    if name == "mcnet_raw":
        os.system(
            f"{cmd_mcnet} --ckpt {ckpath}/{name}/checkpoints/{ckpt_file} --out {outdir}"
        ) if not os.path.exists(outdir) else None
    elif name == "deftan":
        os.system(
            f"{cmd_deftan} --ckpt {ckpath}/{name}/checkpoints/{ckpt_file} --out {outdir}"
        ) if not os.path.exists(outdir) else None
    elif name == "seu_speech":
        os.system(
            f"{cmd_seu} --ckpt {ckpath}/{name}/checkpoints/{ckpt_file} --out {outdir}"
        ) if not os.path.exists(outdir) else None
    elif name == "beamforming_unet":
        os.system(f"{cmd_beam_unet} --out {outdir}") if not os.path.exists(outdir) else None
    else:
        os.system(
            f"{cmd} --name {name} --ckpt {ckpath}/{name}/checkpoints/{ckpt_file} --out {outdir}"
        ) if not os.path.exists(outdir) else None
    # os.system(f"{cmd_valid} --sub --out {outdir}  --src {src_test}  --metrics asr pesqn pesqw stoi")
    cp.r("compute metrics")
    os.system(
        # f"{cmd_valid} --sub --out {outdir}  --excel {ckpath}/{name}_{vt}_{epoch}.xlsx --src {src_dir}  --metrics asr pesqn pesqw l3das"
        f"{cmd_valid} --sub --out {outdir}  --excel {ckpath}/{name}_{vt}_{epoch}.xlsx --src {src_dir}  --metrics asr pesqn pesqw stoi"
    )
