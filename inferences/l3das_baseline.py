import os
import torch
import numpy as np
from rebuild.L3DASBaselines.MMUB import MIMO_UNet_Beamforming
from rebuild.L3DASBaselines.inference import load_model, parse_args, enhance_sound
from utils.trunk import L3DAS22
from utils.audiolib import audioread, audiowrite
from tqdm import tqdm


if __name__ == "__main__":
    args = parse_args()
    args.use_cuda = eval(args.use_cuda)

    model = MIMO_UNet_Beamforming(fft_size=512, hop_size=128, input_channel=4)
    device = "cuda:0"
    state = load_model(model, None, args.model_path, args.use_cuda)

    net = model.to(device)

    if args.vtest:
        dset = L3DAS22(
            dirname="/home/deepni/datasets/l3das/L3DAS22_Task1_test/data",
            patten="**/*_A.wav",
            flist="L3das22_vtest.csv",
            clean_dirname="/home/deepni/datasets/l3das/L3DAS22_Task1_test/labels",
        )
    elif args.valid:
        dset = L3DAS22(
            dirname="/home/deepni/datasets/l3das/L3DAS22_Task1_dev/data",
            patten="**/*_A.wav",
            flist="L3das22_val.csv",
            clean_dirname="/home/deepni/datasets/l3das/L3DAS22_Task1_dev/labels",
        )
    else:
        raise RuntimeError("not supported.")

    for d, fname in tqdm(dset):
        d = d.cuda()
        x = d.permute(0, 2, 1)
        with torch.no_grad():
            out = enhance_sound(x, model, device, 76672, 0.5)
        out = np.squeeze(out)

        fout = os.path.join(args.out, fname)
        outd = os.path.dirname(fout)
        os.makedirs(outd) if not os.path.exists(outd) else None
        audiowrite(fout, out, sample_rate=16000)
