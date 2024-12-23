import argparse
import torch
import numpy as np


def enhance_sound(predictors, model, device, length, overlap):
    """
    Compute enhanced waveform using a trained model,
    applying a sliding crossfading window
    """

    def pad(x, d):
        # zeropad to desired length
        pad = torch.zeros((x.shape[0], x.shape[1], d))
        pad[:, :, : x.shape[-1]] = x
        return pad

    def xfade(x1, x2, fade_samps, exp=1.0):
        # simple linear/exponential crossfade and concatenation
        out = []
        fadein = np.arange(fade_samps) / fade_samps
        fadeout = np.arange(fade_samps, 0, -1) / fade_samps
        fade_in = fadein * exp
        fade_out = fadeout * exp
        x1[:, :, -fade_samps:] = x1[:, :, -fade_samps:] * fadeout
        x2[:, :, :fade_samps] = x2[:, :, :fade_samps] * fadein
        left = x1[:, :, :-fade_samps]
        center = x1[:, :, -fade_samps:] + x2[:, :, :fade_samps]
        end = x2[:, :, fade_samps:]
        return np.concatenate((left, center, end), axis=-1)

    overlap_len = int(length * overlap)  # in samples
    total_len = predictors.shape[-1]
    starts = np.arange(0, total_len, overlap_len)  # points to cut
    # iterate the sliding frames
    for i in range(len(starts)):
        start = starts[i]
        end = starts[i] + length
        if end < total_len:
            cut_x = predictors[:, :, start:end]
        else:
            # zeropad the last frame
            end = total_len
            cut_x = pad(predictors[:, :, start:end], length)

        # compute model's output
        cut_x = cut_x.to(device)
        predicted_x = model(cut_x, device)
        predicted_x = predicted_x.cpu().numpy()

        # reconstruct sound crossfading segments
        if i == 0:
            recon = predicted_x
        else:
            recon = xfade(recon, predicted_x, overlap_len)

    # undo final pad
    recon = recon[:, :, :total_len]

    return recon


def load_model(model, optimizer, path, cuda):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module  # load state dict of wrapped module
    if cuda:
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location="cpu")
    try:
        # model.load_state_dict(checkpoint['model_state_dict'])
        model.load_state_dict(
            torch.load(
                checkpoint["model_state_dict"], map_location=lambda storage, location: storage
            ),
            strict=False,
        )
    except:
        # work-around for loading checkpoints where DataParallel was saved instead of inner module
        from collections import OrderedDict

        model_state_dict_fixed = OrderedDict()
        prefix = "module."
        for k, v in checkpoint["model_state_dict"].items():
            if k.startswith(prefix):
                k = k[len(prefix) :]
            model_state_dict_fixed[k] = v
        model.load_state_dict(model_state_dict_fixed)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "state" in checkpoint:
        state = checkpoint["state"]
    else:
        # older checkpoints only store step, rest of state won't be there
        state = {"step": checkpoint["step"]}
    return state


def enh_file(src):
    """
    src, l3das file;
    """

    # input shape should be B,C,T
    pass


def parse_args():
    parser = argparse.ArgumentParser()
    # i/o parameters
    parser.add_argument("--model_path", type=str, default="baseline_task1_checkpoint")
    parser.add_argument("--results_path", type=str, default="RESULTS/Task1/metrics")
    parser.add_argument("--save_sounds_freq", type=int, default=None)
    # dataset parameters
    parser.add_argument(
        "--predictors_path", type=str, default="DATASETS/processed/task1_predictors_test_uncut.pkl"
    )
    parser.add_argument(
        "--target_path", type=str, default="DATASETS/processed/task1_target_test_uncut.pkl"
    )
    parser.add_argument("--sr", type=int, default=16000)
    # reconstruction parameters
    parser.add_argument("--segment_length", type=int, default=76672)
    parser.add_argument("--segment_overlap", type=float, default=0.5)
    # model parameters
    parser.add_argument(
        "--architecture", type=str, default="MIMO_UNet_Beamforming", help="model name"
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--use_cuda", type=str, default="True")
    parser.add_argument("--enc_dim", type=int, default=64)
    parser.add_argument("--feature_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--segment_size", type=int, default=24)
    parser.add_argument("--nspk", type=int, default=1)
    parser.add_argument("--win_len", type=int, default=16)
    parser.add_argument("--context_len", type=int, default=16)
    parser.add_argument("--fft_size", type=int, default=512)
    parser.add_argument("--hop_size", type=int, default=128)
    parser.add_argument("--input_channel", type=int, default=4)

    parser.add_argument("--vtest", help="input directory", action="store_true")
    parser.add_argument("--valid", help="input directory", action="store_true")
    parser.add_argument("--out", help="predicting output directory", type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    from .MMUB import MIMO_UNet_Beamforming
    from utils.trunk import L3DAS22
    from utils.audiolib import audioread, audiowrite
    from tqdm import tqdm

    args = parse_args()
    args.use_cuda = eval(args.use_cuda)

    model = MIMO_UNet_Beamforming(fft_size=512, hop_size=128, input_channel=4)
    device = "cuda:0"
    state = load_model(model, None, args.model_path, args.use_cuda)

    model = model.to(device)

    x = torch.randn(1, 4, 160000).cuda()
    with torch.no_grad():
        outputs = enhance_sound(x, model, device, 76672, 0.5)
    outputs = np.squeeze(outputs)
    print(outputs.shape)
