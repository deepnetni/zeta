import os
import ast
import numpy as np
from librosa import istft, stft
from scipy.signal import get_window
from utils.audiolib import audioread

from utils.HAids.PyFIG6.pyFIG6 import compute_subbands_SPL
from matplotlib import pyplot as plt


def cal_spl(inp: np.ndarray, nframe=128, nhop=64, fs=16000):
    """
    inp: B, T,
    """
    SPL_offset = 94.9133059 * np.ones(6)
    nbin = nhop + 1
    freso = np.linspace(0, fs // 2, nbin)
    win = get_window("hann", nframe, fftbins=True)
    win = np.sqrt(win)
    ChannelNum_ft = np.array([0, 250, 625, 1375, 2500, 3500, 8001])
    xk = stft(
        inp,  # B,T,
        win_length=nframe,
        n_fft=nframe,
        hop_length=nhop,
        window=win,
        center=False,
    )  # output shape B,F,T
    # xkk = np.stack([xk.real, xk.imag], axis=1)
    xk = xk.transpose(0, 2, 1)  # f,t -> t,f
    spl_in = compute_subbands_SPL(xk, freso, ChannelNum_ft, SPL_offset)
    return spl_in.mean(axis=-1)  # B,T


def draw_spl(src_f, tgt_f, enh_f, enh_f_):
    d1, _ = audioread(src_f)
    d2, _ = audioread(tgt_f)
    d2 = d2[:, 0]
    d3, _ = audioread(enh_f)
    d4, _ = audioread(enh_f_)
    N = min(len(d1), len(d2), len(d3))
    d1, d2, d3, d4 = d1[:N], d2[:N], d3[:N], d4[:N]
    d = np.stack([d1, d2, d3, d4], axis=0)
    spl = cal_spl(d)  # B,T

    for d, param in zip(
        spl,
        [
            {"label": "src", "linestyle": ":", "color": "k", "alpha": 0.5},
            {"label": "tgt", "linestyle": "-", "color": "k"},
            {"label": "enh", "linestyle": "--"},
            {"label": "enh", "linestyle": "--", "color": "r"},
        ],
    ):
        plt.plot(d, **param)
    plt.legend()
    plt.savefig(f"splin.svg", bbox_inches="tight")


if __name__ == "__main__":
    src_f = f"{os.path.expanduser('~')}/github/zeta/show/7_shrink_transform.wav"
    tgt_f = f"{os.path.expanduser('~')}/github/zeta/show/7_shrink_target.wav"
    enh_f = f"{os.path.expanduser('~')}/model_results_trunk/FIG6/fig6_GAN/FTCRN/output/test_noise92/0.0/7_shrink_target.wav"
    # enh_f_ = f"{os.path.expanduser('~')}/model_results_trunk/FIG6/fig6_GAN/FTCRN_BASE_VAD/output/test_noise92/0.0/7_shrink_target.wav"
    enh_f_ = f"{os.path.expanduser('~')}/model_results_trunk/FIG6/fig6_GAN/baseline_fig6/output/test_noise92/0.0/7_shrink_target.wav"

    draw_spl(src_f, tgt_f, enh_f, enh_f_)
