import os
import ast
import numpy as np
from librosa import istft, stft
from scipy.signal import get_window
from utils.audiolib import audioread

from utils.HAids.PyFIG6.pyFIG6 import compute_subbands_SPL
from matplotlib import pyplot as plt
import pandas as pd


def cal_spl(inp: np.ndarray, nframe=128, nhop=64, fs=16000):
    """
    inp: B, T,
    """
    if inp.ndim == 1:
        inp = inp[None, :]
    SPL_offset = 96.7119344 * np.ones(16)
    nbin = nhop + 1
    freso = np.linspace(0, fs // 2, nbin)
    win = get_window("hann", nframe, fftbins=True)
    win = np.sqrt(win)
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

    # fmt: off
    ChannelNum_ft = np.array([0, 250, 375, 500, 625, 750, 1000, 1250, 1625, 2000, 2375, 2875, 3500, 4250, 5125, 6125, 8001])
    # fmt: on

    spl_in = compute_subbands_SPL(xk, freso, ChannelNum_ft, SPL_offset)  # T,B
    return spl_in.mean(axis=-1).squeeze()  # B,T


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
            {"label": "enh_", "linestyle": "--", "color": "r"},
        ],
    ):
        plt.plot(d, **param)
    plt.legend()
    plt.savefig(f"splin.svg", bbox_inches="tight")
    print("done")


def save_wave_excel(src, noisy, tgt, *args):
    d1, _ = audioread(src)
    d2, _ = audioread(tgt)
    d2 = d2[:, 0]
    d3, _ = audioread(noisy)
    d4, _ = audioread(args[0])

    N = min(len(d1), len(d2), len(d4))

    stat = {"Source": d1[:N], "Noisy": d3[:N], "Target": d2[:N]}
    for f, name in args:
        print(name)
        d, _ = audioread(f)
        d = d[:N]
        stat.update({name: d})

    df = pd.DataFrame(stat)
    df.to_excel("save.xlsx", index=False)
    print("done")


def save_spl_excel(src, noisy, tgt, enh, *args):
    d1, _ = audioread(src)
    d2, _ = audioread(tgt)
    d2 = d2[:, 0]
    d3, _ = audioread(enh)
    d4, _ = audioread(noisy)

    d1 = cal_spl(d1)
    d2 = cal_spl(d2)
    d3 = cal_spl(d3)
    d4 = cal_spl(d4)

    N = min(len(d1), len(d2), len(d3))

    stat = {"Source": d1[:N], "Noisy": d4[:N], "Target": d2[:N], "AFA-HearNet": d3[:N]}
    for f, name in args:
        d, _ = audioread(f)
        d = cal_spl(d)
        d = d[:N]
        stat.update({name: d})

    df = pd.DataFrame(stat)
    df.to_excel("save_spl.xlsx", index=False)
    print("done")


def draw_wave_spectrum(src, noisy, tgt, *args):
    fs = 16000
    d1, _ = audioread(src)
    d2, _ = audioread(tgt)
    d2 = d2[:, 0]
    d3, _ = audioread(noisy)
    d4, _ = audioread(args[0][0])

    N = min(len(d1), len(d2), len(d4))
    d1, d2, d3 = d1[:N], d2[:N], d3[:N]

    plt.figure(figsize=(8.5, 12))
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"

    ax = plt.subplot(7, 1, 1)
    # ax.grid(linestyle="--", linewidth=0.1, alpha=0.2)
    ax.plot(d3, color="lightgray", label="Noisy", linewidth=0.5)
    ax.plot(d2, color="blue", label="Target", linewidth=0.5)
    ax.plot(d1, color="red", label="Clean", linewidth=0.5)
    ax.grid(color="gray", linestyle="--", linewidth=0.1)
    # y axis confiugre
    ax.set_ylabel("Amplitude")
    ax.set_ylim(-1.0, 1.0)
    # ax.set_yticks([-0.5, 0.0, 0.5, 1.0])
    # x axis confiugre
    ax.set_xlabel("Time/s")
    ax.set_xlim(0, 5 * fs)
    xticks = ax.get_xticks()
    xticks = xticks[xticks != 0]
    ax.set_xticks(xticks)
    formatted_xticks = [f"{num:.2f}" for num in xticks / fs]
    ax.set_xticklabels(formatted_xticks)

    for i, (f, name) in enumerate(args, start=1):
        ax = plt.subplot(7, 2, i * 2 + 1)
        d, _ = audioread(f)
        d = d[:N]
        ax.plot(d)
        ax.set_xlabel("Time/s")

        ax = plt.subplot(7, 2, i * 2 + 2)
        print(name, i)
        d, _ = audioread(f)
        d = d[:N]
        ax.plot(d)
        ax.set_xlabel("Time1/s")

    plt.legend(loc="upper left", fontsize=12, bbox_to_anchor=(1, 1))
    plt.savefig(f"splin.svg", bbox_inches="tight")
    print("done")


if __name__ == "__main__":
    # fpattern = "28_enlarge"
    fpattern = "35_bypass"

    mdlist = [
        "db_aiat",
        "MP_SENet",
        "FSPEN",
        "SEMamba",
        "PrimeKNet",
        "condConformerVAD",
        # "condConformerVAD_mc36",
    ]
    tag = [
        "DB-AIAT",
        "MP-SENet",
        "FSPEN",
        "SEMamba",
        "PrimeK-Net",
        "AFA-HearNet",
        # "AFA-HearNet-small",
    ]

    src_f = f"{os.path.expanduser('~')}/github/zeta/show/{fpattern}_transform.wav"
    nsy_f = f"{os.path.expanduser('~')}/github/zeta/show/{fpattern}_nearend.wav"
    tgt_f = f"{os.path.expanduser('~')}/github/zeta/show/{fpattern}_target.wav"

    enh_f_l = []
    for m, n in zip(mdlist, tag):
        enh_f_ = f"{os.path.expanduser('~')}/model_results_trunk/FIG6/fig6_GAN_libriDemand/{m}/output/libri_demand_test/{fpattern}_target.wav"
        enh_f_l.append((enh_f_, n))
    # save_wave_excel(src_f, nsy_f, tgt_f, *enh_f_l)
    # save_spl_excel(src_f, nsy_f, tgt_f, enh_f, *enh_f_l)
    draw_wave_spectrum(src_f, nsy_f, tgt_f, *enh_f_l)
