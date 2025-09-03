"""
Description:
Author: deepnetni
Date: 2022-10-26
LastEditors: user
LastEditTime: 2022-11-14
"""
import os
import sys
import numpy as np
import glob
from typing import Union, Tuple, Optional
import soundfile as sf
from matplotlib import pyplot as plt


"""
param {in} x, the farend signal also known as reference signal
param {in} d, the nearend signal, mic signal
param {in} N, length of each block
param {in} M, number of blocks in the filter
param {in} win, window func
return {*}, error signal, weights
"""


def PBFDAF(
    x: np.ndarray, d: np.ndarray, N: int = 256, M: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    # * init core parameters
    mu = 0.9  # * update lag
    threshold = 2.0e-6  # * max update value each time
    alp = 0.9  # * to smooth the power

    # * discard the remaining data at the end that can't form a block
    blks = int(min(len(x), len(d)) / N)
    L = blks * N
    x = x[:L]
    d = d[:L]
    er = np.zeros((L,))
    est = np.zeros((L,))
    pn = np.zeros((N + 1,))

    # * padding zeros in the front
    xpad = np.concatenate(
        [
            np.zeros(
                N,
            ),
            x,
        ],
        dtype=np.float32,
    )
    indices = np.arange(0, len(xpad), N).reshape(-1, 1) + np.arange(N)
    xb = xpad[indices]  # * [nframe, N]
    # * get [xold, xnew]
    xbs = np.concatenate([xb[:-1], xb[1:]], axis=-1)
    # * get [dblock]
    indices = np.arange(0, len(d), N).reshape(-1, 1) + np.arange(N)
    db = d[indices]

    # * do the row-based fft
    Xkbs = np.fft.rfft(xbs, axis=-1)
    # * insert zeros blocks at the end of each block
    Wkb = np.zeros((M, N + 1), dtype=np.complex128)
    XTmp = np.zeros((M, N + 1), dtype=np.complex128)

    for i in range(blks):
        # * estimate the output
        XTmp[0, :] = Xkbs[i, :]
        Ykb = XTmp * Wkb
        Ykb = np.sum(Ykb, axis=0)  # * summing the results of all blocks
        yb = np.real(np.fft.irfft(Ykb, axis=-1))
        yb = yb[N:]

        # * smooth the power spectrum
        pn = alp * pn + (1 - alp) * M * np.real(np.conj(Xkbs[i, :]) * Xkbs[i, :])

        # * calculate the error
        erb = db[i, :] - yb
        er[i * N : (i + 1) * N] = erb  # * record the error history during processing
        est[i * N : (i + 1) * N] = yb  # * record the estimate echo signal
        Ekb = np.fft.rfft(
            np.concatenate(
                [
                    np.zeros(
                        N,
                    ),
                    erb,
                ],
                axis=-1,
            )
        )
        Ekb_norm = Ekb / (pn + 1e-10)  # * norm error in frequency domain

        # * compared to thresholds
        tmp = np.maximum(np.absolute(Ekb_norm), threshold)
        Ekb_norm = Ekb_norm * threshold / tmp
        muEk = mu * Ekb_norm

        # * update weights of the filter
        PP = np.conj(XTmp) * muEk
        pp = np.fft.irfft(PP, axis=-1)
        plpad = np.concatenate([pp[:, :N], np.zeros((M, N))], axis=-1)
        PP = np.fft.rfft(plpad, axis=-1)
        Wkb += PP

        XTmp[1:, :] = XTmp[0:-1, :]

    return er, est


def run_aec(reff, sphf, ofname=None):
    dref, fs = sf.read(reff)
    dmic, _ = sf.read(sphf)

    dmic *= 32768
    dref *= 32768

    cln, est = PBFDAF(dref, dmic, 256, 10)
    cln /= 32768.0
    est /= 32768.0

    plt.figure(figsize=(19.2, 10.8))
    plt.subplot(411)
    plt.plot(dmic / 32768)
    plt.title("Mic")
    plt.grid(visible=True, which="major", ls="-", axis="y", linewidth=0.5)
    plt.subplot(412)
    plt.plot(cln)
    plt.title("Post")
    plt.grid(visible=True, which="major", ls="-", axis="y", linewidth=0.5)
    plt.subplot(413)
    plt.plot(est)
    plt.title("Estimation Echo")
    plt.grid(visible=True, which="major", ls="-", axis="y", linewidth=0.5)
    plt.subplot(414)
    plt.plot(dref / 32768.0)
    plt.title("Reference Signal")
    plt.grid(visible=True, which="major", ls="-", axis="y", linewidth=0.5)
    plt.tight_layout()
    if ofname is not None:
        plt.savefig(ofname, bbox_inches="tight")
    else:
        plt.show()
    plt.close()
    return cln, est, fs


if __name__ == "__main__":
    test_dir = "H:\\experiments\\PBFDAF\\blind_test_set\\clean\\"
    out_dir = "H:\\experiments\\PBFDAF\\"

    # run_aec("E:\\data-Matlab\\x.wav", "E:\\data-Matlab\\d.wav")

    dbtalk = "*_doubletalk_*lpb.wav"
    sgtalk = "*_farend_singletalk_*lpb.wav"

    for reff in glob.glob(test_dir + dbtalk):
        sphf = reff.replace("lpb", "mic")
        if not os.path.exists(sphf):
            continue
        print("processing ", reff)
        fname = os.path.split(reff)[1]
        fname = reff.replace("lpb", "fig")
        fname = fname.replace(".wav", ".svg")
        cln, est, fs = run_aec(reff, sphf, fname)
        # cln, est, fs = run_aec(reff, sphf)
        clnf = reff.replace("lpb", "post")
        sf.write(clnf, cln, fs)
        estf = reff.replace("lpb", "est")
        sf.write(estf, est, fs)
