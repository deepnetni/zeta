import shutil
import sys
import os

import numpy as np

from core.utils.audiolib import (
    audioread,
    apply_rir,
    audiowrite,
    AcousticFeedbackSim,
)
from core.utils.rirGenerator import RIRGenerator
from core.utils.howling_detection import HowlingDection
import pickle
import scipy.signal
from matplotlib import pyplot as plt
from collections import deque


# def work(inp, rir, nframe=256, nhop=128, delta=0.0):
#     dwav = audioread(inp)
#     drir = audioread(rir)

#     dtemp = np.zeros(1, nhop)

#     Nframe = len(dwav) // nhop


def acoustic_feedback(inp, nhop, rir, G: float = 1.0):
    """
    :param inp: (T, ) or (T, C)
    :param nhop:
    :param rir:
    :returns:

    """
    nframe = 2 * nhop
    N = ((len(inp) - nframe) // nhop) * nhop + nframe
    inp = np.pad(inp[:N], pad_width=(nframe // 2, nframe // 2))

    FB = AcousticFeedbackSim(rir, nhop)

    # idx, frameN
    idx = np.arange(0, N - nframe + 1, nhop)[None, :].T + np.arange(nframe)
    frames = inp[idx]
    prev = np.zeros(nhop, dtype=np.float32)

    det = HowlingDection(nhop + 1, thresholds=[10, 10, 0])
    win = np.sqrt(scipy.signal.get_window("hann", nframe, fftbins=True))
    buff = deque([np.zeros(nhop, dtype=np.float32) for _ in range(2)], maxlen=2)

    G_ = FB.compute_MSG(1)
    out = []
    for fi, data in enumerate(frames):
        # 1. mix feedback
        cur = data[:nhop] + prev  # N,
        cur = cur * G

        cur = np.minimum(cur, np.ones_like(cur))
        cur = np.maximum(cur, -np.ones_like(cur))

        # 2. AHS

        # buff.append(cur)
        # cur_frame = np.concatenate(buff, axis=0)
        # cur_frame = cur_frame * win
        # xk = np.fft.rfft(cur_frame)
        # is_hs = det.is_howling(xk)
        # if is_hs is False:
        #     print(fi * 64, is_hs)

        out.append(cur)

        # 2. playback
        prev = FB(cur)

    out = np.concatenate(out, axis=0)
    return out


def geneRIR():
    gene = RIRGenerator.from_yaml("conf_afc/rir/rir_conf.yaml")
    outd = "/home/deepni/datasets/dnsChallenge/gene_rirs"
    if not os.path.exists(outd):
        os.makedirs(outd)

    for i in range(50):
        meta = gene.sample()
        h = meta["h"]
        margin = np.random.uniform(2, 4, len(h)).round(2)
        G = []
        for idx, (h_, m_) in enumerate(zip(h, margin)):
            FB = AcousticFeedbackSim(h_, 64)
            msg, peak_g = FB.compute_MSG(m_)
            print(i, idx, msg, peak_g)
            G.append((msg, peak_g))
        meta.update({"gain": G})
        gene.save(meta, f"{outd}/{i}.pkl")


if __name__ == "__main__":
    # gene = RIRGenerator.from_yaml()
    # meta = gene.sample()
    # gene.save(meta)
    # inp = np.random.randn(128)
    # rir = np.random.randn(1024)

    # geneRIR()
    # sys.exit()

    outd = "/home/deepni/datasets/dnsChallenge/gene_rirs"

    inp, fs = audioread("src.wav")
    inp2, fs = audioread("src_2.wav")
    # with open("out.pkl", "rb") as f:
    with open(f"{outd}/5.pkl", "rb") as f:
        meta = pickle.load(f)

    h = meta["h"][5].squeeze()

    G = 5.5
    # inp = inp * 0.3

    # inp, fs = audioread(
    #     "/home/deepni/github/python_howling_suppression/test/added_howling.wav"
    # )
    out = acoustic_feedback(inp, 64, h, G)
    print(out.shape)

    # inp_ = np.stack([inp, inp2], axis=0)
    # hobj = BLKConvHandler(h, 64)
    # out = hobj.apply(inp_)
    # out = hobj(inp_)
    # o1 = apply_rir(inp, h)
    # N = min(out.shape[-1], len(o1))
    # print(np.allclose(out[:N], o1[:N]), ((out[:N] - o1[:N]) ** 2).sum())
    # print(np.allclose(out[0, :N], o1[:N]), ((out[0, :N] - o1[:N]) ** 2).sum())
    # o2 = apply_rir(inp, h_norm, norm=False)
    # o3 = apply_rir(inp, h, norm=False)
    # audiowrite("out_check.wav", np.stack([o1[:N], out[:N]], axis=-1), fs)

    N = min(len(out), len(inp))
    audiowrite(f"out_how_{G}.wav", np.stack([inp[:N], out[:N]], axis=-1), fs)
