#!/usr/bin/env python3
import os
import sys
import torch

# sys.path.append(__file__.rsplit("/", 2)[0])
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.audiolib import audioread
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import get_window


inp, fs = audioread("/home/deepni/github/zeta/src.wav")
print(inp.shape)
win = np.sqrt(get_window("hann", 256, fftbins=True))
x = torch.stft(
    torch.from_numpy(inp),
    n_fft=256,
    hop_length=128,
    window=torch.from_numpy(win),
    return_complex=True,
)
xk = torch.abs(x)
print(xk.shape)
plt.hist(xk[100, :], bins=50)
plt.savefig("try.svg")
