#!/usr/bin/env python3
import numpy as np
import librosa
from audiolib import audioread
from models.conv_stft import STFT


def mfcc(data: np.ndarray, n_mfcc: int, nframe: int = 1024, nhop: int = 512, fs: int = 16000):
    """
    data: T,C if multi-channel

    return: n_mfcc, T
    """
    mfccs = librosa.feature.mfcc(
        y=data, sr=fs, n_mfcc=n_mfcc, n_fft=nframe, hop_length=nhop, center=True
    )
    return mfccs


if __name__ == "__main__":
    import torch

    data, fs = audioread("../test.wav")
    d = STFT(1024, 512, 1024).transform(torch.from_numpy(data[None, :, 0]).float())
    out = mfcc(data[:, 0], 5, 512, 256)
    print(out.shape, d.shape, out[:, 0])
    pass
