import librosa
import numpy as np
import random
import math
from scipy.signal import get_window
import soundfile as sf
from utils.audiolib import audioread
from typing import Any


class GeneHowling:
    def __init__(self, nframe, nhop=None) -> None:
        if nhop is None:
            self.nhop = nframe // 2
        self.nframe = nframe
        self.win = get_window("hann", self.nframe)

    def compute_MSG(self, rir):
        """marginally stable gain"""
        ir_spec = np.fft.rfft(rir)  # complex, N//2+1
        ir_mag = np.abs(ir_spec)
        ir_phase = np.angle(ir_spec)

        MLG = (np.abs(ir_mag) ** 2).mean()

        # zero_phase_idx = np.where(np.logical_and(-0.1 < ir_phase, ir_phase < 0.1))
        ir_zero_phase_mag = ir_mag[(ir_phase < 0.1) & (ir_phase > -0.1)]
        peak_gain = (ir_zero_phase_mag**2).max()
        MSG = -10 * np.log10(peak_gain / MLG)

        return MSG

    def scale_ir(self, gain):
        pass

    def howling(self, x, ir):
        """
        x: T,
        ir: L,
        """
        L = x.size(0)
        howling = np.zeros(L)
        conv_len = self.nframe + ir.size(0) - 1
        st = 0

        for i in range(L):
            cur_frame = x[st : st + self.nframe]

            st += self.nhop

    def __call__(self, x: np.ndarray, rir: np.ndarray = None) -> Any:
        if rir is None:
            return x

        """
        when if Gain is very close to MSG, the system falls close to the boundary between stable and unstable states,
        and a small variation of the system may cause howling.
        To avoid this problem, Gain should be smaller than by 2-3dB
        """

        target_gain = self.compute_MSG(rir) + 2


if __name__ == "__main__":
    rir_f = "/home/deepni/disk/depositary/DNS-Challenges/rirs/simulated_rirs_16k/smallroom/Room001/Room001-00001.wav"
    rir, fs = audioread(rir_f)
    print(rir.shape, fs)
    obj = GeneHowling(512)
    obj.compute_MSG(rir)
