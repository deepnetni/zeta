"""
Functions for howling frequency detection.

"""
from collections import deque
from functools import reduce
import numpy as np


def peak2average(frame, threshold=10):
    """Peak-to-Average Power Ratio (PAPR)
    Returns all frequency indices where power is greater than avarage power + threshold,
    which are possible candidates where howling occurs.

    Args:
        frame: Spectrum of one frame.
        threshold: Power threshold value in dB.
         The returned candidates should have power greater than average power + threshold.

    Returns:
        A list of selected frequncy indices.
        A list of PAPR value for every freqency bin of this frame.
    """
    eps = np.finfo(np.float32).eps
    power = np.abs(frame) ** 2
    papr = np.zeros(len(power))
    avarage = np.mean(power)
    # for i in range(len(frame)):
    #     papr[i] = 10 * np.log10(power[i] / avarage)
    #     if papr[i] > threshold:
    #         ret.append(i)
    papr = 10 * np.log10((power + eps) / (avarage + eps))
    ret = np.where(papr > threshold)[0]
    return ret.tolist(), papr


def peak2threshold(frame, threshold=10):
    """Peak-to-Threshold Power Ratio (PTPR)
    Returns all frequency indices where power is greater than threshold,
    which are possible candidates where howling occurs.

    Args:
        frame: Spectrum of one frame.
        threshold: Power threshold value in dB.

    Returns:
        A list of selected frequncy indices.
    """
    eps = np.finfo(np.float32).eps
    power = 10 * np.log10(np.abs(frame) ** 2 + eps)
    # ret = []
    # for i in range(len(frame)):
    #     if 10 * np.log10(power[i]) > threshold:
    #         ret.append(i)

    ret = np.where(power > threshold)[0]
    return ret.tolist()


def peak2neighboring(frame, threshold=15):
    """Peak-to-Neighboring Power Ratio (PNPR)
    Returns all frequency indices of power peaks,
    which are greater than neighboring frequency bins by a threshold.

    Args:
        frame: Spectrum of one frame.
        threshold: Power threshold value in dB.

    Returns:
        A list of selected frequncy indices.
    """
    eps = np.finfo(np.float32).eps
    power = np.abs(frame) ** 2 + eps

    # ret = []
    # for i in range(5, len(frame) - 5):
    #     if (
    #         10 * np.log10(power[i] / power[i - 4]) > threshold
    #         and 10 * np.log10(power[i] / power[i - 5]) > threshold
    #         and 10 * np.log10(power[i] / power[i + 4]) > threshold
    #         and 10 * np.log10(power[i] / power[i + 5]) > threshold
    #     ):
    #         ret.append(i)
    center = power[5:-5]
    ref_m4 = power[1:-9]
    ref_m5 = power[0:-10]
    ref_p4 = power[9:-1]
    ref_p5 = power[10:]

    cond = (
        (10 * np.log10(center / ref_m4) > threshold)
        & (10 * np.log10(center / ref_m5) > threshold)
        & (10 * np.log10(center / ref_p4) > threshold)
        & (10 * np.log10(center / ref_p5) > threshold)
    )

    ret = np.where(cond)[0] + 5

    return ret.tolist()


def peak_magnitude_persistence(candidates):
    """Inerframe Peak Magnitude Persistence (IPMP)
    Temporal howling detection criteria. Candidate should meet criteria
    in more than 3 frames out of 5 continuous frames.

    Args:
        candidates: nFreqs X nFrames
                    candidates[f][t] = 1 means a candidate at frequency[f]  at frame[t]
        index: Current frame index. starts with 0.

    Returns:
        A list of selected frequncy indices.
    """
    nf = candidates.shape[-1]
    assert nf >= 5, "cached frames must larger than 5."

    accu = np.sum(candidates[:, -4:], axis=1)

    # ipmp = np.argwhere(accu >= 3).squeeze()
    ipmp = np.flatnonzero(accu >= 3)
    # ipmp = [idx for idx, val in enumerate(accu) if val >= 3]
    return ipmp


def screening(frame, candidates):
    """
    Screen the candidates. Only one frequency in neighboring several frequencies is needed

    Args:
        frame: Current frame spectrum.
        candidates: nFreqs X nFrames
                    candidates[f][t] = 1 means a candidate at frequency[f]  at frame[t]

    Returns:
        A list of selected frequncy indices.
    """
    # ret = []
    # for c in candidates:
    #     if len(ret) == 0:
    #         ret.append(c)
    #     elif ret[len(ret) - 1] > c - 3:
    #         if abs(frame[ret[len(ret) - 1]]) < abs(frame[c]):
    #             ret[len(ret) - 1] = c
    #     else:
    #         ret.append(c)
    # candidates = sorted(candidates)
    ret = []

    for c in candidates:
        if not ret:
            ret.append(c)
        elif ret[-1] > c - 3:
            # Too close to previous one, keep the stronger peak
            if abs(frame[ret[-1]]) < abs(frame[c]):
                ret[-1] = c
        else:
            ret.append(c)

    return ret


class HowlingDection(object):
    def __init__(self, nbin, N=5, thresholds=[10, 10, 0]) -> None:
        self.buff = deque([np.zeros(nbin, dtype=np.int16) for _ in range(N)], maxlen=N)
        self.nbin = nbin
        self.nbin = nbin
        self.thres = thresholds

    def reset(self):
        self.buff = deque(
            [np.zeros(self.nbin, dtype=np.int16)] * len(self.buff), maxlen=self.buff.maxlen
        )

    def is_howling(self, spec: np.ndarray):
        ret = self(spec)
        return True if len(ret) != 0 else False

    def __call__(self, spec: np.ndarray):
        """
        spec: complex type
        """
        # assert spec.dtype == np.complex128, "type error."
        assert np.issubdtype(spec.dtype, np.complexfloating), "Expected complex dtype"

        cands = []
        if self.thres[0]:
            ptpr_idx = peak2threshold(spec, self.thres[0])
            cands.append(ptpr_idx)
        if self.thres[1]:
            papr_idx, _ = peak2average(spec, self.thres[1])
            cands.append(papr_idx)
        if self.thres[2]:
            pnpr_idx = peak2neighboring(spec, 3)
            cands.append(pnpr_idx)

        # intersec_idx = reduce(np.intersect1d, [ptpr_idx, papr_idx, pnpr_idx])
        intersec_idx = reduce(np.intersect1d, cands)
        tmp = np.zeros(spec.shape[0], dtype=np.int16)
        if len(intersec_idx) != 0:
            tmp[intersec_idx] = 1

        self.buff.append(tmp)
        candidates = np.stack(self.buff, axis=-1)  # F,T

        ipmp = peak_magnitude_persistence(candidates)
        result = screening(spec, ipmp)
        return result


def howling_detect(spec, candidates, frame_id):
    # insign = win * frame
    # spec = np.fft.fft(insign, nFFT, axis=0)

    # ==========  Howling Detection Stage =====================#
    ptpr_idx = peak2threshold(spec, 10)
    papr_idx, _ = peak2average(spec, 10)
    pnpr_idx = peak2neighboring(spec, 10)
    # intersec_idx = np.intersect1d(ptpr_idx, np.intersect1d(papr_idx, pnpr_idx))
    intersec_idx = np.intersect1d.reduce([ptpr_idx, papr_idx, pnpr_idx])
    # print("papr:",papr_idx)
    # print("pnpr:",pnpr_idx)
    # print("intersection:", intersec_idx)
    for idx in intersec_idx:
        candidates[idx][frame_id] = 1  # F,T
    ipmp = peak_magnitude_persistence(candidates)
    # print("ipmp:",ipmp)
    result = screening(spec, ipmp)
    # print("result:", result)
    return result


if __name__ == "__main__":
    i = np.random.randn(65)
    r = np.random.randn(65)
    spec = r + 1j * i
    det = HowlingDection(65)
    det.is_howling(spec)

    # ret, papr = peak2average(spec)
