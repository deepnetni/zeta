import torch
import numpy as np
import librosa
from pesq import pesq
from pystoi import stoi

from typing import Callable, Dict, List, Optional, Union
from utils.HAids.PyHASQI.HASQI_revised import HASQI_v2_for_unfixedLen, HASQI_v2


# def l2_norm(s, keepdim=False):
#     """
#     return: sqrt(|vec| * 2)
#     """
#     return torch.linalg.norm(s, dim=-1, keepdim=keepdim)


def l2_norm(s, keepdim=False):
    """
    return: sqrt(|vec| * 2)
    """
    return np.linalg.norm(s, axis=-1, keepdims=keepdim)


# def compute_si_snr(
#     sph: Union[torch.Tensor, np.ndarray],
#     est: Union[torch.Tensor, np.ndarray],
#     zero_mean=True,
# ):
#     """torch-based
#     s1 is the est signal, s2 represent for clean speech
#     """
#     is_numpy = False
#     if isinstance(sph, np.ndarray) or isinstance(est, np.ndarray):
#         sph = torch.from_numpy(sph)
#         est = torch.from_numpy(est)
#         is_numpy = True

#     eps = torch.finfo(sph.dtype).eps

#     if zero_mean is True:
#         s = sph - torch.mean(sph, dim=-1, keepdim=True)
#         s_hat = est - torch.mean(est, dim=-1, keepdim=True)
#     else:
#         s = sph
#         s_hat = est

#     s_target = (
#         (torch.sum(s_hat * s, dim=-1, keepdim=True) + eps)
#         * s
#         / (l2_norm(s, keepdim=True) ** 2 + eps)
#     )
#     e_noise = s_hat - s_target
#     # sisnr = 10 * torch.log10(
#     #     (l2_norm(s_target) ** 2 + eps) / (l2_norm(e_noise) ** 2 + eps)
#     # )
#     sisnr = 10 * torch.log10(
#         (torch.sum(s_target**2, dim=-1) + eps)
#         / (torch.sum(e_noise**2, dim=-1) + eps)
#     )
#     return sisnr.cpu().detach().numpy() if is_numpy else sisnr


def compute_si_snr(
    sph: np.ndarray,
    est: np.ndarray,
    zero_mean=True,
):
    """numpy-based
    s1 is the est signal, s2 represent for clean speech
    """

    eps = np.finfo(sph.dtype).eps

    if zero_mean is True:
        s = sph - sph.mean(axis=-1, keepdims=True)
        s_hat = est - est.mean(axis=-1, keepdims=True)
    else:
        s = sph
        s_hat = est

    s_target = (
        (np.sum(s_hat * s, axis=-1, keepdims=True) + eps)
        * s
        / (l2_norm(s, keepdim=True) ** 2 + eps)
    )
    e_noise = s_hat - s_target
    # sisnr = 10 * torch.log10(
    #     (l2_norm(s_target) ** 2 + eps) / (l2_norm(e_noise) ** 2 + eps)
    # )
    sisnr = 10 * np.log10(
        (np.sum(s_target**2, axis=-1) + eps) / (np.sum(e_noise**2, axis=-1) + eps)
    )
    return sisnr


def compute_snr(
    sph: Union[torch.Tensor, np.ndarray],
    est: Union[torch.Tensor, np.ndarray],
    zero_mean=True,
):
    """numpy-based
    input: B,T
    """
    is_torch = False
    if isinstance(sph, torch.Tensor):
        sph = sph.cpu().detach().numpy()
        is_torch = True

    if isinstance(est, torch.Tensor):
        est = est.cpu().detach().numpy()
        is_torch = True

    assert isinstance(sph, np.ndarray) and isinstance(est, np.ndarray)
    eps = np.finfo(sph.dtype).eps
    if zero_mean is True:
        s = sph - sph.mean(axis=-1, keepdims=True)
        e = est - est.mean(axis=-1, keepdims=True)
    else:
        s = sph
        e = est  # B,T

    diff = np.sum((s - e) ** 2, axis=-1)
    s = np.sum(s**2, axis=-1)
    snr_sc = 10 * np.log10((s + eps) / (diff + eps))
    return snr_sc if is_torch is False else torch.from_numpy(snr_sc)


def compute_erle(mic, est):
    """numpy-based
    Args:
        mic:
        est:

    Returns:

    """
    pow_est = np.sum(est**2, axis=-1, keepdims=True)
    pow_mic = np.sum(mic**2, axis=-1, keepdims=True)

    erle_score = 10 * np.log10(pow_mic / (pow_est + np.finfo(np.float32).eps))
    return erle_score


def compute_pesq(lbl: np.ndarray, est: np.ndarray, fs=16000, norm=False, mode: str = "wb"):
    """numpy-based

    Args:
        lbl:
        est:
        fs:
        norm:

    Returns:

    """
    assert isinstance(lbl, np.ndarray)
    assert isinstance(est, np.ndarray)

    if fs > 16000:
        lbl = librosa.resample(lbl, orig_sr=fs, target_sr=16000)
        est = librosa.resample(est, orig_sr=fs, target_sr=16000)

    try:
        score = pesq(16000 if fs > 16000 else fs, lbl, est, mode)
    except Exception as e:
        # if "No utterances detected" in str(e):
        score = 0
        # print(e)

    if norm:
        score = (score - 1.0) / 3.5

    return score  # scaler


def compute_stoi(lbl: np.ndarray, est: np.ndarray, fs=16000):
    """numpy-based

    Args:
        lbl:
        est:
        fs:

    Returns:

    """
    try:
        score = stoi(lbl, est, fs)
    except Exception as e:
        score = 0
        print(e)

    return score  # scaler


def compute_precision_and_recall(lbl: np.ndarray, est: np.ndarray):
    """numpy-based
    lbl and est are bool value.
    """
    lbl = lbl.flatten()
    est = est.flatten()

    TP = np.sum((lbl == True) & (est == True))  # True Positives
    FP = np.sum((lbl == False) & (est == True))  # False Positives
    FN = np.sum((lbl == True) & (est == False))  # False Negatives

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    return precision, recall


def compute_hasqi(sdata, edata, HL, fs):
    sc = HASQI_v2_for_unfixedLen(sdata, fs, edata, fs, HL)
    # sc = HASQI_v2(sdata, fs, edata, fs, HL)
    if sc < 0:
        sc = 0.0
    return sc


if __name__ == "__main__":
    from torchmetrics.functional.audio import signal_distortion_ratio as SDR
    import torch

    inp = np.random.randn(2, 16000) + 10
    lbl = np.random.randn(2, 16000) + 10

    print(inp.shape)
    # l = compute_pesq(lbl, inp)
    # print(l)

    # inp = np.concatenate([inp, torch.zeros(20000)], axis=-1)
    # lbl = np.concatenate([lbl, torch.zeros(20000)], axis=-1)
    # l = compute_pesq(lbl, inp)
    # print(l)

    score = compute_si_snr(lbl, inp)
    print(score, type(score))

    score = SDR(preds=torch.from_numpy(inp), target=torch.from_numpy(lbl))
    print(score)
