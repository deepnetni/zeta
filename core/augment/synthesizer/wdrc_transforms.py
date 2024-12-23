import sys
from ast import Dict
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np


@dataclass
class TransConfig:
    fs: int = 16000

    # NOTE apply_gain function
    # shrink, bypass, enlarge probability
    gain_mode_p: list = field(default_factory=lambda: [0.4, 0.3, 0.3])
    gain_duration: Optional[list] = field(default_factory=lambda: [2.0, 4.0])
    clean_activity_threshold: float = 0.6
    noise_activity_threshold: float = 0.0
    target_level_lower: float = -35
    target_level_upper: float = -10
    target_level_base: float = 5  # the minimal gain differences

    def __post_init__(self):
        # NOTE apply_gain function
        if self.gain_duration is not None:
            self.gain_duration_min = int(self.gain_duration[0] * self.fs)
            self.gain_duration_max = int(self.gain_duration[1] * self.fs)
        else:
            self.gain_duration_min = None
            self.gain_duration_max = None


TemplateOutputType = Tuple[Optional[np.ndarray], Optional[dict]]
TemplateTransFunc = Callable[[Optional[np.ndarray], TransConfig], TemplateOutputType]

cl_dist_conf = TransConfig()


######################
# The functions part #
######################


def get_func_name():
    return sys._getframe().f_back.f_code.co_name


def rms(audio, db=False):
    audio = np.asarray(audio)
    rms_value = np.sqrt(np.mean(audio**2))
    if db:
        return 20 * np.log10(rms_value + np.finfo(float).eps)
    else:
        return rms_value


def normalize(audio, target_level=-25):
    """Normalize the signal to the target level"""
    EPS = np.finfo(float).eps
    rms = (audio**2).mean() ** 0.5
    scalar = 10 ** (target_level / 20) / (rms + EPS)
    audio = audio * scalar
    return audio


def activitydetector(audio, fs=16000, energy_thresh=0.13, target_level=-25):
    """Return the percentage of the time the audio signal is above an energy threshold"""
    EPS = np.finfo(float).eps

    audio = normalize(audio, target_level)
    window_size = 50  # in ms
    window_samples = int(fs * window_size / 1000)
    sample_start = 0
    cnt = 0
    prev_energy_prob = 0
    active_frames = 0

    a = -1
    b = 0.2
    alpha_rel = 0.05
    alpha_att = 0.8

    while sample_start < len(audio):
        sample_end = min(sample_start + window_samples, len(audio))
        audio_win = audio[sample_start:sample_end]
        frame_rms = 20 * np.log10(sum(audio_win**2) + EPS)
        frame_energy_prob = 1.0 / (1 + np.exp(-(a + b * frame_rms)))

        if frame_energy_prob > prev_energy_prob:
            smoothed_energy_prob = frame_energy_prob * alpha_att + prev_energy_prob * (
                1 - alpha_att
            )
        else:
            smoothed_energy_prob = frame_energy_prob * alpha_rel + prev_energy_prob * (
                1 - alpha_rel
            )

        if smoothed_energy_prob > energy_thresh:
            active_frames += 1
        prev_energy_prob = frame_energy_prob
        sample_start += window_samples
        cnt += 1

    perc_active = active_frames / cnt
    return perc_active


def apply_gain(x: Optional[np.ndarray], conf: TransConfig = cl_dist_conf) -> TemplateOutputType:
    """
    x: T,
    """
    if x is None:
        return None, None

    N = len(x)

    mode = np.random.choice([0, 1, 2], p=conf.gain_mode_p)
    if conf.gain_duration_min is not None and conf.gain_duration_max is not None:
        dur = np.random.randint(conf.gain_duration_min, conf.gain_duration_max + 1)
        dur = min(dur, N - 1)
        st = np.random.randint(0, N - dur)
        ed = st + dur
    else:
        st = 0
        ed = N

    x = np.copy(x)
    clip = x[st:ed]
    pow_db = rms(clip, True)

    if mode == 0:  # shrink
        tgt_pow = (
            np.random.uniform(
                conf.target_level_lower,
                max(conf.target_level_lower, pow_db - conf.target_level_base),
            )
            if pow_db > conf.target_level_lower
            else pow_db
        )
        mode = "shrink" if pow_db > conf.target_level_lower else "bypass"
    elif mode == 1:  # bypass
        tgt_pow = pow_db
        mode = "bypass"
    else:  # enlarge
        tgt_pow = (
            np.random.uniform(
                min(pow_db + conf.target_level_base, conf.target_level_upper),
                conf.target_level_upper,
            )
            if pow_db < conf.target_level_upper
            else pow_db
        )
        mode = "enlarge" if pow_db < conf.target_level_upper else "bypass"

    # tgt_pow = conf.target_level_lower
    g = tgt_pow - pow_db
    scaler = 10 ** (g / 20)
    clip = clip * scaler

    vmax = clip.max()
    if vmax >= 1:
        clip = clip / vmax

    x[st:ed] = clip

    meta = dict(
        mode=mode,
        gain=g.round(3),
        unit=get_func_name(),
        start=round(st / conf.fs, 3),
        end=round(ed / conf.fs, 3),
    )
    percactive = activitydetector(audio=x)
    if percactive > conf.clean_activity_threshold:
        return x, meta
    else:
        return None, None


def transform_pipline(
    data: np.ndarray,
    func: List[TemplateTransFunc],
    conf: TransConfig = cl_dist_conf,
):
    x = data
    meta = {}
    for f in func:
        x, info = f(x, conf)
        if info is not None:
            tag = info.pop("unit")
            meta.setdefault(tag, {}).update(info)

    return x, meta


if __name__ == "__main__":
    import soundfile as sf

    inp, fs = sf.read("check.wav")
    out, meta = transform_pipline(inp, [apply_gain])

    assert out is not None
    out = np.stack([inp, out], axis=1)
    # sf.write("out2.wav", out, fs)
    print("done")
    print(meta)
