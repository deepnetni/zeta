import librosa
import scipy
import yaml
import glob
import os
import random
import soundfile as sf
import numpy as np
from typing import Dict

from .wdrc_transforms import TransConfig, apply_gain, transform_pipline

# from utils.HAids.wdrc import wdrc_process
from utils.HAids.PyFIG6.pyFIG6 import FIG6_compensation_vad
from utils.vad import VAD


def rms(audio, db=False):
    audio = np.asarray(audio)
    rms_value = np.sqrt(np.mean(audio**2))
    if db:
        return 20 * np.log10(rms_value + np.finfo(float).eps)
    else:
        return rms_value


def normalize(audio, target_level=-25, rms_ix_start=0, rms_ix_end=None, return_scalar=False):
    """Function to normalize"""
    rms_value = rms(audio[rms_ix_start:rms_ix_end])
    scalar = 10 ** (target_level / 20) / (rms_value + np.finfo(float).eps)
    audio = audio * scalar
    if return_scalar:
        return audio, scalar
    else:
        return audio


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    def butter_bandpass(lowcut, highcut, fs, order=5) -> np.ndarray:
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = scipy.signal.butter(order, [low, high], analog=False, btype="band", output="sos")
        return sos

    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = scipy.signal.sosfilt(sos.astype(np.float32), data)
    ymax = np.max(np.abs(y))
    xmax = np.max(np.abs(data))
    ymax_is_close_to_xmax = max(1, 1.25 * xmax) > ymax
    if ymax_is_close_to_xmax:
        return y
    else:
        print(
            "Signal is not finite or has very high values after bandpass filtering, returning original data."
        )
        print("Probably the order parameter is too high.")
        print(
            "Max value in bandpass augmented data (the augmentation is ignored), (xmax, ymax):",
            xmax,
            ymax,
        )
        print("lowcut, highcut, fs, order:", lowcut, highcut, fs, order)
        return data


def audioread(path, sr=None, norm=False, dtype=np.float64, resample_type: str = "kaiser_fast"):
    """Read audio from path and return an numpy array.

    Parameters
    ----------
    path: str
        path to audio on disk.
    sr: int, optional
        Sampling rate. Default to None.
        If None, do nothing after reading audio.
        If not None and different from sr of the file, resample to new sr.
    norm:
        Normalize audio level
    dtype:
        Data type
    resample_type:
        librosa.resample resample algorithm
    """
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise ValueError("[{}] does not exist!".format(path))

    while True:
        try:
            x, xsr = sf.read(path, dtype=dtype)
            break
        except RuntimeError:
            print(f"ERRONEOUS PATH: {path}.")

    if len(x.shape) == 1:  # mono
        if sr and (xsr != sr):
            x = librosa.resample(x, orig_sr=xsr, target_sr=sr, res_type=resample_type)
            xsr = sr
        if norm:
            x /= np.max(np.abs(x))
        return x, xsr
    else:  # multi-channel
        x = x.T
        if sr and (xsr != sr):
            x = librosa.resample(x, orig_sr=xsr, target_sr=sr, res_type=resample_type)
            xsr = sr
        # Force mono
        x = x.sum(axis=0) / x.shape[0]
        if norm:
            for chan in range(x.shape[0]):
                x[chan, :] /= np.max(np.abs(x[chan, :]))

        return x, xsr


class DatasetDict:
    def __init__(self, config, sample_rate, resample_type):
        self.sample_rate = sample_rate
        self.resample_type = resample_type
        datasets = dict()
        weights = dict()

        for dataset_name, data_config in config.items():
            if "weight" in data_config and data_config["weight"] <= 0:
                continue

            if dataset_name in datasets:
                raise ValueError(f"Duplicate dataset '{dataset_name}'")

            files = list(glob.glob(os.path.join(data_config["dir"], "**/*.wav"), recursive=True))
            if len(files) > 0:
                datasets[dataset_name] = files
                weights[dataset_name] = data_config["weight"]
            else:
                raise ValueError(f"Empty dataset, ignoring: {dataset_name}")

        if len(weights) == 0:
            raise RuntimeWarning("No datasets are defined.")

        self.datasets = datasets
        names = list(weights.keys())
        props = np.array(list(weights.values()), dtype="float")
        props /= sum(props)
        self.names = names
        self.props = props

    def sample(self, duration, normalize=False):
        name = np.random.choice(self.names, p=self.props)
        data, meta = self.sample_from(
            name, self.datasets[name], duration=duration, normalize=normalize
        )

        return data, meta

    def sample_from(self, dataset_name, file_paths, duration, normalize):
        remaining_samples = round(duration)

        utterances, clip_file_names = [], []
        while remaining_samples > 0:
            file_path = random.sample(file_paths, 1)[0]
            clip_file_names.append(file_path)

            x, sr = audioread(file_path, sr=self.sample_rate, resample_type=self.resample_type)

            if len(x) >= remaining_samples:
                offset = random.randint(0, len(x) - remaining_samples)
                x = x[offset : offset + remaining_samples]

            utterances.append(x)
            remaining_samples -= len(x)

        data = dict(
            audio=np.concatenate(utterances),
        )
        meta = dict(dataset=dataset_name, normalized=normalize, files=clip_file_names)
        return data, meta

    def total_duration(self):
        total_duration = 0.0  # in seconds
        for dataset_name, data_paths in self.datasets.items():
            for path in data_paths:
                x, sr = audioread(path)
                total_duration += len(x) / sr

        return total_duration


class Synthesizer:
    def __init__(self, cfg_path: str):
        with open(cfg_path) as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)

        self.nearend_datasets = DatasetDict(
            self.cfg["onlinesynth_nearend_datasets"],
            sample_rate=self.cfg["onlinesynth_sampling_rate"],
            resample_type=self.cfg["onlinesynth_resampling_type"],
        )
        self.noise_datasets = DatasetDict(
            self.cfg["onlinesynth_nearend_noises"],
            sample_rate=self.cfg["onlinesynth_sampling_rate"],
            resample_type=self.cfg["onlinesynth_resampling_type"],
        )
        try:
            self.self_datasets = DatasetDict(
                self.cfg["onlinesynth_self_record_noises"],
                sample_rate=self.cfg["onlinesynth_sampling_rate"],
                resample_type=self.cfg["onlinesynth_resampling_type"],
            )
        except:
            # print(
            #     f"Warning: {self.cfg['onlinesynth_self_record_noises']} invalid, making gaussian noise in range [noise_db-10, noise_db]."
            # )
            self.self_datasets = None

        self.vad_detect = VAD(10, self.cfg["onlinesynth_sampling_rate"], level=2)
        self.transforms = [apply_gain]

        self.fig6_cfg = self.cfg["fig6_compensation_conf"]
        audiogram_f = self.fig6_cfg["audiogram_file"]
        with open(audiogram_f, "r", encoding="utf-8") as fp:
            ctx = [l.strip().replace("\t", ",").replace(" ", ",") for l in fp.readlines()]
            ctx = [list(map(float, l.split(","))) for l in ctx]
            ctx = np.array(ctx)  # N, 6

        self.audiogram = ctx

    @staticmethod
    def _mix_signals(x_clean, x_noise, snr, rms_clean=None, rms_noise=None, eps=1e-12):
        assert len(x_clean) == len(
            x_noise
        ), f"len(x_clean): {len(x_clean)}, len(x_noise): {len(x_noise)}"

        if rms_clean is None:
            rms_clean = rms(x_clean)
        if rms_noise is None:
            rms_noise = rms(x_noise)
        clean_is_empty = rms_clean < eps

        if clean_is_empty and rms_clean is None:
            scalar = 1.0
        else:
            # a / b / c = a / (b x c)
            scalar = rms_clean / (rms_noise + eps) / (10 ** (snr / 20))

        noise = scalar * x_noise
        mix = x_clean + noise

        norm_scalar = mix.max()
        if norm_scalar > 1.0:
            mix = mix / norm_scalar
            noise = noise / norm_scalar
        else:
            norm_scalar = 1.0

        return mix, noise, norm_scalar

    def _generate_nearend(self):
        dur_nearend = self.cfg["onlinesynth_duration"] * self.cfg["onlinesynth_sampling_rate"]

        if dur_nearend > 0:
            while True:
                data, meta = self.nearend_datasets.sample(duration=dur_nearend)
                x_src = data["audio"]
                # NOTE check if silenet
                if rms(x_src, True) <= -35:
                    continue
                x_transform, info = transform_pipline(
                    x_src, self.transforms, TransConfig(gain_duration=None)
                )
                if x_transform is not None:
                    break
        else:
            raise ValueError("Unsupported nearend duration value!")

        self.vad_detect.reset()
        x_vad = self.vad_detect.vad_waves(x_src)  # T,
        N = len(x_vad)
        x_src = x_src[:N]
        x_transform = x_transform[:N]
        x_nearend = x_transform.copy()

        if random.random() < self.cfg.get("onlinesynth_nearend_apply_gain_change", 0.0):
            gain_change = random.uniform(-18, 18)
            gain_change = 10 ** (gain_change / 20)
            ix_start = random.randint(0, len(x_nearend) - len(x_nearend) // 5)
            ix_end = None
            if random.random() < 0.1:
                ix_end = random.randint(ix_start, len(x_nearend) - 1)
            x_nearend[ix_start:ix_end] *= gain_change
            meta["gain_change"] = gain_change

        x_noise = None
        if random.random() < self.cfg["onlinesynth_nearend_prop_noisy"]:
            noise_data, noise_meta = self.noise_datasets.sample(duration=dur_nearend)
            x_noise = noise_data["audio"]  # get noise

            if random.random() < self.cfg["onlinesynth_nearend_prop_add_gaussian_ne_noise"]:
                # replace the gaussian noise with our self record sample.
                if self.self_datasets is not None:
                    bgnoise_data, bgnoise_meta = self.self_datasets.sample(duration=dur_nearend)
                    bgnoise = bgnoise_data["audio"]
                    x_noise += bgnoise  # adding self record noise
                else:
                    noise_db = rms(x_noise, db=True)
                    # make gaussian noise in range [noise_db-10, noise_db]
                    std = 10 ** ((noise_db - random.random() * 10) / 20)
                    gaussian_noise = std * np.random.randn(len(x_noise))
                    x_noise += gaussian_noise.astype(np.float32)

            snr_interval = self.cfg["onlinesynth_nearend_snr_interval"]
            snr = random.uniform(min(snr_interval), max(snr_interval))

            x_nearend, x_noise, norm_scalar = self._mix_signals(
                x_nearend, x_noise, snr, rms_clean=rms(x_transform), rms_noise=rms(x_noise)
            )
            x_transform = x_transform / norm_scalar
            x_src = x_src / norm_scalar

        # x_target, _ = wdrc_process(x_transform)
        HL = self.audiogram[np.random.choice(self.audiogram.shape[0])]
        info.update({"HL": np.array2string(HL, separator=",")})

        # x_target = FIG6_compensation(
        #     HL,
        #     x_transform,
        #     self.fig6_cfg["sample_rate"],
        #     self.fig6_cfg["nframe"],
        #     self.fig6_cfg["nhop"],
        # )
        # x_target_vad = np.where(x_vad > 0.5, x_target, x_transform)
        x_target_vad = FIG6_compensation_vad(
            HL,
            x_transform,
            self.fig6_cfg["sample_rate"],
            self.fig6_cfg["nframe"],
            self.fig6_cfg["nhop"],
            x_vad,
        )

        x_nearend_fig6 = FIG6_compensation_vad(
            HL,
            x_nearend,
            self.fig6_cfg["sample_rate"],
            self.fig6_cfg["nframe"],
            self.fig6_cfg["nhop"],
        )

        info.update(
            {
                "input_RMS": rms(x_transform, True).round(2),
                "output_RMS:": rms(x_target_vad, True).round(2),
            }
        )
        if x_noise is None:
            x_noise = np.zeros_like(x_nearend)

        assert len(x_nearend) == len(x_transform) == len(x_noise) == len(x_target_vad)

        # # normalize volume
        # nearend_level = self.cfg.get("onlinesynth_nearend_normalize_volume", None)
        # target_level = self.cfg.get("onlinesynth_target_normalize_volume", None)
        # if nearend_level is not None:
        #     x_nearend, norm_scalar = normalize(
        #         x_nearend, target_level=nearend_level, return_scalar=True
        #     )
        #     x_target *= norm_scalar

        # if target_level is not None:
        #     x_target = normalize(x_target, target_level=target_level)

        output = dict(
            src=x_src,  # the original data
            transform=x_transform,  # data before wdrc
            target=x_target_vad,  # wdrc(x_transform)
            nearend=x_nearend,  # mix, x_transform + noise
            nearend_fig6=x_nearend_fig6,
            noise=x_noise,
            info=info,
            vad=x_vad,
        )
        return output

    def _generate_mic(self, data):
        x_target = data["target"]
        x_nearend = data["nearend"]

        assert len(x_target) == len(x_nearend)

        x_mic = x_nearend.copy()
        if (
            random.random()
            < self.cfg["onlinesynth_mic_distortions"]["distortion_types"]["param_ranges"][
                "mic_bandpass"
            ]["likelihood"]
        ):
            bandpass_cfg = self.cfg["onlinesynth_mic_distortions"]["distortion_types"][
                "param_ranges"
            ]["mic_bandpass"]
            x_mic = butter_bandpass_filter(
                x_mic,
                random.uniform(*bandpass_cfg["low_freq"]),
                random.uniform(*bandpass_cfg["high_freq"]),
                self.cfg["onlinesynth_sampling_rate"],
                order=bandpass_cfg["order"],
            )

        # if self.distorter is not None:
        #     x_mic, params = self.distorter.apply_distortions(
        #         x_mic, sample_rate_hz=self.cfg["onlinesynth_sampling_rate"]
        #     )

        data["mic"] = x_mic
        return data

    def generate(self) -> Dict:
        """
        Returns a dict with target (desired signal), nearend (target signal affected by noise and some gain changes)
        and mic (nearend signal affected by various distortions).
        """
        data = self._generate_nearend()
        # data = self._generate_mic(data)

        return data


if __name__ == "__main__":
    """
    Usage example for the Synthesizer. This example generates a training data dict.
    """
    s = Synthesizer(r"synthesizer_config_wdrc.yaml")

    # audio = s.generate()
    # print("#", len(audio["target"]))

    # sf.write(r".\target.wav", audio["target"], 48000)
    # sf.write(r".\nearend.wav", audio["nearend"], 48000)
    # sf.write(r".\mic.wav", audio["mic"], 48000)
