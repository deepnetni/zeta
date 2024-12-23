from .synthesizer import *


class Synthesizer_infant(Synthesizer):
    def __init__(self, cfg_path: str = "../template/synthesizer_config_infant.yaml"):
        super().__init__(cfg_path)

        self.infant_datasets = DatasetDict(
            self.cfg["onlinesynth_infant_datasets"],
            sample_rate=self.cfg["onlinesynth_sampling_rate"],
            resample_type=self.cfg["onlinesynth_resampling_type"],
        )

    def _generate_nearend(self):
        data = super()._generate_nearend()
        mic = data["nearend"]  # sph + noise, shape T,
        noise = data["noise"]

        if random.random() < self.cfg["onlinesynth_prop_add_nearend_speech"]:
            N = np.random.randint(int(len(mic) * 0.6), int(len(mic) * 0.8))
            while True:
                data_cry, _ = self.infant_datasets.sample(duration=N)
                data_cry = data_cry["audio"]

                vad = data_cry[:, 1]

                if vad.mean() > 0.7:
                    break

            Nc = len(data_cry)
            st = np.random.randint(len(mic) - Nc)

            data_cry_pad = np.concatenate(
                [np.zeros((st, 2)), data_cry, np.zeros((len(mic) - Nc - st, 2))], axis=0
            )

            snr_interval = self.cfg["onlinesynth_infant_nearend_snr_interval"]
            snr = random.uniform(min(snr_interval), max(snr_interval))
            x_nearend, x_cry, scaler = self._mix_signals(
                mic, data_cry_pad[:, 0], snr, rms_clean=rms(mic), rms_noise=rms(data_cry[:, 0])
            )
            x_nearend = np.stack([x_nearend, data_cry_pad[:, 1]], axis=1)
        else:
            N = len(noise)
            data_cry, _ = self.infant_datasets.sample(duration=N)
            data_cry = data_cry["audio"]
            data_c, data_vad = data_cry[:, 0], data_cry[:, 1]
            if check_power(noise, -35):
                snr_interval = self.cfg["onlinesynth_infant_noise_snr_interval"]
                snr = random.uniform(min(snr_interval), max(snr_interval))
                data_c, noise, scaler = self._mix_signals(
                    data_c, noise, snr, rms_clean=rms(data_c), rms_noise=rms(noise)
                )
            else:  # apply random gain.
                gain_v = self.cfg["onlinesynth_infant_gain_change"]
                gain_change = random.uniform(min(gain_v), max(gain_v))
                gain_change = 10 ** (gain_change / 20)
                ix_start = random.randint(0, len(data_c) - len(data_c) // 5)
                ix_end = None
                data_c[ix_start:ix_end] *= gain_change

                norm_scalar = data_c.max()
                if norm_scalar > 0.9:
                    norm_scalar /= 0.9
                    data_c /= norm_scalar
                else:
                    norm_scalar = 1.0

            x_nearend = np.stack([data_c, data_vad], axis=-1)

        return dict(mic=x_nearend)

    def generate(self):
        data = self._generate_nearend()
        return data


if __name__ == "__main__":
    aug = Synthesizer_infant()
    aug.generate()
