from clarity.evaluator.hasqi import hasqi_v2
from clarity.utils.audiogram import Audiogram
import sys

sys.path.append("..")
from utils.HAids.wdrc import wdrc_process
from utils.HAids.PyFIG6.pyFIG6 import FIG6_compensation, FIG6_compensation_vad
import soundfile
import numpy as np

print(__file__, __file__.rsplit("/", 2))


if __name__ == "__main__":
    clean_path = "../utils/HAids/PyHASQI/TEST_wavs/clean_fig6_fileid_100.wav"
    noisy_path = "../utils/HAids/PyHASQI/TEST_wavs/noisy_fig6_fileid_100.wav"
    # clean_path = "../utils/HAids/PyHASQI/TEST_wavs/clean_fileid_100.wav"
    # noisy_path = "../utils/HAids/PyHASQI/TEST_wavs/noisy_fileid_100.wav"

    (clean_audio, fs1) = soundfile.read(clean_path)
    (noisy_audio, fs2) = soundfile.read(noisy_path)

    # HL = np.array([80, 85, 90, 80, 90, 80])
    HL = np.array([40, 45, 40, 40, 40, 40])
    freq = np.array([250, 500, 1000, 2000, 4000, 8000])

    hl_aud = Audiogram(levels=HL, frequencies=freq)

    Level1 = 65
    eq = 2

    # out, _ = wdrc_process(clean_audio)
    #
    # out = FIG6_compensation_vad(HL, clean_audio, fs1)
    # soundfile.write("fig6_out.wav", np.stack([clean_audio, out], axis=-1), fs1)

    sc, li, nolin, v = hasqi_v2(clean_audio, fs1, noisy_audio, fs2, hl_aud, eq, Level1)
    print(sc, li, nolin, v)
