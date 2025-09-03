import sys
import soundfile as sf
import json
import ast

# sys.path.append(__file__.rsplit("/", 2)[0])
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.HAids.PyHASQI.HASQI_revised import HASQI_v2, HASQI_v2_for_unfixedLen
from utils.HAids.PyFIG6.pyFIG6 import FIG6_compensation_vad

# clean_path = __file__.rsplit("/", 2)[0] + "/utils/HAids/PyHASQI/TEST_wavs/clean_fig6_fileid_100.wav"
# noisy_path = __file__.rsplit("/", 2)[0] + "/utils/HAids/PyHASQI/TEST_wavs/noisy_fig6_fileid_100.wav"
clean_path = "/home/deepnetni/trunk/dns_wdrc/test/0_enlarge_target.wav"
noisy_path = "/home/deepnetni/trunk/dns_wdrc/test/0_enlarge_nearend.wav"
# noisy_path = "/home/deepni/datasets/dns_wdrc/test/0_enlarge_transform.wav"
hl_f = "/home/deepnetni/trunk/dns_wdrc/test/0_enlarge.json"

with open(hl_f, "r") as fp:
    ctx = json.load(fp)
    HL = ast.literal_eval(ctx["HL"])


(clean_audio, fs1) = sf.read(clean_path)
(noisy_audio, fs2) = sf.read(noisy_path)
print(clean_audio.shape, noisy_audio.shape)

# HL = [80, 85, 90, 80, 90, 80]
noisy_audio_ = FIG6_compensation_vad(HL, noisy_audio)


Level1 = 65
eq = 2

N = 79872
clean_audio = clean_audio[:N]
noisy_audio = noisy_audio[:N]

temp_HASQI = HASQI_v2(clean_audio, fs1, noisy_audio, fs2, HL, eq, Level1)
print(temp_HASQI)
# temp_HASQI = HASQI_v2(clean_audio, fs1, clean_audio, fs2, HL, eq, Level1)
# print(temp_HASQI)
temp_HASQI_ = HASQI_v2_for_unfixedLen(clean_audio, fs1, noisy_audio, fs2, HL, eq, Level1)
print(temp_HASQI_)
