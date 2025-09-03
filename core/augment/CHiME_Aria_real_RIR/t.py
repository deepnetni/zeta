import pickle
import matplotlib.pyplot as plt


# with open("./GICWmAAnG30z7tkWAGvSvfTwTk0PbsYvAABk.pkl", "rb") as f:
with open("./GICWmAD5nKGkVzcFANz0rJOtba0JbsYvAABk.pkl", "rb") as f:
    data = pickle.load(f)


for k, v in data.items():
    print(k)

print("#", data["mic_geometry"], data["room_dim"])
# print(data["mic_to_noise_source"].keys())
# r = data["mic_to_noise_source"]["mic_noise_source5"]["rir"]
print(data["mic_to_mouth_source"].keys())
r = data["mic_to_mouth_source"]["mic_mouth_source1"]["rir"]
plt.plot(r[1, :])
print(r.max(), len(r), r.shape)
plt.savefig("r.svg")
