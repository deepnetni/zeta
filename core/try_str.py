import os
import rir_generator as rir
import numpy as np
from matplotlib import pyplot as plt
import pickle

room = [10, 10, 5]
a = [3, 3, 3]
b = [3.2, 3.2, 3]
c = [3.1, 3.3, 3]
d = [3.22, 3.2, 3]
e = [3.25, 3.3, 3]

h1 = rir.generate(
    c=340,  # sound velocity
    fs=16000,
    L=room,  # room dimensions [x,y,z](m)
    s=a,  # source position
    r=b,  # receiver position [x,y,z](m)
    beta=[0.62] * 6,
    nsample=64,  # number of output samples
    order=10,
)  # (samples, mics)
h2 = rir.generate(
    c=340,  # sound velocity
    fs=16000,
    L=room,  # room dimensions [x,y,z](m)
    s=a,  # source position
    r=c,  # receiver position [x,y,z](m)
    beta=[0.62] * 6,
    nsample=64,  # number of output samples
    order=10,
)  # (samples, mics)
h3 = rir.generate(
    c=340,  # sound velocity
    fs=16000,
    L=room,  # room dimensions [x,y,z](m)
    s=b,  # source position
    r=c,  # receiver position [x,y,z](m)
    beta=[0.62] * 6,
    nsample=64,  # number of output samples
    order=10,
)  # (samples, mics)

h4 = rir.generate(
    c=340,  # sound velocity
    fs=16000,
    L=room,  # room dimensions [x,y,z](m)
    s=a,  # source position
    r=d,  # receiver position [x,y,z](m)
    beta=[0.62] * 6,
    nsample=64,  # number of output samples
    order=10,
)  # (samples, mics)
h5 = rir.generate(
    c=340,  # sound velocity
    fs=16000,
    L=room,  # room dimensions [x,y,z](m)
    s=a,  # source position
    r=e,  # receiver position [x,y,z](m)
    beta=[0.62] * 6,
    nsample=64,  # number of output samples
    order=10,
)  # (samples, mics)
h6 = rir.generate(
    c=340,  # sound velocity
    fs=16000,
    L=room,  # room dimensions [x,y,z](m)
    s=d,  # source position
    r=e,  # receiver position [x,y,z](m)
    beta=[0.62] * 6,
    nsample=64,  # number of output samples
    order=10,
)  # (samples, mics)

h = np.concatenate([h1, h2, h3], axis=-1)

np.savetxt("ab.txt", h1.squeeze(), fmt="%.6f")
np.savetxt("ac.txt", h2.squeeze(), fmt="%.6f")
np.savetxt("bc.txt", h3.squeeze(), fmt="%.6f")
np.savetxt("ad.txt", h4.squeeze(), fmt="%.6f")
np.savetxt("ae.txt", h5.squeeze(), fmt="%.6f")
np.savetxt("de.txt", h6.squeeze(), fmt="%.6f")
plt.subplot(611)
plt.plot(h1.squeeze())
plt.subplot(312)
plt.plot(h2.squeeze())
plt.subplot(313)
plt.plot(h3.squeeze())
plt.savefig("a.svg")


a = np.zeros((2, 3))
print(a.shape)
