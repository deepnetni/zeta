import numpy as np


freq = np.arange(0, 8000 + 1, 100)
print(freq)

fc = np.array([0, 250, 500, 750, 1375, 2500, 3500, 4875, 8000])
print(fc)


for i, (low, high) in enumerate(zip(fc[:-1], fc[1:])):
    idx = np.where((freq >= low) & (freq < high))
    print(idx)
