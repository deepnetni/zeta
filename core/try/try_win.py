from scipy.signal import get_window
import numpy as np


win = get_window("hann", 64, fftbins=True)
win = np.sqrt(win)
print(win)

np.savetxt("window64w32.txt", (win,), fmt="%.8f", delimiter=",")
