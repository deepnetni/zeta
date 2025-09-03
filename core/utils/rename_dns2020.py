import os
import numpy as np
import shutil
import re


if __name__ == "__main__":
    dirp = "/home/deepni/disk/ArchiveDataset/DNS-Challenges/2020/synthesis/train/clean"
    files = os.listdir(dirp)

    dirp_ = "/home/deepni/disk/ArchiveDataset/DNS-Challenges/2020/synthesis/dev/clean"

    f = files[0]
    print(f)

    for f in files:
        ret = re.sub(r"(fileid.*).wav", r"noisy_\3.wav", f)
        os.rename(os.path.join(dirp, f), os.path.join(dirp_, ret))
