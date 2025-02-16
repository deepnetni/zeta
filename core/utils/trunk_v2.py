import csv
import os
import re
import sys
from pathlib import Path

import librosa
import soundfile as sf

sys.path.append(str(Path(__file__).parent.parent))

import ast
import json
import random
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from models.conv_stft import STFT
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.audiolib import audioread
from utils.gcc_phat import gcc_phat
from utils.logger import get_logger
from utils.register import tables


def clip_to_shortest(batch: List):
    """TODO
    comming soon.
    """

    batch.sort(key=lambda x: x[0].shape[-1], reverse=True)

    for x in batch:
        print(x[0].shape, x[1].shape)


def pad_to_longest(batch):
    """
    batch: [(data, label), (...), ...], B,T,C
    the input data, label must with shape (T,C) if time domain
    """
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)  # data length

    seq_len = [d.size(0) for d, _ in batch]
    data, label = zip(*batch)  # B,T,C
    data = pad_sequence(data, batch_first=True).float()
    label = pad_sequence(label, batch_first=True).float()

    # data = pack_padded_sequence(data, seq_len, batch_first=True, enforce_sorted=True)

    return data, label, torch.tensor(seq_len)


def pad_to_longest_aec(batch):
    """
    batch: [(mic, ref, label), (...), ...], B,T,C
    the input data, label must with shape (T,C) if time domain
    """
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)  # data length

    seq_len = [d.size(0) for d, _, _ in batch]
    mic, ref, label = zip(*batch)  # B,T,C
    mic = pad_sequence(mic, batch_first=True).float()
    ref = pad_sequence(ref, batch_first=True).float()
    label = pad_sequence(label, batch_first=True).float()

    # data = pack_padded_sequence(data, seq_len, batch_first=True, enforce_sorted=True)

    return mic, ref, label, torch.tensor(seq_len)


class TrunkBasic(Dataset):
    """Dataset base class
    Search microphone wav files by pattern.
    """

    def __init__(
        self,
        dirname: str,
        pattern: str = "**/*.wav",
        clean_dirname: Optional[str] = None,
        flist: Optional[str] = None,
        nlen: float = 0.0,
        min_len: float = 0.0,
        fs: int = 16000,
        seed: Optional[int] = None,
        csv_dir: str = __file__.rsplit("/", 3)[0] + "/manifest",
        keymap: Optional[Tuple[str, ...]] = None,
        **kwargs,
    ):
        super().__init__()
        across_files = kwargs.get("across_files", False)

        self.dir = Path(dirname)
        self.clean_dir = Path(clean_dirname) if clean_dirname is not None else Path(dirname)
        # self.logger = get_logger(dirname)
        self.logger = get_logger(self.__class__.__name__)
        self.csv_dir = csv_dir
        self.keymap = keymap

        if flist is None:
            flist = os.path.join(csv_dir, os.path.split(dirname)[-1] + ".csv")
        elif os.path.isabs(flist):  # abspath
            flist = flist
        else:  # relative path
            flist = os.path.join(csv_dir, flist)

        self.N = int(nlen * fs)
        self.minN = int(min_len * fs)
        self.fs = fs

        mic_list = list(map(str, self.dir.rglob(pattern)))

        if os.path.exists(flist):
            f_pairs = self.load_f_list(flist, (str(self.dir), str(self.clean_dir)))
        else:
            f_pairs = self._mapping_pair(mic_list)
            if len(f_pairs) == 0:
                raise RuntimeError("Empty wav files.")
            self.save_f_list(flist, f_pairs, (str(self.dir), str(self.clean_dir)))

        if across_files:
            self.f_list = self._rearange_across_files(f_pairs)
        else:
            self.f_list = self._rearange(f_pairs)

        if seed is not None:
            random.seed(seed)
            random.shuffle(self.f_list)

        self.logger.info(f"Loading {dirname} {len(self.f_list)} files.")

    @property
    def dirname(self):
        return str(self.dir)

    def _rearange_across_files(self, flist):
        """split audios through wave file length
        flist: [((f1, f2,...,f_target), N), (...)]

        return: multi-files will combined into a list if file length is not enought.
            e.g., [[('f1':x,'st':x,'end','pad',x),('f2':x,'st':x,'end','pad',x),... ], [('f3':x,'st':x,'end','pad',x)]]
        """
        f_list = []
        buffer = []
        buffer_len = 0

        for f, nlen in flist:
            if self.N != 0:
                st, end = 0, int(nlen)
                remain_N = buffer_len + end - st

                while remain_N >= self.N:
                    if len(buffer) != 0:
                        N = self.N - buffer_len
                        f_list.append([*buffer, {"f": f, "start": st, "end": st + N, "pad": 0}])
                        buffer.clear()
                        buffer_len = 0
                    else:
                        N = min(self.N, end - st)
                        f_list.append([{"f": f, "start": st, "end": st + N, "pad": 0}])

                    st += N
                    remain_N -= N

                if end - st > self.minN:
                    buffer.append({"f": f, "start": st, "end": end, "pad": 0})
                    buffer_len += end - st

            else:  # self.N == 0
                f_list.append({"f": f, "start": 0, "end": int(nlen), "pad": 0})

        return f_list

    def _rearange(self, flist):
        """split audios through wave file length
        flist: [((f1, f2,...,f_target), N), (...)]
        """
        f_list = []

        for f, nlen in flist:
            if self.N != 0 and self.minN != 0:
                st = 0
                nlen = int(nlen)
                if nlen < self.minN:
                    continue

                while nlen - st >= self.N:
                    f_list.append({"f": f, "start": st, "end": st + self.N, "pad": 0})
                    st += self.N

                if nlen - st >= self.minN:
                    f_list.append({"f": f, "start": st, "end": nlen, "pad": self.N - (nlen - st)})
            else:
                f_list.append({"f": f, "start": 0, "end": int(nlen), "pad": 0})

        return f_list

    def load_f_list(
        self,
        fname: str,
        relative: Union[str, Tuple[str, ...], None] = None,
    ) -> List:
        """
        relative: str or tuple of str each corresponding to element in the file.
        return: [([f1, f2, ..., fn], N), (...)]
        """
        f_list = []
        with open(fname, "r+") as fp:
            ctx = csv.reader(fp)
            for element in ctx:
                element, num = element[:-1], element[-1]
                if isinstance(relative, tuple):
                    f_items = list(
                        map(
                            lambda x, y: os.path.join(x, y.replace("\\", "/")),
                            relative,
                            element,
                        )
                    )
                elif relative is not None:
                    f_items = list(
                        map(
                            lambda x: os.path.join(relative, x.replace("\\", "/")),
                            element,
                        )
                    )

                else:  # relative is None
                    f_items = element

                # [([...], num), ([...], num), ...]
                f_list.append((f_items, num))

        return f_list

    def save_f_list(
        self,
        fname: str,
        f_list: List,
        relative: Union[str, Tuple[str, ...], None] = None,
    ):
        """
        fname: csv file contains the training files path
        f_list:  [((f1, f2, ..., fn), L), ...]
        relative: str or tuple of str each corresponding to element in f_list
        """
        dirname = os.path.dirname(fname)
        os.makedirs(dirname) if not os.path.exists(dirname) else None

        with open(fname, "w+", newline="") as fp:
            writer = csv.writer(fp)
            for f, num in f_list:
                if isinstance(relative, tuple):
                    line = tuple(
                        list(map(lambda x, y: os.path.relpath(x, y), f, relative)) + [num]
                    )  # ([f1,f2,...,fn,N])
                elif relative is not None:  # single str
                    line = tuple(list(map(lambda x: os.path.relpath(x, relative), f)) + [num])
                else:  # relative is None
                    line = f
                # ([f1,f2,...,f_target, num])
                writer.writerow(line)

    def __len__(self):
        return len(self.f_list)

    def __iter__(self):
        self.pick_idx = 0
        return self

    def _mapping_pair(self, mic_list):
        """Rewrite according your dataset structure.
        search the target wavs by mic.

        return a list [((mic, mic2, ..., clean), N), ...].
        """
        element = []
        for f_mic in mic_list:
            _, f_mic_name = os.path.split(f_mic)
            f_sph = f_mic_name.replace(*self.keymap) if self.keymap is not None else f_mic_name
            f_sph = os.path.join(self.clean_dir, f_sph)

            dmic, _ = audioread(f_mic)
            element.append(((f_mic, f_sph), len(dmic)))
        return element


class FIG6Trunk(TrunkBasic):
    def __init__(
        self,
        dirname: str,
        pattern: str = "**/*.wav",
        clean_dirname: Optional[str] = None,
        flist: Optional[str] = None,
        nlen: float = 0,
        min_len: float = 0,
        fs: int = 16000,
        seed: Optional[int] = None,
        csv_dir: str = __file__.rsplit("/", 3)[0] + "/manifest",
        keymap: Optional[Tuple[str, ...]] = None,
        **kwargs,
    ):
        super().__init__(
            dirname,
            pattern,
            clean_dirname,
            flist,
            nlen,
            min_len,
            fs,
            seed,
            csv_dir,
            keymap,
            **kwargs,
        )

        self.load_vad = kwargs.get("vad", False)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        el = self.f_list[index]
        f_mic, f_sph = el["f"]
        st, ed, pd = el["start"], el["end"], el["pad"]

        hl_f = re.sub(r"(\w*)_nearend.wav", r"\1.json", f_mic)
        with open(hl_f, "r") as fp:
            ctx = json.load(fp)
            hl = ast.literal_eval(ctx["HL"])

        d_mic, fs_1 = audioread(f_mic, sub_mean=True)
        d_sph, fs_2 = audioread(f_sph)  # T,2
        # d_sph, fs_2 = sf.read(f_sph)  # T,2
        assert fs_1 == fs_2

        d_mic = np.pad(d_mic[st:ed], (0, pd), "constant", constant_values=0)
        # padding inner first
        d_sph = np.pad(d_sph[st:ed, :], ((0, pd), (0, 0)), "constant", constant_values=0)

        if not self.load_vad:
            # f_vad = re.sub(r"(\w*)_nearend.wav", r"\1_vad.wav", f_mic)
            # d_vad, _ = audioread(f_vad, sub_mean=False)
            # d_sph = np.stack([d_sph, d_vad], axis=-1)  # T,2
            d_sph = d_sph[..., 0]

        return (
            torch.from_numpy(d_mic).float(),
            torch.from_numpy(d_sph).float(),
            torch.tensor(hl).float(),
        )

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """used for predict api
        fname should contain the dataset name.
        return: data, relative path
        """
        if self.pick_idx < len(self.f_list):
            el = self.f_list[self.pick_idx]
            f_mic, f_sph = el["f"]
            # st, ed, pd = el["start"], el["end"], el["pad"]
            hl_f = re.sub(r"(\w*)_nearend.wav", r"\1.json", f_mic)
            with open(hl_f, "r") as fp:
                ctx = json.load(fp)
                hl = ast.literal_eval(ctx["HL"])

            d_mic, _ = audioread(f_mic, sub_mean=True)
            fname = str(Path(f_sph).relative_to(self.clean_dir.parent))

            self.pick_idx += 1

            return (
                torch.from_numpy(d_mic).float()[None, :],
                torch.tensor(hl).float()[None, :],
                fname,
            )
        else:
            raise StopIteration


if __name__ == "__main__":
    # from torchmetrics.functional.audio import signal_noise_ratio as SDR

    dset = FIG6Trunk(
        dirname="/home/deepnetni/trunk/dns_wdrc/dev",
        flist="../manifest/fig6_sig_dev.csv",
        pattern="[!.]*_nearend.wav",
        keymap=("nearend.wav", "target.wav"),
        # vad=True,
        vad=True,
    )

    mic, sph, hl = dset[0]
    print(mic.shape, hl.shape, sph.shape)
    sys.exit()

    # a = [
    #     (("1.wav", "b.wav"), 10),
    #     (("2.wav", "b.wav"), 10),
    #     (("3.wav", "b.wav"), 10),
    #     (("4.wav", "b.wav"), 10),
    # ]

    # o = dset._rearange_across_files(a)
    # print(o)
