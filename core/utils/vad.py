import webrtcvad
import numpy as np
from collections import deque


class VAD:
    def __init__(self, blk_ms=10, fs=16000, level=3):
        """
        level: range from 0 to 3, while 0 is the least aggressive about filtering out non-speech.
            3 is the most aggresive to label speech.
        blk_len should be 10, 20, 30ms
        """
        self.fs = fs
        self.block_len = blk_ms * fs // 1000
        self.block_bytes = int(2 * self.block_len)  # only support int16 dtype
        self.web_vad = webrtcvad.Vad(level)
        # self.web_vad.set_mode(level)
        self.history = deque(maxlen=128)
        self.active = False
        self.data = b""
        self.detect_blks = 4

    def is_speech(self, data: np.ndarray):
        """make sure input one block at a time.
        data should be (N,)
        """
        if data.ndim == 2:
            data = data.squeeze()
        elif data.ndim > 2:
            raise RuntimeError()

        self.data += data.tobytes()

        while len(self.data) >= self.block_bytes:
            # NOTE: break out when activate, ignoring the precedding input data.
            blk = self.data[: self.block_bytes]
            self.data = self.data[self.block_bytes :]

            if self.web_vad.is_speech(blk, self.fs) is True:
                self.history.append(1)
            else:
                self.history.append(0)

            active_num = 0
            # make desicion when got at least 8 blks
            # -1 index refer to the last append value
            for i in range(-self.detect_blks, 0):
                try:
                    active_num += self.history[i]
                except IndexError:
                    continue

            if not self.active:
                if active_num >= self.detect_blks // 2:
                    self.active = True
                    break
                elif len(self.history) == self.history.maxlen and sum(self.history) == 0:
                    # drop half history when full
                    for _ in range(int(self.history.maxlen / 2)):
                        self.history.popleft()
            else:  # activate
                # if active_num == 0:
                if active_num < self.detect_blks // 2:
                    self.active = False
                elif sum(self.history) > self.history.maxlen * 0.9:
                    # drop half history when over 90% activate frames
                    for _ in range(int(self.history.maxlen / 2)):
                        self.history.popleft()

        return self.active

    def vad_waves(self, data):
        """
        NOTE: couldn't input multi-samples at one time
        data: 1,T or T,

        return T_,
        """
        if data.ndim == 2:
            data = data.squeeze()
        elif data.ndim > 2:
            raise RuntimeError()

        N = len(data)
        frames = N // self.block_len

        vad_values = []
        for i in range(frames):
            st = i * self.block_len
            inp = data[st : st + self.block_len]
            inp = inp * 32767
            inp = inp.astype(np.int16)
            act = self.is_speech(inp)
            if act:
                vad_values.extend([0.95] * self.block_len)
            else:
                vad_values.extend([0] * self.block_len)

        return np.array(vad_values)

    def vad_frames(self, data):
        """
        NOTE: couldn't input multi-samples at one time
        data: 1,T or T,

        return F,
        """
        if data.ndim == 2:
            data = data.squeeze()
        elif data.ndim > 2:
            raise RuntimeError()

        N = len(data)
        frames = N // self.block_len

        vad_values = []
        for i in range(frames):
            st = i * self.block_len
            inp = data[st : st + self.block_len]
            inp = inp * 32767
            inp = inp.astype(np.int16)
            act = self.is_speech(inp)
            if act:
                vad_values.extend([1])
            else:
                vad_values.extend([0])

        return np.array(vad_values)

    def reset(self):
        self.data = b""
        self.active = False
        self.history.clear()


if __name__ == "__main__":
    import soundfile as sf

    # f = "/home/ll/datasets/blind_test_set/clean/-3sybEBJmEC8P7T6LAFqoA_doubletalk_mic.wav"
    # inp, fs = sf.read(f)
    # inp = inp * 32768
    # inp = inp.astype(np.int16)
    # print(inp.dtype)
    # inp = np.random.randn(1, 160)
    inp = np.zeros([1, 160])
    print(inp.shape)
    v = VAD(blk_ms=10)

    vads = v.vad_waves(inp)
    print(vads.shape, vads)
    # for st in range(1, 10000, 10):
    #     out = v.is_speech(inp[:, st : st + 10])
    #     print(out)
