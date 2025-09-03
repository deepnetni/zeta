import sys
import librosa
from matplotlib import use
import numpy as np
from pathlib import Path
import sounddevice as sd
import torch
import time
import threading

sys.path.append(str(Path(__file__).parent.parent))

from utils.audiolib import audioread, to_frames
from CGRNN_HS.model import CGRNNHS_stream
from typing import Dict
from scipy.signal import get_window
from utils.logger import get_logger

testf = "/home/deepnetni/datasets/howling/test/3998.wav"


def query_devices(device: int, kind) -> Dict:
    """
    kind: "input" or "output"
    """
    try:
        caps = sd.query_devices(device, kind=kind)
    except ValueError:
        message = bold(f"Invalid {kind} audio interface {device}.\n")
        message += (
            "If you are on Mac OS X, try installing Soundflower "
            "(https://github.com/mattingalls/Soundflower).\n"
            "You can list available interfaces with `python3 -m sounddevice` on Linux and OS X, "
            "and `python.exe -m sounddevice` on Windows. You must have at least one loopback "
            "audio interface to use this."
        )
        print(message, file=sys.stderr)
        sys.exit(1)
    return caps


def create_model(nframe=512, nhop=256):
    net = CGRNNHS_stream(nframe, nhop)
    ckpt = torch.load("../CGRNN_HS/cgrnn_gan_best.pth")
    net.load_state_dict(ckpt)
    net.to(torch.device("cuda"))
    net.eval()

    return net


# use_net = False
use_net = False


class DaemonTask(threading.Thread):
    def __init__(self, name: str, daemon: bool = True):
        super().__init__(name=name, daemon=daemon)

    def run(self):
        global use_net

        while True:
            flag = input("input 1 to enable net:")
            try:
                if int(flag) == 1:
                    print("open net")
                    use_net = True
                else:
                    print("close net")
                    use_net = False
            except:
                continue

            time.sleep(1)


if __name__ == "__main__":
    # data, fs = audioread(testf)
    # N = (len(data) // 512) * 512
    # data = data[:N]
    net = create_model()
    # start daemon thread
    th = DaemonTask(name="check_flag")

    state = None

    nframe = 512
    nhop = 256
    fs_in = 16000
    fs_out = 48000
    audio = np.zeros(nframe, dtype=np.float32)
    prev = np.zeros(nhop, dtype=np.float32)
    count_frame = 0

    win = get_window("hann", nframe, fftbins=True)
    div = win[-(nframe - nhop) :] ** 2 + win[: (nframe - nhop)] ** 2

    log = get_logger("live")
    log.info(f"start capture and process with block {1000*nhop / fs_in:.2f}ms")

    # in_device = 0
    # caps = query_devices(in_device, "input")
    # print("input:")
    # for k, v in caps.items():
    #     print(k, v)
    # in_channels = caps["max_input_channels"]
    # defaut_sample = caps["default_samplerate"]
    stream_in = sd.InputStream(device=None, samplerate=16000, channels=2)

    # out_device = 6
    # caps = query_devices(out_device, "output")
    # defaut_sample = caps["default_samplerate"]
    # print("output:")
    # for k, v in caps.items():
    #     print(k, v)
    # if defaut_sample != fs:
    #     data = librosa.resample(data, orig_sr=fs, target_sr=defaut_sample)
    stream_out = sd.OutputStream(device=None, samplerate=48000, channels=1)
    # stream_out = sd.OutputStream(device=None, samplerate=defaut_sample, channels=1)

    th.start()
    st = time.time()
    stream_in.start()
    stream_out.start()

    while True:
        try:
            audio[:-nhop] = audio[nhop:]
            data, overflow = stream_in.read(nhop)
            data = data[..., 0]

            if use_net:
                audio[-nhop:] = data

                xk = np.fft.rfft(audio * win, n=nframe, axis=-1)
                xk = np.concatenate([xk.real, xk.imag], axis=-1)
                xk = xk.reshape(1, -1, 1)  # B(1),F,T(1)

                inp = torch.from_numpy(xk).float().cuda()  # B,F,T

                with torch.no_grad():
                    outk, state = net(inp, state)  # B,F,T

                xk = outk.squeeze().cpu().numpy()
                r, i = np.array_split(xk, 2, axis=-1)
                xk = r + 1j * i

                xt = np.real(np.fft.irfft(xk, axis=-1)) * win
                out = ((prev + xt[:nhop]) / div).astype(np.float32)
                prev = xt[-nhop:]
            else:
                out = data

            audio_post = librosa.resample(out, orig_sr=fs_in, target_sr=fs_out)
            underflow = stream_out.write(audio_post)
            count_frame += 1

            if overflow or underflow:
                log.warn(
                    f"Not processing audio fast enough, time per frame is {1000*(time.time()-st)/count_frame:.2f}ms,"
                    f"should be less than {1000*nhop/fs_in:.2f}ms"
                )
        except KeyboardInterrupt:
            log.info(f"Stopping")
            # th.join()  # the daemon threading will exit when it's the last existing thread.
            break

    # stream_out.start()
    # try:
    #     for d in data.reshape(-1, 512):
    #         ret = stream_out.write(d)
    # except Exception as e:
    #     print(e)

    stream_out.close()
    stream_in.close()
    print("end")

    # stream = p.open(
    #     channels=1,
    #     format=pyaudio.paFloat32,
    #     rate=fs,
    #     frames_per_buffer=4096,
    #     output=True,
    # )

    # stream.close()
    # p.terminate()
