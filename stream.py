import sys
from pathlib import Path
import librosa

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import torch
from model import CGRNNHS, CGRNNHS_stream

from utils.conv_stft import ConvSTFT
from utils.audiolib import audioread, to_frames, ola


def audio_to_frames(fname, nframe, nhop):
    data, fs = audioread(fname)
    frames, xk, N = to_frames(data, nframe, nhop)

    return frames, xk, data[:N]


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    nframe = 512
    nhop = 256
    net = CGRNNHS_stream(nframe, nhop)
    net_raw = CGRNNHS(nframe, nhop)
    ckpt = torch.load("./cgrnn_gan_best.pth")
    net.load_state_dict(ckpt)
    net.to(torch.device("cuda"))
    net.eval()
    net_raw.load_state_dict(ckpt)
    net_raw.to(torch.device("cuda"))
    net_raw.eval()
    fname = "/home/deepnetni/datasets/howling/test/4000.wav"

    frames, xk, data = audio_to_frames(fname, nframe, nhop)  # T,F
    xk = xk.transpose(1, 0)  # F,T
    xk = torch.from_numpy(xk).float().cuda()[None, ...]  # B,F,T
    state = None
    out = []

    # frame to frame
    nT = xk.shape[-1]
    for i in range(nT):
        dk = xk[:, :, i][..., None]  # B,F,1
        with torch.no_grad():
            ok, state = net(dk, state)  # B,F,1
        out.append(ok.cpu().squeeze().numpy())

    out = np.array(out)  # T,F
    # end to end
    # with torch.no_grad():
    #     out, state = net(xk)
    # print(out.shape)  # B,F,T
    # out = out.squeeze().cpu().numpy().transpose(1, 0)  # T,F
    out = ola(out, nframe, nhop)  # T
    plt.plot(out, alpha=0.1, color="b")

    # origin output
    data = torch.from_numpy(data).float().view(1, -1).cuda()

    with torch.no_grad():
        out_raw, state = net_raw(data)

    out_raw = out_raw.cpu().squeeze(dim=0).numpy()
    diff = np.sum(np.abs(out - out_raw))
    print(out.shape, out_raw.shape, diff.shape)

    plt.plot(out_raw, alpha=0.1, color="r")
    print(diff, np.allclose(out, out_raw, atol=1e-4, rtol=1e-5))

    # librosa_stft = librosa.stft(
    #     data,
    #     win_length=nframe,
    #     n_fft=256,
    #     hop_length=nhop,
    #     window="hann",
    #     center=True,
    # )
    # print(data.shape, librosa_stft.shape, xk.shape)
    # r, i = np.array_split(xk, 2, axis=-1)
    # spec = (r + 1j * i).transpose(1, 0)
    # print(spec.shape, librosa_stft.shape)
    # diff = np.sum((spec - librosa_stft) ** 2)
    # print("#", diff, np.allclose(spec, librosa_stft, atol=1e-6))

    sys.exit()

    data_ = ola(xk, nframe, nhop)
    diff = np.sum((data - data_) ** 2)
    print(diff, np.allclose(data, data_, atol=1e-6))

    layer = ConvSTFT(nframe=nframe, nhop=nhop, out_feature="complex")  # B,F,T
    out = layer(torch.from_numpy(data.reshape(1, -1)).float())

    out = out[0].permute(1, 0).numpy()
    print(xk.shape, data.shape, out.shape)
    diff = np.sum((out - xk) ** 2)
    print(diff, np.allclose(out, xk, atol=1e-6))
