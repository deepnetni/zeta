import sys
import numpy as np
import onnxruntime as ort

# import torch
# import torch.onnx as onnx
# from tqdm import tqdm
# from core.models.VADModel import CRNN_VAD_new_onnx
from core.utils.audiolib import audioread, audiowrite
from pathlib import Path
from scipy.signal import get_window

# print(onnx.producer_version)


# def to_onnx():
#     net = CRNN_VAD_new_onnx(257)
#     ckpt = r"/home/deepni/model_results_trunk/CRY/trained_infant_vad/crnn_vad/checkpoints/epoch_0100.pth"

#     # ckpt = r"/home/deepni/model_results_trunk/CRY/trained_infant_noise_vad/crnn_vad/checkpoints/epoch_0055.pth"
#     net.load_state_dict(torch.load(ckpt)["net"])
#     net.eval()

#     mic = torch.randn(1, 2, 1, 257)

#     h = [torch.zeros(2, 1, 128)]

#     onnx.export(
#         net,
#         (mic, *h),
#         "infant_vad.onnx",
#         input_names=["mic", "state"],
#         output_names=["vad", "state_"],
#     )
#     run_sess = ort.InferenceSession("infant_vad.onnx")
#     inputs = {inp.name for inp in run_sess.get_inputs()}
#     print(inputs)


# def check_xk(data, win):
#     import librosa

#     xk_stft = librosa.stft(  # B,F,T
#         data,
#         # win_length=512,
#         n_fft=512,
#         hop_length=256,
#         window=win,
#         center=True,
#     )

#     print(xk_stft.shape)
#     return xk_stft


def inference():
    run_sess = ort.InferenceSession("infant_vad.onnx")
    inputs = {inp.name for inp in run_sess.get_inputs()}
    print(inputs)

    nframe = 512
    nhop = 256
    # mic, _ = audioread(Path.home() / "trunk/cry/test/awake_149.wav")
    mic, _ = audioread("333.wav")
    win = np.sqrt(get_window("hann", nframe, fftbins=True))

    N = len(mic)
    data = np.zeros([1, nframe]).astype(np.float32)
    Nframe = N // nhop

    # xk_stft = check_xk(mic, win)  # F,T
    # print(xk_stft[128 : 128 + 10, 1])

    out_list = []
    state = np.zeros([2, 1, 128], dtype=np.float32)

    for t in range(Nframe):
        d = mic[t * nhop : (t + 1) * nhop][None, :]  # B(1),nhop
        data = np.concatenate([data[:, nhop:], d], axis=-1)  # B(1),D(512)
        xk = np.fft.rfft(data * win, n=nframe, axis=-1)  # B(1),F(257)

        # complex to float32 with shape b(1),2,t(1),f
        xk_inp = np.stack([xk.real, xk.imag], axis=1)[:, :, None, :]
        xk_inp = xk_inp.astype(np.float32)

        ort_out = run_sess.run(
            None,
            {
                "mic": xk_inp,
                "state": state,
            },
        )
        # output B(1),T(1),1
        state = ort_out[1]

        out_list.append(ort_out[0])

    out = np.concatenate(out_list, axis=1).squeeze()  # T,
    out = out.repeat(nhop)
    out_hard = np.where(out > 0.75, 0.95, 0)
    N = min(len(out), len(mic))

    audiowrite(
        "onnx_vad_out.wav", np.stack([mic[:N], out_hard[:N], out[:N]], axis=-1), 16000
    )


if __name__ == "__main__":
    # to_onnx()
    inference()
    print("done")
