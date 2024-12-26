import sys
import torch
import torch.onnx as onnx
from models.MSA_DPCRN import MSA_DPCRN_SPEC, MSA_DPCRN_SPEC_onnx
import onnxruntime as ort
from tqdm import tqdm
import numpy as np
from models.PQMF import PQMF
from models.conv_stft import STFT
from utils.gcc_phat import gcc_phat
from utils.audiolib import audioread, audiowrite

print(onnx.producer_version)


def to_onnx():
    net = MSA_DPCRN_SPEC_onnx(
        nframe=128,
        nhop=64,
        nfft=128,
        cnn_num=[16, 32, 64],
        stride=[2, 2, 1],
        rnn_hidden_num=64,
    )
    ckpt = r"msa_dpcrn_epoch_0100.pth"

    net.load_state_dict(torch.load(ckpt))
    net.eval()

    mic = torch.randn(1, 2, 1, 65)
    ref = torch.randn(1, 2, 1, 65)

    h = [
        torch.zeros(4, 1, 17, 64),
        torch.zeros(4, 1, 17, 64),
    ]

    onnx.export(
        net,
        # (mic, ref, h[0], h[1], h[2], h[3]),
        (mic, ref, *h),
        "msa_dpcrn.onnx",
        input_names=["mic", "ref", "h", "c"],
        output_names=["spec", "w", "ho", "co"],
    )
    run_sess = ort.InferenceSession("msa_dpcrn.onnx")
    inputs = {inp.name for inp in run_sess.get_inputs()}
    print(inputs)


def inference():
    # mic = torch.randn(1, 2, 2, 65)
    # ref = torch.randn(1, 2, 2, 65)
    run_sess = ort.InferenceSession("msa_dpcrn.onnx")
    inputs = {inp.name for inp in run_sess.get_inputs()}
    print(inputs)

    qmf = PQMF(2)
    mic, _ = audioread("aec_test/mic_3.wav")
    ref, _ = audioread("aec_test/ref_3.wav")

    align = True
    if align:
        fs = 16000
        tau, _ = gcc_phat(mic, ref, fs=fs, interp=1)
        tau = max(0, int((tau - 0.001) * fs))
        ref = np.concatenate([np.zeros(tau), ref], axis=-1, dtype=np.float32)[: mic.shape[-1]]
    else:
        N = min(len(mic), len(ref))
        N = 16000 * 5
        mic = mic[:N]
        ref = ref[:N]

    stft = STFT(nframe=128, nhop=64, win="hann sqrt")
    # B,2,T
    mic_lh = qmf.analysis(torch.from_numpy(mic).float())
    mic_l, mic_h = mic_lh[:, 0, ...], mic_lh[:, 1, ...]
    # B,T; B,T//2
    # print(mic.shape, mic_l.shape)
    ref_lh = qmf.analysis(torch.from_numpy(ref).float())
    ref_l, ref_h = ref_lh[:, 0, ...], ref_lh[:, 1, ...]

    mic = stft.transform(mic_l)
    ref = stft.transform(ref_l)

    out_list = []
    state = [
        np.zeros([4, 1, 17, 64], dtype=np.float32),
        np.zeros([4, 1, 17, 64], dtype=np.float32),
    ]
    for nt in tqdm(range(mic.size(2))):
        mic_frame = mic[..., nt, :].unsqueeze(2)
        ref_frame = ref[..., nt, :].unsqueeze(2)
        ort_out = run_sess.run(
            None,
            {
                "mic": mic_frame.numpy(),
                "ref": ref_frame.numpy(),
                "h": state[0],
                "c": state[1],
            },
        )
        # for i, o in enumerate(ort_out):
        #     print(i, o.shape)
        state = [ort_out[i] for i in range(2, 4)]
        out_list.append(torch.from_numpy(ort_out[0]))

    out = torch.concat(out_list, dim=2)
    out = stft.inverse(out).cpu()  # B,T
    out = out.squeeze().numpy()
    audiowrite("aec_test/onnx_out.wav", out, 8000)


if __name__ == "__main__":
    # to_onnx()
    inference()
    print("done")
