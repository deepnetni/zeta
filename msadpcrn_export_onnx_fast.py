import sys
import torch
import torch.onnx as onnx
from core.MSA_DPCRN_FAST import MSA_DPCRN_SPEC_ALIGN, MSA_DPCRN_SPEC_ALIGN_onnx, MSA_DPCRN_ALIGN
import onnxruntime as ort
from tqdm import tqdm
import numpy as np
from core.models.conv_stft import STFT
from core.utils.gcc_phat import gcc_phat
from core.utils.audiolib import audioread, audiowrite

print(onnx.producer_version)


def original():
    net = MSA_DPCRN_ALIGN(
        nframe=512,
        nhop=256,
        nfft=512,
        cnn_num=[16, 32, 64, 64],
        stride=[2, 2, 1, 1],
        rnn_hidden_num=64,
    )

    ckpt = r"E:\model_results_trunk\AEC\trained_msadpcrn_align\msa_dpcrn\checkpoints\epoch_0002.pth"

    net.load_state_dict(torch.load(ckpt)["net"])
    net.eval()

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

    # stft = STFT(nframe=512, nhop=256, win="hann sqrt")
    stft = STFT(nframe=512, nhop=256, win="hann")

    mic = torch.from_numpy(mic).float()[None, ...]
    ref = torch.from_numpy(ref).float()[None, ...]

    with torch.no_grad():
        out = net(mic, ref)
    out = out.cpu().detach().squeeze().numpy()
    audiowrite("aec_test/onnx_out_original.wav", out, 16000)
    return out


def e2e():
    net = MSA_DPCRN_SPEC_ALIGN(
        nframe=512,
        nhop=256,
        nfft=512,
        cnn_num=[16, 32, 64, 64],
        stride=[2, 2, 1, 1],
        rnn_hidden_num=64,
    )
    ckpt = r"E:\model_results_trunk\AEC\trained_msadpcrn\msa_dpcrn_mstft_pmsqe\checkpoints\epoch_0100.pth"

    net.load_state_dict(torch.load(ckpt)["net"])
    net.eval()

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

    stft = STFT(nframe=512, nhop=256)

    mic = torch.from_numpy(mic).float()[None, ...]
    ref = torch.from_numpy(ref).float()[None, ...]
    mic = stft.transform(mic)
    ref = stft.transform(ref)

    with torch.no_grad():
        out = net(mic, ref)
    out = stft.inverse(out).cpu().detach()  # B,T
    out = out.squeeze().numpy()
    # audiowrite("aec_test/onnx_out_e2e.wav", out, 16000)
    return out


def to_onnx():
    net = MSA_DPCRN_SPEC_ALIGN_onnx(
        nframe=512,
        nhop=256,
        nfft=512,
        cnn_num=[16, 32, 64, 64],
        stride=[2, 2, 1, 1],
        rnn_hidden_num=64,
    )
    # ckpt = r"msa_dpcrn_epoch_0100.pth"
    ckpt = r"E:\model_results_trunk\AEC\trained_msadpcrn_align\msa_dpcrn\checkpoints\epoch_0002.pth"

    net.load_state_dict(torch.load(ckpt)["net"])
    net.eval()

    mic = torch.randn(1, 2, 1, 257)
    ref = torch.randn(1, 2, 1, 257)

    h = [
        torch.zeros(4, 1, 64, 64),  # 65 is the freq, 64 is the hidden size
        torch.zeros(4, 1, 64, 64),
    ]

    onnx.export(
        net,
        # (mic, ref, h[0], h[1], h[2], h[3]),
        (mic, ref, *h),
        "msa_dpcrn_fast.onnx",
        input_names=["mic", "ref", "h", "c"],
        output_names=["spec", "ho", "co"],
    )
    run_sess = ort.InferenceSession("msa_dpcrn_fast.onnx")
    inputs = {inp.name for inp in run_sess.get_inputs()}
    print(inputs)


def inference():
    # mic = torch.randn(1, 2, 2, 65)
    # ref = torch.randn(1, 2, 2, 65)
    run_sess = ort.InferenceSession("msa_dpcrn_new.onnx")
    inputs = {inp.name for inp in run_sess.get_inputs()}
    print(inputs)

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

    stft = STFT(nframe=512, nhop=256)

    mic = torch.from_numpy(mic).float()[None, ...]
    ref = torch.from_numpy(ref).float()[None, ...]
    mic = stft.transform(mic)
    ref = stft.transform(ref)
    # print(mic.shape)
    out_list = []
    state = [
        np.zeros([4, 1, 64, 64], dtype=np.float),
        np.zeros([4, 1, 64, 64], dtype=np.float),
    ]
    for nt in tqdm(range(mic.size(2))):
        mic_frame = mic[..., (nt,), :]
        ref_frame = ref[..., (nt,), :]
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
        state = [ort_out[i] for i in range(1, 3)]
        out_list.append(torch.from_numpy(ort_out[0]))

    out = torch.concat(out_list, dim=2)
    out = stft.inverse(out).cpu()  # B,T
    out = torch.clamp(out, -1, 1)
    out = out.squeeze().numpy()
    audiowrite("aec_test/onnx_out.wav", out, 16000)
    return out


if __name__ == "__main__":
    to_onnx()
    # o1 = inference()
    # o2 = e2e()
    # o3 = original()
    # N = min(len(o1), len(o3))
    # diff = np.sum(np.abs(o3[:N] - o1[:N]))
    print("done")
