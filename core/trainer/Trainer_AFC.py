import os
import pickle
import sys
from typing import Dict, Optional, Tuple

# adding xx/core
# sys.path.append(__file__.rsplit("/", 2)[0])
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional.audio.sdr import signal_distortion_ratio as SDR
from tqdm import tqdm
from core.utils.audiolib_pt import AcousticFeedbackSim

from utils.check_flops import check_flops
from utils.focal_loss import BCEFocalLoss
from utils.HAids.PyFIG6.pyFIG6 import FIG6_compensation_vad
from utils.HAids.PyHASQI.preset_parameters import generate_filter_params
from utils.losses import loss_pmsqe
from utils.record import REC
from Engine import Engine
from utils.audiolib import audiowrite

# os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# torch.autograd.set_detect_anomaly(True)


class Trainer(Engine):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        with open("out.pkl", "rb") as f:
            meta = pickle.load(f)

        self.nblk = 64

    def _fit_generator_step(self, mic, rir, G):
        N = mic.shape[-1]
        nB = mic.shape[0]
        idx = torch.arange(0, N - self.nblk + 1, self.nblk).reshape(-1, 1) + torch.arange(
            self.nblk
        )  # nF, nD

        mic_frames = mic[:, idx]  # b,t,d
        # 0,0 D; 0, 1 T; 0,0 B
        mic_frames = F.pad(mic_frames, (0, 0, 0, 1, 0, 0))

        fb_data = torch.zeros(nB, self.nblk).float().to(mic.device)
        ref = torch.zeros(nB, self.nblk).float().to(mic.device)
        FB = AcousticFeedbackSim(rir, self.nblk).to(rir.device)

        out_ = []
        h_stat = None
        self.net.reset_buff(mic)
        for i in range(mic_frames.shape[1]):
            d = mic_frames[:, i, :]  # nB,nD

            mix = d + fb_data
            if self.kwargs.get("woRef", False):
                enh, h_stat = self.net(mix, h_stat, online=True)  # B,T
            else:
                enh, h_stat = self.net(mix, ref, h_stat, online=True)  # B,T

            ref = enh.detach() * G
            fb_data = FB(ref)

            out_.append(enh)

        N = (mic_frames.shape[1] - 1) * self.nblk
        out = torch.concat(out_[1:], dim=-1)
        # ! OLA will padding one frame.
        loss_dict = self.loss_fn_list(mic[..., :N], out)
        loss = loss_dict["loss"]

        return out.detach(), loss, loss_dict

    def _fit_each_epoch(self, epoch):
        loss_tracker = REC()

        pbar = tqdm(
            self.train_loader,
            # ncols=160,
            leave=True,
            desc=f"Epoch-{epoch}/{self.epochs}",
        )

        for sph, h, G in pbar:
            mic = sph.to(self.device)  # B,T
            h = h.to(self.device)  # B,N
            G = G.to(self.device)  # B,1
            # sph = sph.to(self.device)  # B,T

            self.optimizer.zero_grad()

            enh, loss, loss_dict = self._fit_generator_step(mic, h, G)

            loss.backward()
            # loss.backward(retain_graph=True)
            # torch.nn.utils.clip_grad_norm_(self.net.parameters(), 3, 2)
            self.optimizer.step()

            loss_tracker.update(loss_dict)
            pbar.set_postfix(loss_tracker.state_dict())

        return loss_tracker.state_dict()

    def _valid_step(self, mic, rir, G, nlen, ret_loss) -> Tuple[Tensor, Dict]:
        N = mic.shape[-1]
        nB = mic.shape[0]

        idx = torch.arange(0, N - self.nblk + 1, self.nblk).reshape(-1, 1) + torch.arange(
            self.nblk
        )  # nF, nD
        mic_frames = mic[:, idx]  # nB,nF,nD
        mic_frames = F.pad(mic_frames, (0, 0, 0, 1, 0, 0))

        fb_data = torch.zeros(nB, self.nblk).float().to(mic.device)
        ref = torch.zeros(nB, self.nblk).float().to(mic.device)
        FB = AcousticFeedbackSim(rir, self.nblk).to(self.device)
        # FB.reset_cache(mic)

        out_ = []
        h_stat = None
        self.net.reset_buff(mic)
        for i in range(mic_frames.shape[1]):
            d = mic_frames[:, i, :]  # B,nblk

            mix = d + fb_data
            if self.kwargs.get("woRef", False):
                with torch.no_grad():
                    enh, h_stat = self.net(mix, h_stat, online=True)  # B,T
            else:
                with torch.no_grad():
                    enh, h_stat = self.net(mix, ref, h_stat, online=True)  # B,T

            ref = enh * G
            fb_data = FB(ref)
            out_.append(enh)

        N = (mic_frames.shape[1] - 1) * self.nblk
        out = torch.concat(out_[1:], dim=-1)

        metric_dict = self.valid_fn_list(mic, out, nlen, ret_loss)

        return out, metric_dict

    def _valid_each_epoch(self, epoch):
        metric_rec = REC()

        pbar = tqdm(
            self.valid_loader,
            # ncols=300,
            leave=True,
            desc=f"Valid-{epoch}/{self.epochs}",
        )

        draw = False

        for sph, h, G, nlen in pbar:
            mic = sph.to(self.device)  # B,T
            h = h.to(self.device)  # B,N
            G = G.to(self.device)  # B,1
            # sph = sph.to(self.device)  # b,c,t,f
            nlen = self.stft.nLen(nlen).to(self.device)

            enh, metric_dict = self._valid_step(mic, h, G, nlen=nlen, ret_loss=True)

            if draw is True:
                with torch.no_grad():
                    sxk = self.stft.transform(sph)
                    exk = self.stft.transform(enh)
                self._draw_spectrogram(epoch, sxk, exk, titles=("sph", "enh"))
                draw = False

            metric_rec.update(metric_dict)
            # pbar.set_postfix(metric_rec.state_dict())
            show_dict = dict(**metric_rec.state_dict())
            show_dict = {k: v for k, v in show_dict.items() if "vloss" not in k}
            self.pbar_postfix_color(pbar, show_dict)

        out = {}
        for k, v in metric_rec.state_dict().items():
            if k in self.raw_metrics.get("valid", {}):
                if not isinstance(v, dict):
                    out[k] = {"raw": self.raw_metrics["valid"][k], "enh": v}
                else:
                    out[k] = {"raw": self.raw_metrics["valid"][k], **v}
            else:
                out[k] = v

        return out

    def _vtest_each_epoch(self, epoch):
        out = {}

        metric_rec = REC()
        dirname = os.path.split(self.vtest_dset.dirname)[-1]
        pbar = tqdm(
            self.vtest_loader,
            # ncols=300,
            leave=False,
            desc=f"vTest-{epoch}/{dirname}",
        )

        for sph, h, G, nlen in pbar:
            mic = sph.to(self.device)  # B,T
            h = h.to(self.device)  # B,N
            G = G.to(self.device)  # B,1
            # sph = sph.to(self.device)  # b,c,t,f
            nlen = self.stft.nLen(nlen).to(self.device)

            enh, metric_dict = self._valid_step(mic, h, G, nlen=nlen, ret_loss=False)

            metric_rec.update(metric_dict)
            self.pbar_postfix_color(pbar, metric_rec.state_dict(), "yellow")
            # pbar.set_postfix(metric_rec.state_dict())

        dirn = {}
        for k, v in metric_rec.state_dict().items():
            if k in self.raw_metrics.get("vtest", {}):
                # dirn[k] = {"raw": self.raw_metrics["vtest"][k], "enh": v}
                if not isinstance(v, dict):
                    out[k] = {"raw": self.raw_metrics["vtest"][k], "enh": v}
                else:
                    out[k] = {"raw": self.raw_metrics["vtest"][k], **v}
            else:
                dirn[k] = v

        out[dirname] = dirn
        return out

    def prediction_per_epoch(self, epoch):
        outdir = super().prediction_per_epoch(epoch)

        n_idx = 0
        for mic, rir, G, fname in self.vpred_dset:
            if n_idx >= 20:
                break

            mic = mic.to(self.device)  # 1,T
            rir = rir.to(self.device)  # 1,N
            G = G.to(self.device)  # 1,1

            FB = AcousticFeedbackSim(rir, self.nblk).to(self.device)
            FB_ = AcousticFeedbackSim(rir, self.nblk).to(self.device)
            # self.FB.reset_cache(mic)
            # self.FB_.reset_cache(mic)

            N = mic.shape[-1]
            idx = torch.arange(0, N - self.nblk + 1, self.nblk).reshape(-1, 1) + torch.arange(
                self.nblk
            )  # nF, nD
            mic_frames = mic[:, idx]
            mic_frames = F.pad(mic_frames, (0, 0, 0, 1, 0, 0))

            nB = mic.shape[0]
            fb_data = torch.zeros(nB, self.nblk).float().to(self.device)
            fb_data_ = torch.zeros(nB, self.nblk).float().to(self.device)
            ref = torch.zeros(nB, self.nblk).float().to(self.device)

            out_ = []
            raw_ = []

            h = None
            self.net.reset_buff(mic)
            for i in range(mic_frames.shape[1]):
                # d = mic[i : i + self.nblk]
                d = mic_frames[:, i, :]

                mix = d + fb_data
                raw = d + fb_data_
                if self.kwargs.get("woRef", False):
                    with torch.no_grad():
                        enh, h = self.net(mix, h, online=True)  # B,T
                else:
                    with torch.no_grad():
                        enh, h = self.net(mix, ref, h, online=True)  # B,T

                enh_g = enh * G
                fb_data = FB(enh_g.detach())
                fb_data_ = FB_(raw * G)
                ref = enh_g.detach()
                out_.append(enh)
                raw_.append(raw)

            N = (mic_frames.shape[1] - 1) * self.nblk
            out = torch.concat(out_[1:], dim=-1)
            raw = torch.concat(raw_, dim=-1)

            audiowrite(
                f"{outdir}/{fname}",
                np.stack(
                    [
                        mic.squeeze().cpu().numpy()[:N],
                        out.squeeze().cpu().numpy()[:N],
                        raw.squeeze().cpu().numpy()[:N],
                    ],
                    axis=-1,
                ),
                self.fs,
            )
            n_idx += 1

    def _net_flops(self) -> int:
        from thop import profile
        import warnings
        import copy

        x = torch.randn(1, 64)
        self.net.reset_buff(x)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="This API is being deprecated")
            if self.kwargs.get("woRef", False):
                flops, _ = profile(
                    copy.deepcopy(self.net).cpu(),
                    inputs=(x, None, True),
                    verbose=False,
                )
            else:
                flops, _ = profile(
                    copy.deepcopy(self.net).cpu(),
                    inputs=(x, x, None, True),
                    verbose=False,
                )
        return flops * 250


class TrainerE2E(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def align(self, x, N=None):
        N = self.nblk if not N else N
        L = (x.size(-1) // N) * N
        return x[..., :L]

    def _fit_generator_step(self, mic, rir, G):
        mic = self.align(mic)
        FB = AcousticFeedbackSim(rir, self.nblk).to(self.device)

        # B,T
        ref = F.pad(mic[..., : -self.nblk], (self.nblk, 0, 0, 0)) * G
        fb_data = FB.apply_full(ref)
        mix = mic + fb_data

        if self.kwargs.get("woRef", False):
            out, _ = self.net(mix, online=False)  # B,T
        else:
            out, _ = self.net(mix, ref, online=False)  # B,T

        loss_dict = self.loss_fn_list(mic, out)
        loss = loss_dict["loss"]

        return out.detach(), loss, loss_dict

    def _valid_step_e2e(self, mic, rir, G, nlen, ret_loss) -> Tuple[Tensor, Dict, str]:
        mic = self.align(mic)
        FB = AcousticFeedbackSim(rir, self.nblk).to(self.device)

        # * step1, teacher forcing validation
        ref_full = F.pad(mic[..., : -self.nblk], (self.nblk, 0, 0, 0)) * G
        fb_data_full = FB.apply_full(ref_full)
        mix = mic + fb_data_full

        if self.kwargs.get("woRef", False):
            with torch.no_grad():
                enh, _ = self.net(mix, online=False)
        else:
            with torch.no_grad():
                enh, _ = self.net(mix, ref_full, online=False)

        metric_dict = self.valid_fn_list(mic, enh, nlen, ret_loss)

        return enh, metric_dict, "E"

    def _valid_step_recurrent(self, mic, rir, G, nlen, ret_loss) -> Tuple[Tensor, Dict, str]:
        mic = self.align(mic)
        FB = AcousticFeedbackSim(rir, self.nblk).to(self.device)

        # * step2, recurresive learing validation
        nB, N = mic.shape[0], mic.shape[-1]
        idx = torch.arange(0, N - self.nblk + 1, self.nblk).reshape(-1, 1) + torch.arange(
            self.nblk
        )  # nF, nD
        mic_frames = mic[:, idx]
        mic_frames = F.pad(mic_frames, (0, 0, 0, 1, 0, 0))

        fb_data = torch.zeros(nB, self.nblk).float().cuda()
        ref = torch.zeros(nB, self.nblk).float().cuda()

        out_ = []
        h_stat = None
        self.net.reset_buff(mic)
        FB.reset_cache(mic)
        for i in range(mic_frames.shape[1]):
            d = mic_frames[:, i, :]

            mix = d + fb_data

            if self.kwargs.get("woRef", False):
                with torch.no_grad():
                    enh, h_stat = self.net(mix, h_stat, online=True)  # B,T
            else:
                with torch.no_grad():
                    enh, h_stat = self.net(mix, ref, h_stat, online=True)  # B,T

            ref = enh * G
            fb_data = FB(ref)
            out_.append(enh)

        N = (mic_frames.shape[1] - 1) * self.nblk
        out = torch.concat(out_[1:], dim=-1)

        metric_dict = self.valid_fn_list(mic, out, nlen, ret_loss)

        return out, metric_dict, "R"

    def _valid_step(self, mic, rir, G, nlen, ret_loss) -> Tuple[Tensor, Dict]:
        _, meta_e, t1 = self._valid_step_e2e(mic, rir, G, nlen, ret_loss)
        out_r, metric_dict, t2 = self._valid_step_recurrent(mic, rir, G, nlen, ret_loss)

        # * step3, merge two strategies.
        metric_dict = self.merge_metric(meta_e, metric_dict, tags=(t1, t2))
        # metric_dict.update({f"{k}_TF": v for k, v in meta_e.items()})
        # print(metric_dict)

        return out_r, metric_dict
