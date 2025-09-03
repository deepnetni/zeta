import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional.audio.sdr import signal_distortion_ratio as SDR
from tqdm import tqdm

from utils.check_flops import check_flops
from utils.focal_loss import BCEFocalLoss
from utils.HAids.PyFIG6.pyFIG6 import FIG6_compensation_vad
from utils.HAids.PyHASQI.preset_parameters import generate_filter_params
from utils.losses import loss_pmsqe
from utils.record import REC
from Engine import Engine
from utils.audiolib import audiowrite


class Trainer(Engine):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _fit_each_epoch(self, epoch):
        loss_tracker = REC()

        pbar = tqdm(
            self.train_loader,
            # ncols=160,
            leave=True,
            desc=f"Epoch-{epoch}/{self.epochs}",
        )

        for mic, sph in pbar:
            mic = mic.to(self.device)  # B,T
            sph = sph.to(self.device)  # B,T

            self.optimizer.zero_grad()

            ref = None

            enh, loss, loss_dict = self._fit_generator_step(mic, ref, sph=sph)

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.net.parameters(), 3, 2)
            self.optimizer.step()
            loss_tracker.update(loss_dict)

            pbar.set_postfix(loss_tracker.state_dict())

        return loss_tracker.state_dict()

    def _valid_each_epoch(self, epoch):
        metric_rec = REC()

        pbar = tqdm(
            self.valid_loader,
            # ncols=300,
            leave=True,
            desc=f"Valid-{epoch}/{self.epochs}",
        )

        draw = False

        for mic, sph, nlen in pbar:
            mic = mic.to(self.device)  # B,T,6
            sph = sph.to(self.device)  # b,c,t,f
            nlen = self.stft.nLen(nlen).to(self.device)

            ref = None
            enh, metric_dict = self._valid_step(mic, sph=sph, nlen=nlen, ret_loss=True)

            if draw is True:
                with torch.no_grad():
                    sxk = self.stft.transform(sph)
                    exk = self.stft.transform(enh)
                self._draw_spectrogram(epoch, sxk, exk, titles=("sph", "enh"))
                draw = False

            metric_rec.update(metric_dict)
            # pbar.set_postfix(metric_rec.state_dict())
            show_dict = dict(**metric_rec.state_dict())
            if "vloss" in show_dict:
                del show_dict["vloss"]
            self.pbar_postfix_color(pbar, show_dict)

        out = {}
        for k, v in metric_rec.state_dict().items():
            if k in self.raw_metrics["valid"]:
                out[k] = {"raw": self.raw_metrics["valid"][k], "enh": v}
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

        for mic, sph, nlen in pbar:
            mic = mic.to(self.device)  # B,T,6
            sph = sph.to(self.device)  # b,c,t,f
            nlen = self.stft.nLen(nlen).to(self.device)

            enh, metric_dict = self._valid_step(mic, sph=sph, nlen=nlen, ret_loss=False)

            metric_rec.update(metric_dict)
            self.pbar_postfix_color(pbar, metric_rec.state_dict(), "yellow")
            # pbar.set_postfix(metric_rec.state_dict())

        dirn = {}
        for k, v in metric_rec.state_dict().items():
            if k in self.raw_metrics["vtest"]:
                dirn[k] = {"raw": self.raw_metrics["vtest"][k], "enh": v}
            else:
                dirn[k] = v

        out[dirname] = dirn
        return out

    def _fit_generator_step(self, *inputs, sph):
        mic, ref = inputs
        enh = self.net(mic, ref)  # B,T
        sph = sph[..., : enh.size(-1)]

        loss_dict = self.loss_fn_list(sph, enh)
        loss = loss_dict["loss"]

        return enh, loss, loss_dict

    def _valid_step(self, *inps, sph, nlen, ret_loss) -> Tuple[Tensor, Dict]:
        mic = inps
        with torch.no_grad():
            enh = self.net(mic, ref)  # B,T,M

        metric_dict = self.valid_fn_list(sph, enh, nlen, ret_loss)

        return enh, metric_dict

    def prediction_per_epoch(self, epoch):
        outdir = super().prediction_per_epoch(epoch)

        idx = 0
        for mic, fname in self.vpred_dset:
            if idx >= 20:
                break

            mic = mic.to(self.device)  # B,T,6

            with torch.no_grad():
                enh = self.net(mic, ref)

            N = enh.shape[-1]
            audiowrite(
                f"{outdir}/{fname}",
                np.concatenate(
                    [
                        mic.squeeze().cpu().numpy()[:N],
                        enh.squeeze().cpu().numpy()[:, None],
                    ],
                    axis=-1,
                ),
                self.fs,
            )
            idx += 1
