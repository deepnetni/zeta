import os
import sys

from core.utils.losses import loss_pmsqe

sys.path.append(__file__.rsplit("/", 2)[0])

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional.audio.sdr import signal_distortion_ratio as SDR
from tqdm import tqdm

from core.Trainer_wGAN_for_fig6 import Trainer
from core.utils.focal_loss import BCEFocalLoss
from core.utils.record import REC
from core.utils.HAids.PyHASQI.preset_parameters import generate_filter_params


def vad_to_frames(vad: Tensor, nframe: int, nhop: int):
    """

    :param vad: B,T
    :param nframe:
    :param nhop:
    return: B,T(#Frame),1

    """
    npad = int(nframe // 2)
    vad = vad.squeeze(-1)

    vad = F.pad(vad, (npad, npad), value=0.0)
    N = vad.shape[-1]
    idx = torch.arange(nframe).reshape(1, -1)
    step = torch.arange(0, N - nframe + 1, nhop).reshape(-1, 1)
    idx = step + idx

    frames = vad[:, idx]  # B,T(#frame),D
    frames = torch.mean(frames, dim=-1, keepdim=True)  # B,T,1
    ones_vec = torch.ones_like(frames)
    zeros_vec = torch.zeros_like(frames)

    vad_label = torch.where(frames >= 0.5, ones_vec, zeros_vec)

    return vad_label


class TrainerVAD(Trainer):
    def __init__(
        self,
        train_dset: Dataset,
        valid_dset: Dataset,
        vtest_dset: Dataset,
        train_batch_sz: int,
        vpred_dset: Optional[Dataset] = None,
        **kwargs,
    ):
        super().__init__(train_dset, valid_dset, vtest_dset, train_batch_sz, vpred_dset, **kwargs)
        self.focal = BCEFocalLoss(gamma=1, alpha=0.7).to(self.device)

    def _predict_step(self, *inputs) -> Tensor:
        with torch.no_grad():
            enh, _ = self.net(*inputs)

        return enh

    def _fit_generator_step(self, *inputs, sph, one_labels, lbl_vad):
        mic, HL = inputs
        enh, est_vad = self.net(mic, HL)  # B,T
        sph = sph[..., : enh.size(-1)]
        # loss_dict = self.loss_fn_apc_denoise(sph, enh, lbl_vad, est_vad)
        loss_dict = self.loss_fn(sph, enh)
        # * vad loss
        lbl_vad = vad_to_frames(lbl_vad, 512, 256)  # B,T,1
        vad_lv = self.focal(lbl_vad, est_vad)

        fake_metric = self.net_D(sph, enh, HL)
        loss_GAN = F.mse_loss(fake_metric.flatten(), one_labels)

        loss = loss_dict["loss"] + loss_GAN + vad_lv
        loss_dict.update({"loss_G": loss_GAN.detach(), "vad_lv": vad_lv.detach()})

        return enh, loss, loss_dict

    def _fit_discriminator_step(self, *inputs, sph, one_labels):
        enh, HL = inputs
        max_metric = self.net_D(sph, sph, HL)
        pred_metric = self.net_D(sph, enh.detach(), HL)

        hasqi_score = self.batch_hasqi_score(sph, enh, HL)
        if hasqi_score is not None:
            loss_D = F.mse_loss(pred_metric.flatten(), hasqi_score) + F.mse_loss(
                max_metric.flatten(), one_labels
            )
        else:
            loss_D = None

        return loss_D

    def _valid_step(self, *inps, sph, lbl_vad, nlen) -> Tuple[Tensor, Dict]:
        mic, HL = inps
        with torch.no_grad():
            enh, est_vad = self.net(mic, HL)  # B,T,M

        metric_dict = self.valid_fn(sph, enh, nlen)
        lbl_vad = vad_to_frames(lbl_vad, 512, 256)  # B,T,1
        vad_lv = self.focal(lbl_vad, est_vad)

        hasqi_score = self.batch_hasqi_score(sph, enh, HL)
        if hasqi_score is not None:
            hasqi_score = hasqi_score.mean()
        else:
            hasqi_score = torch.tensor(0.0)

        metric_dict.update({"HASQI": hasqi_score, "vad_lv": vad_lv})

        return enh, metric_dict

    def _fit_each_epoch(self, epoch):
        losses_rec = REC()

        if hasattr(self.net, "setup_num"):
            self.net.setup_num(epoch)

        pbar = tqdm(
            self.train_loader,
            # ncols=160,
            leave=True,
            desc=f"Epoch-{epoch}/{self.epochs}",
        )
        generate_filter_params(119808)
        for mic, lbl, HL in pbar:
            sph, lbl_vad = lbl[..., 0], lbl[..., 1]
            mic = mic.to(self.device)  # B,T
            sph = sph.to(self.device)  # B,T
            HL = HL.to(self.device)  # B,6
            lbl_vad = lbl_vad.to(self.device)  # B,6
            one_labels = torch.ones(mic.shape[0]).float().cuda()  # B,

            ###################
            # Train Generator #
            ###################
            self.optimizer.zero_grad()

            enh, loss, loss_dict = self._fit_generator_step(
                mic, HL, sph=sph, one_labels=one_labels, lbl_vad=lbl_vad
            )

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.net.parameters(), 3, 2)
            self.optimizer.step()
            losses_rec.update(loss_dict)

            #######################
            # Train Discriminator #
            #######################
            self.optimizer_D.zero_grad()
            loss_D = self._fit_discriminator_step(enh, HL, sph=sph, one_labels=one_labels)
            if loss_D is not None:
                loss_D.backward()
                # torch.nn.utils.clip_grad_norm_(self.net.parameters(), 3, 2)
                self.optimizer_D.step()
            else:
                loss_D = torch.tensor([0.0])

            losses_rec.update({"loss_D": loss_D.detach()})
            pbar.set_postfix(**losses_rec.state_dict())

        return losses_rec.state_dict()

    def _valid_each_epoch(self, epoch):
        metric_rec = REC()

        pbar = tqdm(
            self.valid_loader,
            # ncols=300,
            leave=True,
            desc=f"Valid-{epoch}/{self.epochs}",
        )

        draw = False

        generate_filter_params(119808)
        for mic, lbl, HL, nlen in pbar:
            sph, lbl_vad = lbl[..., 0], lbl[..., 1]
            mic = mic.to(self.device)  # B,T,6
            sph = sph.to(self.device)  # b,c,t,f
            HL = HL.to(self.device)  # B,6
            lbl_vad = lbl_vad.to(self.device)  # B,6
            nlen = self.stft.nLen(nlen).to(self.device)
            # nlen = nlen.to(self.device)  # B

            enh, metric_dict = self._valid_step(mic, HL, sph=sph, lbl_vad=lbl_vad, nlen=nlen)

            if draw is True:
                with torch.no_grad():
                    sxk = self.stft.transform(sph)
                    exk = self.stft.transform(enh)
                self._draw_spectrogram(epoch, sxk, exk, titles=("sph", "enh"))
                draw = False

            # record the loss
            metric_rec.update(metric_dict)
            pbar.set_postfix(**metric_rec.state_dict())
            # break

        out = {}
        for k, v in metric_rec.state_dict().items():
            if k in self.raw_metrics["valid"]:
                out[k] = {"raw": self.raw_metrics["valid"][k], "enh": v}
            else:
                out[k] = v
        # return metric_rec.state_dict()
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
        # vtest_outdir = os.path.join(self.vtest_outdir, dirname)
        # shutil.rmtree(vtest_outdir) if os.path.exists(vtest_outdir) else None

        generate_filter_params(240000)
        for mic, lbl, HL, nlen in pbar:
            sph, lbl_vad = lbl[..., 0], lbl[..., 1]
            mic = mic.to(self.device)  # B,T,6
            sph = sph.to(self.device)  # b,c,t,f
            HL = HL.to(self.device)  # B,6
            lbl_vad = lbl_vad.to(self.device)  # B,6
            nlen = self.stft.nLen(nlen).to(self.device)

            enh, metric_dict = self._valid_step(mic, HL, sph=sph, lbl_vad=lbl_vad, nlen=nlen)
            # with torch.no_grad():
            #     enh = self.net(mic, HL)

            # metric_dict = self.valid_fn(sph, enh, nlen, return_loss=False)
            # hasqi_score = self.batch_hasqi_score(sph, enh, HL)
            # if hasqi_score is not None:
            #     hasqi_score = hasqi_score.mean()
            # else:
            #     hasqi_score = torch.tensor(0.0)

            # metric_dict.update({"HASQI": hasqi_score})
            # record the loss
            metric_rec.update(metric_dict)
            # pbar.set_postfix(**metric_rec.state_dict())
            # break

        dirn = {}
        for k, v in metric_rec.state_dict().items():
            if k in self.raw_metrics["vtest"]:
                dirn[k] = {"raw": self.raw_metrics["vtest"][k], "enh": v}
            else:
                dirn[k] = v
        out[dirname] = dirn
        return out
