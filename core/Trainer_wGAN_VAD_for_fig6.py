import os
import sys
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

from .Trainer_wGAN_for_fig6 import Trainer
from utils.check_flops import check_flops
from utils.focal_loss import BCEFocalLoss
from utils.HAids.PyFIG6.pyFIG6 import FIG6_compensation_vad
from utils.HAids.PyHASQI.preset_parameters import generate_filter_params
from utils.losses import loss_pmsqe
from utils.record import REC

# sys.path.append(__file__.rsplit("/", 1)[0])


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

    def loss_fn(self, clean: Tensor, enh: Tensor) -> Dict:
        """
        clean: B,T
        """
        # * pase loss
        # assert self.pase is not None
        # clean_pase = self.pase(clean.unsqueeze(1))  # B,1,T
        # clean_pase = clean_pase.flatten(0)
        # enh_pase = self.pase(enh.unsqueeze(1))
        # enh_pase = enh_pase.flatten(0)
        # pase_lv = F.mse_loss(clean_pase, enh_pase)

        specs_enh = self.stft.transform(enh)  # B,2,T,F
        specs_sph = self.stft.transform(clean)

        pmsqe_score = loss_pmsqe(specs_sph, specs_enh)
        sc_loss, mag_loss = self.ms_stft_loss(enh, clean)
        # loss = sc_loss + mag_loss + 0.3 * pmsqe_score  # + 0.25 * pase_loss
        # loss = 0.5 * pase_lv + sc_loss + mag_loss + 0.3 * pmsqe_score  # + 0.25 * pase_loss
        loss = sc_loss + mag_loss + 0.3 * pmsqe_score  # + 0.25 * pase_loss

        return {
            "loss": loss,
            "pmsq": 0.3 * pmsqe_score.detach(),
            "sc": sc_loss.detach(),
            "mag": mag_loss.detach(),
            # "pase": 0.5 * pase_lv.detach(),
        }

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

        loss = loss_dict["loss"] + 0.5 * loss_GAN + 0.3 * vad_lv
        loss_dict.update({"loss_G": 0.5 * loss_GAN.detach(), "vad_lv": 0.3 * vad_lv.detach()})

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

        if nlen.unique().numel() == 1:
            hasqi_score = self.batch_hasqi_score(sph, enh, HL)
        else:
            hasqi_score = self.batch_hasqi_score_unfix(sph, enh, HL)
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
            pbar.set_postfix(losses_rec.state_dict())

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
            pbar.set_postfix(metric_rec.state_dict())
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

        generate_filter_params(119808)
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


class TrainerSEVAD(TrainerVAD):
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

    def _predict_step(self, *inputs) -> Tensor:
        mic, _ = inputs
        with torch.no_grad():
            enh, _ = self.net(mic)

        return enh

    def _fit_each_epoch(self, epoch):
        losses_rec = REC()

        pbar = tqdm(
            self.train_loader,
            ncols=160,
            leave=True,
            desc=f"Epoch-{epoch}/{self.epochs}",
        )

        generate_filter_params(119808)
        # skip_count = 0
        for mic, lbl, cln, HL in pbar:
            mic = mic.to(self.device)  # B,T
            sph, lbl_vad = lbl[..., 0], lbl[..., 1]
            cln = cln.to(self.device)  # B,T
            HL = HL.to(self.device)  # B,6
            lbl_vad = lbl_vad.to(self.device)  # B,6

            ###################
            # Train Generator #
            ###################
            self.optimizer.zero_grad()

            enh, loss, loss_dict = self._fit_generator_step(
                mic, HL, sph=sph, cln=cln, lbl_vad=lbl_vad
            )

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
            # has_nan_inf = 0
            # for params in self.net.parameters():
            #     if params.requires_grad:
            #         has_nan_inf += torch.sum(torch.isnan(params.grad))
            #         has_nan_inf += torch.sum(torch.isinf(params.grad))
            # if has_nan_inf == 0:
            #     self.optimizer.step()
            # else:
            #     skip_count += 1
            self.optimizer.step()
            losses_rec.update(loss_dict)

            pbar.set_postfix(losses_rec.state_dict())
            # show_state = losses_rec.state_dict()
            # pbar.set_postfix(**show_state)

        return losses_rec.state_dict()

    def _fit_generator_step(self, *inputs, sph, cln, lbl_vad):
        mic, HL = inputs
        enh, est_vad = self.net(mic)  # B,T
        # sph = sph[..., : enh.size(-1)]
        cln = cln[..., : enh.size(-1)]

        lbl_vad = vad_to_frames(lbl_vad, 512, 256)  # B,T,1
        vad_lv = self.focal(lbl_vad, est_vad)

        loss_dict = self.loss_fn(cln, enh)

        loss = loss_dict["loss"] + vad_lv
        loss_dict.update({"vad_lv": vad_lv.detach()})

        return enh, loss, loss_dict

    def _valid_step(self, *inps, sph, lbl_vad, nlen) -> Tuple[Tensor, Dict]:
        mic, HL = inps
        with torch.no_grad():
            enh, est_vad = self.net(mic)  # B,T,M

        with Parallel(n_jobs=8) as parallel:
            fig6_out = parallel(
                delayed(FIG6_compensation_vad)(hl, e, self.fs, 128, 64)
                for e, hl in zip(enh.cpu().numpy(), HL.cpu().numpy())
            )
        enh_fig6 = torch.from_numpy(np.array(fig6_out)).to(enh.device)
        N = enh_fig6.shape[-1]

        metric_dict = self.valid_fn(sph[:, :N], enh_fig6, torch.full_like(nlen, N))
        lbl_vad = vad_to_frames(lbl_vad, 512, 256)  # B,T,1
        vad_lv = self.focal(lbl_vad, est_vad)

        hasqi_score = self.batch_hasqi_score(sph, enh_fig6, HL)
        if hasqi_score is not None:
            hasqi_score = hasqi_score.mean()
        else:
            hasqi_score = torch.tensor(0.0)

        metric_dict.update({"HASQI": hasqi_score, "vad_lv": vad_lv})

        return enh, metric_dict

    def _net_flops(self) -> int:
        import copy

        x = torch.randn(1, 16000)

        flops, params = check_flops(copy.deepcopy(self.net).cpu(), x)
        return flops
