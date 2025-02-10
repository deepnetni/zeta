import os
import sys

# sys.path.append(__file__.rsplit("/", 2)[0])  # up two /
sys.path.append(__file__.rsplit("/", 1)[0])  # up two /
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Optimizer, lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional.audio.sdr import signal_distortion_ratio as SDR
from tqdm import tqdm

from models.APC_SNR.apc_snr import APC_SNR_multi_filter
from models.conv_stft import STFT
from models.pase.models.frontend import wf_builder
from utils.audiolib import audiowrite
from utils.check_flops import check_flops
from utils.composite_metrics import eval_composite
from utils.Engine import EngineGAN
from utils.HAids.PyFIG6.pyFIG6 import FIG6_compensation_vad
from utils.HAids.PyHASQI.HASQI_revised import HASQI_v2
from utils.losses import loss_phase, loss_pmsqe
from utils.record import REC
from utils.stft_loss import MultiResolutionSTFTLoss
from utils.trunk_v2 import FIG6Trunk
from utils.HAids.PyHASQI.preset_parameters import generate_filter_params


def pad_to_longest(batch):
    """
    batch: [(mic, ref, label), (...), ...], B,T,C
    the input data, label must with shape (T,C) if time domain
    """
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)  # data length

    seq_len = [d.size(0) for d, _, _ in batch]
    mic, label, hl = zip(*batch)  # B,T,C
    mic = pad_sequence(mic, batch_first=True).float()
    hl = pad_sequence(hl, batch_first=True).float()
    label = pad_sequence(label, batch_first=True).float()

    # data = pack_padded_sequence(data, seq_len, batch_first=True, enforce_sorted=True)

    return mic, label, hl, torch.tensor(seq_len)


class Trainer(EngineGAN):
    def __init__(
        self,
        train_dset: Dataset,
        valid_dset: Dataset,
        vtest_dset: Dataset,
        train_batch_sz: int,
        vpred_dset: Optional[Dataset] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # self.net_ae = net_ae.to(self.device)
        # self.net_ae.eval()

        self.train_loader = DataLoader(
            train_dset,
            batch_size=train_batch_sz,
            num_workers=kwargs.get("train_num_workers", 6),
            pin_memory=True,
            shuffle=True,
            worker_init_fn=self._worker_set_seed,
            generator=self._set_generator(),
        )
        self.train_dset = train_dset
        # g = torch.Generator()
        # g.manual_seed(0)
        self.valid_loader = DataLoader(
            valid_dset,
            batch_size=kwargs.get("valid_batch_sz", 2),
            num_workers=kwargs.get("valid_num_workers", 4),
            pin_memory=True,
            shuffle=False,
            worker_init_fn=self._worker_set_seed,
            generator=self._set_generator(),
            collate_fn=pad_to_longest,
            # generator=g,
        )
        self.valid_dset: FIG6Trunk = valid_dset

        self.vtest_loader = DataLoader(
            vtest_dset,
            batch_size=kwargs.get("vtest_batch_sz", 2),
            num_workers=kwargs.get("vtest_num_workers", 4),
            pin_memory=True,
            shuffle=False,
            worker_init_fn=self._worker_set_seed,
            generator=self._set_generator(),
            collate_fn=pad_to_longest,
            # generator=g,
        )
        self.vtest_dset = vtest_dset

        if vpred_dset is None:
            self.vpred_dset = vtest_dset
        else:
            self.vpred_dset = vpred_dset

        self.stft = STFT(nframe=512, nhop=256).to(self.device)
        self.stft.eval()

        self.raw_metrics = self._load_dsets_metrics(self.dsets_mfile)

        self.ms_stft_loss = MultiResolutionSTFTLoss(
            fft_sizes=[1024, 512, 256],
            hop_sizes=[512, 256, 128],
            win_lengths=[1024, 512, 256],
        ).to(self.device)
        self.ms_stft_loss.eval()

        # self.pase = wf_builder("core/config/frontend/PASE+.cfg")
        # assert self.pase is not None
        # self.pase.cuda()
        # self.pase.eval()
        # self.pase.load_pretrained("core/pretrained/pase_e199.ckpt", load_last=True, verbose=False)

        # self.APC_criterion = APC_SNR_multi_filter(
        #     model_hop=128,
        #     model_winlen=512,
        #     mag_bins=256,
        #     theta=0.01,
        #     hops=[8, 16, 32, 64],
        # ).to(self.device)

    def _config_scheduler(self, name: str, optimizer: Optimizer):
        supported = {
            "stepLR": lambda p: lr_scheduler.StepLR(p, step_size=10, gamma=0.5),
            "reduceLR": lambda p: lr_scheduler.ReduceLROnPlateau(
                p, mode="min", factor=0.5, patience=1
            ),
        }
        return supported[name](optimizer)

    def batch_hasqi_score(self, sph, est, HL):
        if isinstance(sph, torch.Tensor):
            sph = sph.cpu().detach().numpy()
        if isinstance(est, torch.Tensor):
            est = est.cpu().detach().numpy()
        if isinstance(HL, torch.Tensor):
            HL = HL.cpu().detach().numpy()

        with Parallel(n_jobs=8) as parallel:
            hasqi_score = parallel(
                delayed(HASQI_v2)(o, self.fs, e, self.fs, ht) for o, e, ht in zip(sph, est, HL)
            )

        if -1 in hasqi_score or 0 in hasqi_score:
            return None
        # hasqi_score = np.array(hasqi_score)

        return torch.FloatTensor(hasqi_score).to(self.device)

    def _valid_dsets(self):
        dset_dict = {}
        # -----------------------#
        ##### valid dataset  #####
        # -----------------------#
        metric_rec = REC()
        pbar = tqdm(
            self.valid_loader,
            # ncols=300,
            leave=False,
            desc=f"v-{self.valid_dset.dirname}",
        )

        generate_filter_params(119808)
        for mic, sph, hl, nlen in pbar:
            # print(mic.shape, sph.shape, hl.shape, nlen.shape)
            mic = mic.to(self.device)  # B,T; mic data
            sph = sph.to(self.device)  # B,T; clean + compensation
            nlen = self.stft.nLen(nlen).to(self.device)  # B,

            x_noisy_fig6_l = []
            for i in range(mic.shape[0]):
                mic_ = mic[i].cpu().numpy()
                hl_ = hl[i].cpu().numpy()
                x_noisy_fig6 = FIG6_compensation_vad(hl_, mic_, self.fs, 128, 64)
                x_noisy_fig6_l.append(x_noisy_fig6)

            x_noisy_fig6 = np.stack(x_noisy_fig6_l, axis=0)
            x_noisy_fig6 = x_noisy_fig6[:, : nlen[0]]
            sph = sph[:, : nlen[0]]
            hasqi_score = self.batch_hasqi_score(sph, x_noisy_fig6, hl)
            if hasqi_score is not None:
                hasqi_score = hasqi_score.mean()
            else:
                hasqi_score = torch.tensor(0.0)

            metric_dict = self.valid_fn(sph, mic, nlen, return_loss=False)
            metric_dict.pop("score")
            # record the loss
            metric_rec.update({**metric_dict, "HASQI": hasqi_score})
            pbar.set_postfix(**metric_rec.state_dict())

        dset_dict["valid"] = metric_rec.state_dict()

        # -----------------------#
        ##### vtest dataset ######
        # -----------------------#
        metric_rec = REC()
        pbar = tqdm(
            self.vtest_loader,
            # ncols=300,
            leave=False,
            desc=f"v-{self.vtest_dset.dirname}",
        )

        # generate_filter_params(240000)
        for mic, sph, hl, nlen in pbar:
            mic = mic.to(self.device)  # B,T,6
            sph = sph.to(self.device)  # b,c,t,f
            nlen = self.stft.nLen(nlen).to(self.device)  # B,

            x_noisy_fig6_l = []
            for i in range(mic.shape[0]):
                mic_ = mic[i].cpu().numpy()
                hl_ = hl[i].cpu().numpy()
                x_noisy_fig6 = FIG6_compensation_vad(hl_, mic_, self.fs, 128, 64)
                x_noisy_fig6_l.append(x_noisy_fig6)

            x_noisy_fig6 = np.stack(x_noisy_fig6_l, axis=0)
            x_noisy_fig6 = x_noisy_fig6[:, : nlen[0]]
            sph = sph[:, : nlen[0]]
            hasqi_score = self.batch_hasqi_score(sph, x_noisy_fig6, hl)
            if hasqi_score is not None:
                hasqi_score = hasqi_score.mean()
            else:
                hasqi_score = torch.tensor(0.0)

            metric_dict = self.valid_fn(sph, mic, nlen, return_loss=False)
            metric_dict.pop("score")
            metric_dict.update(
                eval_composite(sph.cpu().numpy(), mic.cpu().numpy(), sample_rate=16000)
            )

            # record the loss
            metric_rec.update({**metric_dict, "HASQI": hasqi_score})
            pbar.set_postfix(**metric_rec.state_dict())

        dset_dict["vtest"] = metric_rec.state_dict()
        print(dset_dict)

        return dset_dict

    def loss_fn_apc_denoise_(self, clean: Tensor, enh: Tensor) -> Dict:
        """loss_fn_apc_denoise_wphase_loss
        clean: B,T
        """
        specs_enh = self.stft.transform(enh)  # B,2,T,F
        specs_sph = self.stft.transform(clean)

        # * pase loss
        assert self.pase is not None
        clean_pase = self.pase(clean.unsqueeze(1))  # B,1,T
        clean_pase = clean_pase.flatten(0)
        enh_pase = self.pase(enh.unsqueeze(1))
        enh_pase = enh_pase.flatten(0)
        pase_loss = F.mse_loss(clean_pase, enh_pase)

        # apc loss
        APC_SNR_loss, apc_pmsqe_loss = self.APC_criterion(enh + 1e-8, clean + 1e-8)

        # phase loss
        ph_lv, phase_dict = loss_phase(specs_sph, specs_enh)

        loss = 0.05 * APC_SNR_loss + apc_pmsqe_loss + 0.25 * pase_loss + 0.2 * ph_lv

        return {
            "loss": loss,
            "pmsqe": apc_pmsqe_loss.detach(),
            "apc_snr": 0.05 * APC_SNR_loss.detach(),
            "pase": 0.25 * pase_loss.detach(),
            "phase": 0.2 * ph_lv.detach(),
            **phase_dict,
        }

    def loss_fn_apc_denoise(self, clean: Tensor, enh: Tensor) -> Dict:
        """
        clean: B,T
        """
        # specs_enh = self.stft.transform(pred) # B,2,T,F
        # specs_sph = self.stft.transform(clean)

        # * pase loss
        assert self.pase is not None
        clean_pase = self.pase(clean.unsqueeze(1))  # B,1,T
        clean_pase = clean_pase.flatten(0)
        enh_pase = self.pase(enh.unsqueeze(1))
        enh_pase = enh_pase.flatten(0)
        pase_loss = F.mse_loss(clean_pase, enh_pase)

        # apc loss
        APC_SNR_loss, apc_pmsqe_loss = self.APC_criterion(enh + 1e-8, clean + 1e-8)
        loss = 0.05 * APC_SNR_loss + apc_pmsqe_loss + 0.25 * pase_loss
        return {
            "loss": loss,
            "pmsqe": apc_pmsqe_loss.detach(),
            "apc_snr": 0.05 * APC_SNR_loss.detach(),
            "pase": 0.25 * pase_loss.detach(),
        }

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
        # pase_loss = F.mse_loss(clean_pase, enh_pase)

        specs_enh = self.stft.transform(enh)  # B,2,T,F
        specs_sph = self.stft.transform(clean)

        # sisnr_lv = loss_sisnr(clean, enh)
        pmsqe_score = loss_pmsqe(specs_sph, specs_enh)
        # mse_mag, mse_pha = loss_compressed_mag(specs_sph, specs_enh)
        # loss = 0.05 * sisnr_lv + mse_pha + mse_mag + 0.3 * pmsqe_score
        sc_loss, mag_loss = self.ms_stft_loss(enh, clean)
        # loss = 0.05 * sisnr_lv + sc_loss + mag_loss + 0.3 * pmsqe_score
        loss = sc_loss + mag_loss + 0.3 * pmsqe_score  # + 0.25 * pase_loss
        # sdr_lv = -SDR(preds=enh, target=clean).mean()
        # sc_loss, mag_loss = self.ms_stft_loss(enh, clean)
        # else:
        #     for idx, n in enumerate(nlen):
        #         cln_ = clean[idx, :n]  # B,T
        #         enh_ = enh[idx, :n]
        #         sc_, mag_ = self.ms_stft_loss(enh_, cln_)
        #         sc_loss = sc_loss + sc_
        #         mag_loss = mag_loss + mag_

        return {
            "loss": loss,
            # "sisnr": 0.05 * sisnr_lv.detach(),
            # "mag": mse_mag.detach(),
            # "pha": mse_pha.detach(),
            "pmsq": 0.3 * pmsqe_score.detach(),
            "sc": sc_loss.detach(),
            "mag": mag_loss.detach(),
            # "pase_lv": 0.25 * pase_loss.detach(),
        }

    def loss_D_fn(self, clean: Tensor, enh: Tensor) -> Dict:
        """
        clean: B,T
        """
        # apc loss
        # loss_APC_SNR, loss_pmsqe = self.APC_criterion(enh + 1e-8, clean + 1e-8)
        # loss = 0.05 * loss_APC_SNR + loss_pmsqe  # + loss_pase
        # return {
        #     "loss": loss,
        #     "pmsqe": loss_pmsqe.detach(),
        #     "apc_snr": 0.05 * loss_APC_SNR.detach(),
        # }
        # sisdr_lv = loss_sisnr(clean, enh)
        specs_enh = self.stft.transform(enh)
        specs_sph = self.stft.transform(clean)
        pmsqe_score = 0.3 * loss_pmsqe(specs_sph, specs_enh)
        # mse_mag, mse_pha = loss_compressed_mag(specs_sph, specs_enh)
        # loss = 0.05 * sisnr_lv + mse_pha + mse_mag + pmsq_score
        # return {
        #     "loss": loss,
        #     "sisnr": sisnr_lv.detach(),
        #     "mag": mse_mag.detach(),
        #     "pha": mse_pha.detach(),
        #     "pmsq": pmsq_score.detach(),
        # }
        # sdr_lv = -SDR(preds=enh, target=clean).mean()
        sc_loss, mag_loss = self.ms_stft_loss(enh, clean)

        # * pase loss
        # clean = clean.unsqueeze(1)  # B,1,T
        # enh = enh.unsqueeze(1)
        # clean_pase = self.pase(clean)
        # clean_pase = clean_pase.flatten(0)
        # enh_pase = self.pase(enh)
        # enh_pase = enh_pase.flatten(0)
        # loss_pase = F.mse_loss(clean_pase, enh_pase)

        loss = sc_loss + mag_loss + pmsqe_score  # + loss_pase  # + 0.05 * sdr_lv

        return {
            "loss": loss,
            "sc": sc_loss.detach(),
            "mag": mag_loss.detach(),
            "pmsqe": pmsqe_score.detach(),
            # "pase": loss_pase.detach(),
            # "sdr": 0.05 * sdr_lv.detach(),
        }

    def valid_fn(
        self, sph: Tensor, enh: Tensor, nlen_list: Tensor, return_loss: bool = True
    ) -> Dict:
        """
        B,T
        """

        nB = sph.size(0)
        # if nlen_list.unique().numel() == 1:
        if True:
            sph = sph[..., : nlen_list[0]]
            enh = enh[..., : nlen_list[0]]
            np_l_sph = sph.cpu().numpy()
            np_l_enh = enh.cpu().numpy()

            sisnr_sc = self._si_snr(sph, enh).mean()
            sdr_sc = SDR(preds=enh, target=sph).mean()
        else:
            sisnr_l = []
            sdr_l = []
            np_l_enh, np_l_sph = [], []

            sisnr_sc, sdr_sc = 0, 0
            for i in range(nB):
                sph_ = sph[i, : nlen_list[i]]  # B,T
                enh_ = enh[i, : nlen_list[i]]
                np_l_sph.append(sph_.cpu().numpy())
                np_l_enh.append(enh_.cpu().numpy())

                sisnr_l.append(self._si_snr(sph_.cpu().numpy(), enh_.cpu().numpy()))
                sdr_l.append(SDR(preds=enh_, target=sph_).cpu().numpy())
                sisnr_sc = np.array(sisnr_l).mean()
                sdr_sc = np.array(sdr_l).mean()

        # sisnr_sc_ = self._si_snr(sph, enh).mean()
        pesq_wb_sc = self._pesq(np_l_sph, np_l_enh, fs=16000).mean()
        pesq_nb_sc = self._pesq(np_l_sph, np_l_enh, fs=16000, mode="nb").mean()
        stoi_sc = self._stoi(np_l_sph, np_l_enh, fs=16000).mean()

        # composite = self._eval(clean, enh, 16000)
        # composite = {k: np.mean(v) for k, v in composite.items()}
        # pesq = composite.pop("pesq")

        state = {
            "score": pesq_wb_sc,
            "sisnr": sisnr_sc,
            # "sisnr_pad": sisnr_sc_.cpu().detach().numpy(),
            "sdr": sdr_sc,
            "pesq": pesq_wb_sc,
            "pesq_nb": pesq_nb_sc,
            "stoi": stoi_sc,
        }

        if return_loss:
            loss_dict = self.loss_fn(sph[..., : enh.size(-1)], enh)
            # loss_dict = self.loss_fn_apc_denoise(sph, enh)
        else:
            loss_dict = {}

        # return dict(state, **composite)
        return dict(state, **loss_dict)

    def _fit_generator_step(self, *inputs, sph, one_labels):
        """each training step in epoch, revised it if model has different output formats.

        :param sph:
        :param one_labels:
        :returns:

        """
        mic, HL = inputs
        enh = self.net(mic, HL)  # B,T
        sph = sph[..., : enh.size(-1)]
        # loss_dict = self.loss_fn_apc_denoise(sph, enh)
        loss_dict = self.loss_fn(sph, enh)

        fake_metric = self.net_D(sph, enh, HL)
        loss_GAN = F.mse_loss(fake_metric.flatten(), one_labels)
        loss = loss_dict["loss"] + loss_GAN
        loss_dict.update({"loss_G": loss_GAN.detach()})

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

    def _valid_step(self, *inps, sph, nlen) -> Tuple[Tensor, Dict]:
        mic, HL = inps
        with torch.no_grad():
            enh = self.net(mic, HL)  # B,T,M

        metric_dict = self.valid_fn(sph, enh, nlen)
        hasqi_score = self.batch_hasqi_score(sph, enh, HL)
        if hasqi_score is not None:
            hasqi_score = hasqi_score.mean()
        else:
            hasqi_score = torch.tensor(0.0)
        metric_dict.update({"HASQI": hasqi_score})

        return enh, metric_dict

    def _predict_step(self, *inputs) -> Tensor:
        with torch.no_grad():
            enh = self.net(*inputs)

        return enh

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
        for mic, sph, HL in pbar:
            mic = mic.to(self.device)  # B,T
            sph = sph.to(self.device)  # B,T
            HL = HL.to(self.device)  # B,6
            one_labels = torch.ones(mic.shape[0]).float().cuda()  # B,

            ###################
            # Train Generator #
            ###################
            self.optimizer.zero_grad()

            enh, loss, loss_dict = self._fit_generator_step(mic, HL, sph=sph, one_labels=one_labels)

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
        for mic, sph, HL, nlen in pbar:
            mic = mic.to(self.device)  # B,T,6
            sph = sph.to(self.device)  # b,c,t,f
            HL = HL.to(self.device)  # B,6
            nlen = self.stft.nLen(nlen).to(self.device)
            # nlen = nlen.to(self.device)  # B

            enh, metric_dict = self._valid_step(mic, HL, sph=sph, nlen=nlen)

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
        for mic, sph, HL, nlen in pbar:
            mic = mic.to(self.device)  # B,T,6
            sph = sph.to(self.device)  # b,c,t,f
            HL = HL.to(self.device)  # B,6
            nlen = self.stft.nLen(nlen).to(self.device)

            enh, metric_dict = self._valid_step(mic, HL, sph=sph, nlen=nlen)
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

    def prediction_per_epoch(self, epoch):
        outdir = super().prediction_per_epoch(epoch)

        idx = 0
        for mic, HL, fname in self.vpred_dset:
            if idx >= 20:
                break

            mic = mic.to(self.device)  # B,T,6
            HL = HL.to(self.device)  # B,6

            enh = self._predict_step(mic, HL)
            # with torch.no_grad():
            #     enh = self.net(mic, HL)

            N = enh.shape[-1]
            audiowrite(
                f"{outdir}/{fname}",
                np.stack(
                    [
                        mic.squeeze().cpu().numpy()[:N],
                        enh.squeeze().cpu().numpy(),
                    ],
                    axis=-1,
                ),
                self.fs,
            )
            idx += 1

    def _net_flops(self) -> int:
        import copy

        # from thop import profile

        x = torch.randn(1, 16000)
        hl = torch.randn(1, 6)

        flops, params = check_flops(copy.deepcopy(self.net).cpu(), x, hl)
        # with warnings.catch_warnings():
        #     warnings.filterwarnings("ignore", message="This API is being deprecated")
        #     flops, _ = profile(
        #         copy.deepcopy(self.net).cpu(),
        #         inputs=(x, hl),
        #         verbose=False,
        #     )
        return flops


class TrainerMultiOutputs(Trainer):
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
        with torch.no_grad():
            enh, _, _ = self.net(*inputs)

        return enh

    def _fit_generator_step(self, *inputs, sph, one_labels):
        mic, HL = inputs
        enh, cbi, cb = self.net(mic, HL)  # B,T
        sph = sph[..., : enh.size(-1)]
        loss_dict = self.loss_fn_apc_denoise(sph, enh)

        fake_metric = self.net_D(sph, enh, HL)
        loss_GAN = F.mse_loss(fake_metric.flatten(), one_labels)

        loss_embedding = F.mse_loss(cb, cbi.detach())
        loss_commitment = F.mse_loss(cbi, cb.detach())

        loss = loss_dict["loss"] + loss_GAN + 0.25 * loss_commitment + loss_embedding
        loss_dict.update(
            {
                "loss_G": loss_GAN.detach(),
                "loss_emb": loss_embedding.detach(),
                "loss_com": 0.25 * loss_commitment.detach(),
            }
        )

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

    def _valid_step(self, *inps, sph, nlen) -> Tuple[Tensor, Dict]:
        mic, HL = inps
        with torch.no_grad():
            enh, cbi, cb = self.net(mic, HL)  # B,T,M

        loss_embedding = F.mse_loss(cb, cbi.detach())
        loss_commitment = F.mse_loss(cbi, cb.detach())

        metric_dict = self.valid_fn(sph, enh, nlen)
        hasqi_score = self.batch_hasqi_score(sph, enh, HL)
        if hasqi_score is not None:
            hasqi_score = hasqi_score.mean()
        else:
            hasqi_score = torch.tensor(0.0)

        metric_dict.update(
            {
                "HASQI": hasqi_score,
                "embedding": loss_embedding.detach(),
                "commitment": 0.25 * loss_commitment.detach(),
            }
        )

        return enh, metric_dict


class TrainerPhase(Trainer):
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
        with torch.no_grad():
            enh, _ = self.net(*inputs)

        return enh

    def loss_fn_apc_denoise(self, clean: Tensor, enh: Tensor, phase) -> Dict:
        """loss_fn_apc_denoise_wphase_loss
        clean: B,T
        """
        # specs_enh = self.stft.transform(enh)  # B,2,T,F
        specs_sph = self.stft.transform(clean)

        # * pase loss
        assert self.pase is not None
        clean_pase = self.pase(clean.unsqueeze(1))  # B,1,T
        clean_pase = clean_pase.flatten(0)
        enh_pase = self.pase(enh.unsqueeze(1))
        enh_pase = enh_pase.flatten(0)
        pase_loss = F.mse_loss(clean_pase, enh_pase)

        # apc loss
        APC_SNR_loss, apc_pmsqe_loss = self.APC_criterion(enh + 1e-8, clean + 1e-8)

        # phase loss
        ph_lv, phase_dict = loss_phase(specs_sph, phase)

        loss = 0.05 * APC_SNR_loss + apc_pmsqe_loss + 0.25 * pase_loss + 0.2 * ph_lv

        return {
            "loss": loss,
            "pmsqe": apc_pmsqe_loss.detach(),
            "apc_snr": 0.05 * APC_SNR_loss.detach(),
            "pase": 0.25 * pase_loss.detach(),
            "phase": 0.2 * ph_lv.detach(),
            **phase_dict,
        }

    def _fit_generator_step(self, *inputs, sph, one_labels):
        mic, HL = inputs
        enh, phase = self.net(mic, HL)  # B,T
        sph = sph[..., : enh.size(-1)]
        loss_dict = self.loss_fn_apc_denoise(sph, enh, phase)

        fake_metric = self.net_D(sph, enh, HL)
        loss_GAN = F.mse_loss(fake_metric.flatten(), one_labels)

        loss = loss_dict["loss"] + loss_GAN
        loss_dict.update({"loss_G": loss_GAN.detach()})

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

    def _valid_step(self, *inps, sph, nlen) -> Tuple[Tensor, Dict]:
        mic, HL = inps
        with torch.no_grad():
            enh, phase = self.net(mic, HL)  # B,T,M

        metric_dict = self.valid_fn(sph, enh, nlen)
        hasqi_score = self.batch_hasqi_score(sph, enh, HL)
        if hasqi_score is not None:
            hasqi_score = hasqi_score.mean()
        else:
            hasqi_score = torch.tensor(0.0)

        metric_dict.update(
            {
                "HASQI": hasqi_score,
            }
        )

        return enh, metric_dict


class TrainerGumbelCodebook(Trainer):
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
        with torch.no_grad():
            enh, _ = self.net(*inputs)

        return enh

    def _fit_generator_step(self, *inputs, sph, one_labels):
        mic, HL = inputs
        enh, cb_dict = self.net(mic, HL)  # B,T
        sph = sph[..., : enh.size(-1)]
        loss_dict = self.loss_fn_apc_denoise(sph, enh)

        fake_metric = self.net_D(sph, enh, HL)
        loss_GAN = F.mse_loss(fake_metric.flatten(), one_labels)

        dv_loss = self.net.diversity_loss(cb_dict).mean()

        loss = loss_dict["loss"] + loss_GAN + 0.1 * dv_loss
        loss_dict.update(
            {
                "loss_G": loss_GAN.detach(),
                "loss_dv": 0.1 * dv_loss.detach(),
            }
        )

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

    def _valid_step(self, *inps, sph, nlen) -> Tuple[Tensor, Dict]:
        mic, HL = inps
        with torch.no_grad():
            enh, cb_dict = self.net(mic, HL)  # B,T,M

        dv_loss = self.net.diversity_loss(cb_dict).mean()

        metric_dict = self.valid_fn(sph, enh, nlen)
        hasqi_score = self.batch_hasqi_score(sph, enh, HL)
        if hasqi_score is not None:
            hasqi_score = hasqi_score.mean()
        else:
            hasqi_score = torch.tensor(0.0)

        metric_dict.update(
            {
                "HASQI": hasqi_score,
                "diversity": dv_loss.detach(),
            }
        )

        return enh, metric_dict
