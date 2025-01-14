import os
import sys

from tqdm import tqdm

sys.path.append(__file__.rsplit("/", 2)[0])
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional.audio.sdr import signal_distortion_ratio as SDR

from core.models.APC_SNR.apc_snr import APC_SNR_multi_filter
from core.models.conv_stft import STFT
from core.models.pase.models.frontend import wf_builder
from core.utils.Engine import Engine
from core.utils.losses import *
from core.utils.record import REC
from core.utils.stft_loss import MultiResolutionSTFTLoss
from core.utils.trunk import pad_to_longest_aec
from core.utils.audiolib import audiowrite
from core.utils.check_flops import check_flops


class Trainer(Engine):
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
        self.nframe = kwargs.get("nframe", 512)
        self.nhop = kwargs.get("nhop", 256)  # 16ms

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
            collate_fn=pad_to_longest_aec,
            # generator=g,
        )
        self.valid_dset = valid_dset

        self.vtest_loader = DataLoader(
            vtest_dset,
            batch_size=kwargs.get("vtest_batch_sz", 2),
            num_workers=kwargs.get("vtest_num_workers", 4),
            pin_memory=True,
            shuffle=False,
            worker_init_fn=self._worker_set_seed,
            generator=self._set_generator(),
            collate_fn=pad_to_longest_aec,
            # generator=g,
        )
        self.vtest_dset = vtest_dset

        if vpred_dset is None:
            self.vpred_dset = vtest_dset
        else:
            self.vpred_dset = vpred_dset

        # self.stft = STFT(nframe=128, nhop=64, win="hann sqrt").to(self.device)
        self.stft = STFT(nframe=self.nframe, nhop=self.nhop, win="hann sqrt").to(self.device)
        self.stft.eval()

        self.raw_metrics = self._load_dsets_metrics(self.dsets_mfile)

        self.ms_stft_loss = MultiResolutionSTFTLoss(
            fft_sizes=[1024, 512, 256],
            hop_sizes=[512, 256, 128],
            win_lengths=[1024, 512, 256],
        ).to(self.device)
        self.ms_stft_loss.eval()

        self.pase = wf_builder("core/config/frontend/PASE+.cfg")
        assert self.pase is not None
        self.pase.cuda()
        self.pase.eval()
        self.pase.load_pretrained("core/pretrained/pase_e199.ckpt", load_last=True, verbose=False)

        self.APC_criterion = APC_SNR_multi_filter(
            model_hop=128,
            model_winlen=512,
            mag_bins=256,
            theta=0.01,
            hops=[8, 16, 32, 64],
        ).to(self.device)

        self.loss_fn_ = self.loss_fn

    def _valid_dsets(self):
        dset_dict = {}
        # -----------------------#
        ##### valid dataset  #####
        # -----------------------#
        metric_rec = REC()
        pbar = tqdm(
            self.valid_loader,
            ncols=120,
            leave=False,
            desc=f"v-{self.valid_dset.dirname}",
        )

        for mic, ref, sph, nlen in pbar:
            mic = mic.to(self.device)
            # ref = ref.to(self.device)
            sph = sph.to(self.device)
            nlen = self.stft.nLen(nlen).to(self.device)  # B,

            metric_dict = self.valid_fn(sph, mic, nlen, return_loss=False)
            metric_dict.pop("score")
            # record the loss
            metric_rec.update(metric_dict)
            pbar.set_postfix(**metric_rec.state_dict())

        dset_dict["valid"] = metric_rec.state_dict()

        # -----------------------#
        ##### vtest dataset ######
        # -----------------------#
        metric_rec = REC()
        pbar = tqdm(
            self.vtest_loader,
            ncols=120,
            leave=False,
            desc=f"v-{self.vtest_dset.dirname}",
        )

        for mic, ref, sph, nlen in pbar:
            mic = mic.to(self.device)
            ref = ref.to(self.device)
            sph = sph.to(self.device)
            nlen = self.stft.nLen(nlen).to(self.device)  # B,

            metric_dict = self.valid_fn(sph, mic, nlen, return_loss=False)
            metric_dict.pop("score")

            # record the loss
            metric_rec.update(metric_dict)
            pbar.set_postfix(**metric_rec.state_dict())

        dset_dict["vtest"] = metric_rec.state_dict()

        return dset_dict

    def loss_fn(self, clean: Tensor, enh: Tensor) -> Dict:
        """
        clean: B,T
        """
        specs_enh = self.stft.transform(enh)  # B,2,T,F
        specs_sph = self.stft.transform(clean)

        # sisnr_lv = loss_sisnr(clean, enh)
        pmsqe_score = loss_pmsqe(specs_sph, specs_enh)
        # mse_mag, mse_pha = loss_compressed_mag(specs_sph, specs_enh)
        # loss = 0.05 * sisnr_lv + mse_pha + mse_mag + 0.3 * pmsqe_score
        sc_loss, mag_loss = self.ms_stft_loss(enh, clean)
        # loss = 0.05 * sisnr_lv + sc_loss + mag_loss + 0.3 * pmsqe_score
        loss = sc_loss + mag_loss + 0.3 * pmsqe_score
        return {
            "loss": loss,
            # "sisnr": 0.05 * sisnr_lv.detach(),
            # "mag": mse_mag.detach(),
            # "pha": mse_pha.detach(),
            "pmsq": 0.3 * pmsqe_score.detach(),
            "sc": sc_loss.detach(),
            "mag": mag_loss.detach(),
        }
        # sdr_lv = -SDR(preds=enh, target=clean).mean()
        # sc_loss, mag_loss = self.ms_stft_loss(enh, clean)
        # else:
        #     for idx, n in enumerate(nlen):
        #         cln_ = clean[idx, :n]  # B,T
        #         enh_ = enh[idx, :n]
        #         sc_, mag_ = self.ms_stft_loss(enh_, cln_)
        #         sc_loss = sc_loss + sc_
        #         mag_loss = mag_loss + mag_

    def loss_fn_better(self, clean: Tensor, enh: Tensor) -> Dict:
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
        loss_APC_SNR, apc_pmsqe_loss = self.APC_criterion(enh + 1e-8, clean + 1e-8)
        loss = 0.05 * loss_APC_SNR + apc_pmsqe_loss + pase_loss
        return {
            "loss": loss,
            "pmsqe": apc_pmsqe_loss.detach(),
            "apc_snr": 0.05 * loss_APC_SNR.detach(),
            "pase": pase_loss.detach(),
        }

        # sisdr_lv = loss_sisnr(clean, enh)
        # pmsqe_score = 0.3 * loss_pmsqe(specs_sph, specs_enh)
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
        # sc_loss, mag_loss = self.ms_stft_loss(enh, clean)
        # else:
        #     for idx, n in enumerate(nlen):
        #         cln_ = clean[idx, :n]  # B,T
        #         enh_ = enh[idx, :n]
        #         sc_, mag_ = self.ms_stft_loss(enh_, cln_)
        #         sc_loss = sc_loss + sc_
        #         mag_loss = mag_loss + mag_

        # loss = sc_loss + mag_loss + pmsqe_score  # + loss_pase  # + 0.05 * sdr_lv
        # loss = sc_loss + mag_loss

        # return {
        #     "loss": loss,
        #     "sc": sc_loss.detach(),
        #     "mag": mag_loss.detach(),
        #     # "pmsqe": pmsqe_score.detach(),
        #     # "pase": loss_pase.detach(),
        #     # "sdr": 0.05 * sdr_lv.detach(),
        # }

    def valid_fn(
        self,
        sph: Tensor,
        enh: Tensor,
        nlen_list: Tensor,
        return_loss: bool = True,
    ) -> Dict:
        """
        B,T
        """
        sisnr_l = []
        sdr_l = []
        np_l_sph = []
        np_l_enh = []

        nB = sph.size(0)

        for i in range(nB):
            sph_ = sph[i, : nlen_list[i]]  # B,T
            enh_ = enh[i, : nlen_list[i]]
            np_l_sph.append(sph_.cpu().numpy())
            np_l_enh.append(enh_.cpu().numpy())

            sisnr_l.append(self._si_snr(sph_, enh_))
            sdr_l.append(SDR(preds=enh_, target=sph_).cpu().numpy())

        sisnr_sc = np.array(sisnr_l).mean()
        # sisnr_sc_ = self._si_snr(sph, enh).mean()
        sdr_sc = np.array(sdr_l).mean()
        pesq_wb_sc = self._pesq(np_l_sph, np_l_enh, fs=self.fs).mean()
        # pesq_nb_sc = self._pesq(np_l_sph, np_l_enh, fs=self.fs, mode="nb").mean()
        stoi_sc = self._stoi(np_l_sph, np_l_enh, fs=self.fs).mean()

        # composite = self._eval(clean, enh, 16000)
        # composite = {k: np.mean(v) for k, v in composite.items()}
        # pesq = composite.pop("pesq")

        state = {
            "score": float(pesq_wb_sc + stoi_sc) / 2.0,
            "sisnr": sisnr_sc,
            # "sisnr_pad": sisnr_sc_.cpu().detach().numpy(),
            "sdr": sdr_sc,
            # "pesq_nb": pesq_nb_sc,
            "pesq": pesq_wb_sc,
            "stoi": stoi_sc,
        }

        if return_loss:
            loss_dict = self.loss_fn_(sph[..., : enh.size(-1)], enh)
        else:
            loss_dict = {}

        # return dict(state, **composite)
        return dict(state, **loss_dict)

    def _fit_each_epoch(self, epoch):
        losses_rec = REC()

        pbar = tqdm(
            self.train_loader,
            ncols=160,
            leave=True,
            desc=f"Epoch-{epoch}/{self.epochs}",
        )
        for mic, ref, sph in pbar:
            mic = mic.to(self.device)
            ref = ref.to(self.device)
            # mic_xk = self.stft.transform(mic)
            # ref_xk = self.stft.transform(ref)
            sph = sph.to(self.device)

            self.optimizer.zero_grad()
            enh = self.net(mic, ref)
            # enh = self.stft.inverse(enh_xk)
            loss_dict = self.loss_fn_(sph[:, : enh.size(-1)], enh)

            loss = loss_dict["loss"]
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(self.net.parameters(), 3, 2)
            self.optimizer.step()

            # record the loss
            losses_rec.update(loss_dict)
            pbar.set_postfix(**losses_rec.state_dict())

        return losses_rec.state_dict()

    def _valid_each_epoch(self, epoch):
        metric_rec = REC()

        pbar = tqdm(
            self.valid_loader,
            ncols=160,
            leave=True,
            desc=f"Valid-{epoch}/{self.epochs}",
        )

        draw = False

        for mic, ref, sph, nlen in pbar:
            mic = mic.to(self.device)
            ref = ref.to(self.device)
            sph = sph.to(self.device)
            nlen = self.stft.nLen(nlen).to(self.device)

            with torch.no_grad():
                enh = self.net(mic, ref)

            metric_dict = self.valid_fn(sph, enh, nlen)

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
            if "valid" in self.raw_metrics and k in self.raw_metrics["valid"]:
                out[k] = {"raw": self.raw_metrics["valid"][k], "enh": v}
            else:
                out[k] = v
        # return metric_rec.state_dict()
        return out

    # def vtest_fn(self, sph: Tensor, enh: Tensor) -> Dict:
    #     sisnr = self._si_snr(sph.cpu().detach().numpy(), enh.cpu().detach().numpy())
    #     sisnr = np.mean(sisnr)

    #     pesq_sc = self._pesq(
    #         sph.cpu().detach().numpy(),
    #         enh.cpu().detach().numpy(),
    #         fs=16000,
    #         norm=False,
    #     ).mean()
    #     # pesq = np.mean(pesq)
    #     # composite = self._eval(clean, enh, 16000)
    #     # composite = {k: np.mean(v) for k, v in composite.items()}
    #     # pesq = composite.pop("pesq")

    #     stoi_sc = self._stoi(
    #         sph.cpu().detach().numpy(),
    #         enh.cpu().detach().numpy(),
    #         fs=16000,
    #     ).mean()
    #     # stoi = np.mean(stoi)

    #     state = {"pesq": pesq_sc, "stoi": stoi_sc, "sisnr": sisnr}

    #     # return dict(state, **composite)
    #     return state

    def _vtest_each_epoch(self, epoch):
        out = {}

        metric_rec = REC()
        dirname = os.path.split(self.vtest_dset.dirname)[-1]
        pbar = tqdm(
            self.vtest_loader,
            ncols=120,
            leave=False,
            desc=f"vTest-{epoch}/{dirname}",
        )
        # vtest_outdir = os.path.join(self.vtest_outdir, dirname)
        # shutil.rmtree(vtest_outdir) if os.path.exists(vtest_outdir) else None

        for mic, ref, sph, nlen in pbar:
            mic = mic.to(self.device)
            ref = ref.to(self.device)
            sph = sph.to(self.device)
            nlen = self.stft.nLen(nlen).to(self.device)

            with torch.no_grad():
                enh = self.net(mic, ref)

            metric_dict = self.valid_fn(sph, enh, nlen, return_loss=False)
            metric_rec.update(metric_dict)
            # pbar.set_postfix(**metric_rec.state_dict())

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
        for mic, ref, _, fname in self.vpred_dset:
            if idx >= 20:
                break

            mic = mic.to(self.device)  # 1, T
            ref = ref.to(self.device)  # 1, T
            with torch.no_grad():
                est = self.net(mic, ref)  # 1,T

            N = est.shape[-1]
            audiowrite(
                f"{outdir}/{fname}",
                np.stack(
                    [
                        mic.squeeze().cpu().numpy()[:N],
                        est.squeeze().cpu().detach().numpy(),
                    ],
                    axis=-1,
                ),
                self.fs,
            )

            idx += 1

    def _net_flops(self) -> int:
        import copy

        # from thop import profile

        # x = torch.randn(1, 2, int(16000 / self.nhop), self.nhop + 1)
        x = torch.randn(1, 16000)

        flops, params = check_flops(copy.deepcopy(self.net).cpu(), x, x)
        # with warnings.catch_warnings():
        #     warnings.filterwarnings("ignore", message="This API is being deprecated")
        #     flops, _ = profile(
        #         copy.deepcopy(self.net).cpu(),
        #         inputs=(x, x),
        #         verbose=False,
        #     )
        return flops
