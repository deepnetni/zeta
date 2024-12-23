import argparse
import os
import sys
from typing import Dict, List
import yaml

import warnings
import numpy as np
import torch
import torch.nn.functional as F

# from matplotlib import pyplot as plt
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# from utils.conv_stft_loss import MultiResolutionSTFTLoss
from tqdm import tqdm

from utils.Engine import EngineGAN
from utils.ini_opts import read_ini
from utils.losses import loss_compressed_mag, loss_pmsqe, loss_sisnr
from utils.record import REC, RECDepot
from utils.stft_loss import MultiResolutionSTFTLoss
from utils.trunk import pad_to_longest
from utils.audiolib import audioread, audiowrite
from utils.register import tables
from utils.logger import cprint
from utils.composite_metrics import eval_composite
from models.conv_stft import STFT

from datasets_manager import get_datasets

# from models.mcse_skip_exp import aia_mcse_skip_sd
from torchmetrics.functional.audio import signal_distortion_ratio as SDR
from torchmetrics.functional.audio import (
    scale_invariant_signal_distortion_ratio as si_sdr,
)
from models.APC_SNR.apc_snr import APC_SNR_multi_filter
from models.pase.models.frontend import wf_builder

from rebuild.FTCRN import *


class Train(EngineGAN):
    def __init__(
        self,
        train_dset: Dataset,
        valid_dset: Dataset,
        vtest_dset: Dataset,
        batch_sz: int,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # self.net_ae = net_ae.to(self.device)
        # self.net_ae.eval()

        self.train_loader = DataLoader(
            train_dset,
            batch_size=batch_sz,
            num_workers=6,
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
            batch_size=2,
            # batch_size=1,
            num_workers=4,
            pin_memory=True,
            shuffle=False,
            worker_init_fn=self._worker_set_seed,
            generator=self._set_generator(),
            collate_fn=pad_to_longest,
            # generator=g,
        )
        self.valid_dset = valid_dset

        self.vtest_loader = DataLoader(
            vtest_dset,
            batch_size=2,
            # batch_size=1,
            num_workers=4,
            pin_memory=True,
            shuffle=False,
            worker_init_fn=self._worker_set_seed,
            generator=self._set_generator(),
            collate_fn=pad_to_longest,
            # generator=g,
        )
        self.vtest_dset = vtest_dset

        # self.vtest_loader = (
        #     [vtest_dset] if isinstance(vtest_dset, Dataset) else vtest_dset
        # )

        self.stft = STFT(nframe=512, nhop=256).to(self.device)
        self.stft.eval()

        self.ms_stft_loss = MultiResolutionSTFTLoss(
            fft_sizes=[1024, 512, 256],
            hop_sizes=[512, 256, 128],
            win_lengths=[1024, 512, 256],
        ).to(self.device)
        self.ms_stft_loss.eval()

        self.raw_metrics = self._load_dsets_metrics(self.dsets_mfile)
        # self.pase = wf_builder("config/frontend/PASE+.cfg")
        # self.pase.cuda()
        # self.pase.eval()
        # self.pase.load_pretrained(
        #     "pretrained/pase_e199.ckpt", load_last=True, verbose=False
        # )

        # self.APC_criterion = APC_SNR_multi_filter(
        #     model_hop=128,
        #     model_winlen=512,
        #     mag_bins=256,
        #     theta=0.01,
        #     hops=[8, 16, 32, 64],
        # ).to(self.device)

    @staticmethod
    def _config_optimizer(name: str, params, **kwargs):
        return super(Train, Train)._config_optimizer(
            name, filter(lambda p: p.requires_grad, params, **kwargs)
        )

    def loss_fn(self, clean: Tensor, enh: Tensor, nlen=None) -> Dict:
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
        # else:
        #     cln_ = clean[0, : nlen[0]]  # B,T
        #     enh_ = enh[0, : nlen[0]]
        #     sc_loss, mag_loss = self.ms_stft_loss(enh_, cln_)
        #     for idx, n in enumerate(nlen[1:], start=1):
        #         cln_ = clean[idx, :n]  # B,T
        #         enh_ = enh[idx, :n]
        #         sc_, mag_ = self.ms_stft_loss(enh_, cln_)
        #         sc_loss = sc_loss + sc_
        #         mag_loss = mag_loss + mag_

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

    def _fit_each_epoch(self, epoch):
        losses_rec = REC()

        pbar = tqdm(
            self.train_loader,
            ncols=160,
            leave=True,
            desc=f"Epoch-{epoch}/{self.epochs}",
        )
        for mic, sph, HL in pbar:
            mic = mic.to(self.device)  # B,T
            sph = sph.to(self.device)  # B,T
            HL = HL.to(self.device)  # B,6
            one_labels = torch.ones(mic.shape[0]).float().cuda()

            ###################
            # Train Generator #
            ###################
            self.optimizer.zero_grad()
            enh = self.net(mic, HL)
            loss_dict = self.loss_fn(sph[:, : enh.size(-1)], enh)

            out_D = self.net_D(sph, enh, HL)
            loss_GAN = F.mse_loss(out_D.flatten(), one_labels)
            loss = loss_dict["loss"] + loss_GAN
            loss_dict.update({"loss_GAN": loss_GAN.detach()})

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.net.parameters(), 3, 2)
            self.optimizer.step()
            losses_rec.update(loss_dict)

            #######################
            # Train Discriminator #
            #######################
            self.optimizer_D.zero_grad()
            enh = self.net_D(mic)
            loss_dict = self.loss_fn(sph[:, : enh.size(-1)], enh)

            loss = loss_dict["loss"]
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(self.net.parameters(), 3, 2)
            self.optimizer.step()
            losses_rec.update(loss_dict)

            pbar.set_postfix(**losses_rec.state_dict())

        return losses_rec.state_dict()

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

        for mic, sph, nlen in pbar:
            mic = mic.to(self.device)  # B,T,6
            sph = sph.to(self.device)  # b,c,t,f
            nlen = self.stft.nLen(nlen).to(self.device)  # B,

            metric_dict = self.valid_fn(sph, mic[..., 4], nlen, return_loss=False)
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

        for mic, sph, nlen in pbar:
            mic = mic.to(self.device)  # B,T,6
            sph = sph.to(self.device)  # b,c,t,f
            nlen = self.stft.nLen(nlen).to(self.device)  # B,

            metric_dict = self.valid_fn(sph, mic[..., 4], nlen, return_loss=False)
            metric_dict.pop("score")
            metric_dict.update(
                eval_composite(sph.cpu().numpy(), mic[..., 4].cpu().numpy(), sample_rate=16000)
            )

            # record the loss
            metric_rec.update(metric_dict)
            pbar.set_postfix(**metric_rec.state_dict())

        dset_dict["vtest"] = metric_rec.state_dict()
        print(dset_dict)

        return dset_dict

    def prediction_per_epoch(self, epoch):
        outdir = super().prediction_per_epoch(epoch)

    def valid_fn(
        self, sph: Tensor, enh: Tensor, nlen_list: Tensor, return_loss: bool = True
    ) -> Dict:
        """
        B,T
        """
        sisnr_l = []
        sdr_l = []

        B = sph.size(0)
        sph_ = sph[0, : nlen_list[0]]  # B,T
        enh_ = enh[0, : nlen_list[0]]
        sisnr_l.append(self._si_snr(sph_.cpu().numpy(), enh_.cpu().numpy()))
        np_l_sph = [sph_.cpu().numpy()]
        np_l_enh = [enh_.cpu().numpy()]
        sdr_l.append(SDR(preds=enh_, target=sph_).cpu().numpy())

        for i in range(1, B):
            sph_ = sph[i, : nlen_list[i]]  # B,T
            enh_ = enh[i, : nlen_list[i]]
            np_l_sph.append(sph_.cpu().numpy())
            np_l_enh.append(enh_.cpu().numpy())

            sisnr_l.append(self._si_snr(sph_.cpu().numpy(), enh_.cpu().numpy()))
            sdr_l.append(SDR(preds=enh_, target=sph_).cpu().numpy())

        sisnr_sc = np.array(sisnr_l).mean()
        # sisnr_sc_ = self._si_snr(sph, enh).mean()
        sdr_sc = np.array(sdr_l).mean()
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
            loss_dict = self.loss_fn(sph[..., : enh.size(-1)], enh, nlen_list)
        else:
            loss_dict = {}

        # return dict(state, **composite)
        return dict(state, **loss_dict)

    def _valid_each_epoch(self, epoch):
        metric_rec = REC()

        pbar = tqdm(
            self.valid_loader,
            ncols=160,
            leave=True,
            desc=f"Valid-{epoch}/{self.epochs}",
        )

        draw = False

        for mic, sph, nlen in pbar:
            mic = mic.to(self.device)  # B,T,6
            sph = sph.to(self.device)  # b,c,t,f
            nlen = self.stft.nLen(nlen).to(self.device)
            # nlen = nlen.to(self.device)  # B

            with torch.no_grad():
                enh = self.net(mic)  # B,T,M

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
            if k in self.raw_metrics["valid"]:
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

        for mic, sph, nlen in pbar:
            mic = mic.to(self.device)  # B,T,6
            sph = sph.to(self.device)  # b,c,t,f
            nlen = self.stft.nLen(nlen).to(self.device)

            with torch.no_grad():
                enh = self.net(mic)

            metric_dict = self.valid_fn(sph, enh, nlen, return_loss=False)
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

    def _net_flops(self) -> int:
        from thop import profile
        import copy

        x = torch.randn(1, 16000)
        hl = torch.randn(1, 6)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="This API is being deprecated")
            flops, _ = profile(
                copy.deepcopy(self.net).cpu(),
                inputs=(x, hl),
                verbose=False,
            )
        return flops


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="train mode", action="store_true", default=True)
    parser.add_argument("--pred", help="predict mode", action="store_true")

    parser.add_argument("--ckpt", help="ckpt path", type=str)
    parser.add_argument("--src", help="input directory", type=str)
    parser.add_argument("--out", help="predicting output directory", type=str)
    parser.add_argument("--valid_first", help="valid first", action="store_true")

    parser.add_argument("--valid", help="input directory", action="store_true")
    parser.add_argument("--vtest", help="input directory", action="store_true")

    parser.add_argument("--conf", help="config file", default="")
    parser.add_argument("--name", help="name of the model")
    parser.add_argument("--epoch", help="epoch", type=int)

    args = parser.parse_args()
    args.train = False if args.pred else True
    return args


def fetch_config(cfg_fname):
    print("##", cfg_fname)
    if os.path.splitext(cfg_fname)[-1] == ".ini":
        cfg = read_ini(cfg_fname)
    elif os.path.splitext(cfg_fname)[-1] == ".yaml":
        with open(cfg_fname, "r") as fp:
            cfg = yaml.load(fp, Loader=yaml.FullLoader)
    else:
        raise RuntimeError("File not supported.")

    return cfg


def overrides(conf, args):
    def value(v, default_v):
        return v if v is not None else default_v

    conf["config"]["name"] = value(args.name, conf["config"]["name"])
    conf["config"]["epochs"] = value(args.epoch, conf["config"]["epochs"])

    return conf


if __name__ == "__main__":
    args = parse()

    cfg = fetch_config(args.conf)
    cfg = overrides(cfg, args)
    md_conf = cfg["md_conf"]
    md_name = cfg["config"]["name"]
    tables.print() if args.train else None
    cprint.r(f"current: {md_name}")
    model = tables.models.get(md_name)
    assert model is not None
    net = model(**md_conf)

    train_dset, valid_dset, vtest_dset = get_datasets(cfg["dataset"])

    if args.train:
        cfg["config"]["info_dir"] = f'{cfg["config"]["info_dir"]}'

        init = cfg["config"]
        eng = Train(
            train_dset,
            valid_dset,
            vtest_dset,
            net=net,
            net_D=Discriminator(ndf=16),
            batch_sz=2,
            valid_first=args.valid_first,
            **init,
        )
        print(eng)

        eng.fit()

    elif args.pred:  # pred
        assert args.ckpt is not None
        assert args.out is not None
        net.load_state_dict(torch.load(args.ckpt))
        net.cuda()
        net.eval()

        if args.valid:
            dset = valid_dset
        elif args.vtest:
            dset = vtest_dset
        else:
            raise RuntimeError("not supported.")

        for d, HL, fname in tqdm(dset):
            d = d.cuda()
            HL = HL.cuda()

            with torch.no_grad():
                out = net(d, HL)
            out = out.cpu().detach().squeeze().numpy()

            fout = os.path.join(args.out, fname)
            outd = os.path.dirname(fout)
            os.makedirs(outd) if not os.path.exists(outd) else None
            audiowrite(fout, out, sample_rate=16000)
