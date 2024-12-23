import argparse
import os
import sys
import warnings
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml

# from matplotlib import pyplot as plt
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio as si_sdr
from torchmetrics.functional.audio import signal_distortion_ratio as SDR

# from utils.conv_stft_loss import MultiResolutionSTFTLoss
from tqdm import tqdm

from datasets_manager import get_datasets
from models.APC_SNR.apc_snr import APC_SNR_multi_filter
from models.conv_stft import STFT
from models.pase.models.frontend import wf_builder
from models.VADModel import *
from utils.audiolib import audioread, audiowrite
from utils.Engine import Engine
from utils.focal_loss import BCEFocalLoss
from utils.ini_opts import read_ini
from utils.logger import CPrint, cprint
from utils.losses import loss_compressed_mag, loss_pmsqe, loss_sisnr
from utils.metrics import compute_precision_and_recall
from utils.record import REC, RECDepot
from utils.register import tables
from utils.stft_loss import MultiResolutionSTFTLoss
from utils.trunk import VADTrunk, pad_to_longest


class Train(Engine):
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
            collate_fn=pad_to_longest,
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
            collate_fn=pad_to_longest,
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

        self.focal = BCEFocalLoss(gamma=1, alpha=0.7).to(self.device)

        # self.ms_stft_loss = MultiResolutionSTFTLoss(
        #     fft_sizes=[1024, 512, 256],
        #     hop_sizes=[512, 256, 128],
        #     win_lengths=[1024, 512, 256],
        # ).to(self.device)
        # self.ms_stft_loss.eval()

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

    def loss_fn(self, lbl: Tensor, pred: Tensor) -> Dict:
        """
        clean: B,T,1
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

        # specs_enh = self.stft.transform(enh)
        # specs_sph = self.stft.transform(clean)
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

        # loss = sc_loss + mag_loss + pmsqe_score  # + loss_pase  # + 0.05 * sdr_lv

        focal = self.focal(lbl, pred)
        cross = F.binary_cross_entropy(pred, lbl)
        loss = focal + cross

        return {
            "loss": loss,
            "focal": focal.cpu().detach(),
            "cross": cross.cpu().detach(),
            # "sc": sc_loss.detach(),
            # "mag": mag_loss.detach(),
            # "pmsqe": pmsqe_score.detach(),
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
        for mic, vad in pbar:
            mic = mic.to(self.device)  # B,T
            # B,T -> B,T,1
            vad = vad_to_frames(vad, self.nframe, self.nhop)
            vad = vad.to(self.device)  # B,T
            xk_mic = self.stft.transform(mic)  # b,c,t,f

            self.optimizer.zero_grad()
            enh = self.net(xk_mic)
            loss_dict = self.loss_fn(vad, enh)

            loss = loss_dict["loss"]
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(self.net.parameters(), 3, 2)
            self.optimizer.step()

            # record the loss
            losses_rec.update(loss_dict)
            pbar.set_postfix(**losses_rec.state_dict())

        return losses_rec.state_dict()

    def _valid_dsets(self):
        dset_dict = {}
        return dset_dict

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

            metric_dict = self.valid_fn(sph, mic[..., 0], nlen, return_loss=False)
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
            mic = mic.to(self.device)  # B,T,1
            sph = sph.to(self.device)  # B,T,1
            nlen = self.stft.nLen(nlen).to(self.device)  # B,

            metric_dict = self.valid_fn(sph, mic[..., 0], nlen, return_loss=False)
            metric_dict.pop("score")

            # record the loss
            metric_rec.update(metric_dict)
            pbar.set_postfix(**metric_rec.state_dict())

        dset_dict["vtest"] = metric_rec.state_dict()

        return dset_dict

    def valid_fn_vad(
        self, vad: Tensor, est: Tensor, nlen_list: Tensor, return_loss: bool = True
    ) -> Dict:
        """
        vad: B,T,1
        est: B,T,1
        """

        B = vad.size(0)
        np_l_vad, np_l_est = [], []
        precision, recall = [], []

        for i in range(B):
            vad_ = vad[i, : nlen_list[i], :]  # B,T,1
            est_ = est[i, : nlen_list[i], :]  # B,T,1
            np_l_vad.append(vad_.cpu().numpy() > 0.7)  # bool list
            np_l_est.append(est_.cpu().numpy() > 0.7)

            p, r = compute_precision_and_recall(
                np.array(np_l_vad),
                np.array(np_l_est),
            )
            precision.append(p)
            recall.append(r)

        precis = np.array(precision).mean().round(4)
        recall = np.array(recall).mean().round(4)
        state = {
            "score": precis + recall,
            "precision": precis,
            "recall": recall,
        }

        if return_loss:
            loss_dict = self.loss_fn(vad, est)
        else:
            loss_dict = {}

        return dict(state, **loss_dict)

    def prediction_per_epoch(self, epoch):
        outdir = super().prediction_per_epoch(epoch)

        idx = 0
        for mic, fname in self.vpred_dset:
            if idx >= 20:
                break

            mic = mic.to(self.device)  # T,

            xk_mic = self.stft.transform(mic)  # b,c,t,f
            with torch.no_grad():
                est_vad = self.net(xk_mic)  # B,T,1

            # T,
            est_vad = pack_frames_vad(est_vad, self.nframe, self.nhop).cpu().numpy()
            N = min(est_vad.shape[-1], mic.shape[-1])
            audiowrite(
                f"{outdir}/{fname}",
                np.stack([mic.cpu().numpy()[:N], est_vad[:N]], axis=1),
                self.fs,
            )
            idx += 1

    def prediction_per_epoch_batch(self, epoch):
        outdir = super().prediction_per_epoch(epoch)

        idx = 0
        for mic, vad, nlen in self.vtest_loader:
            if idx >= 20:
                break

            mic = mic.to(self.device)  # B,T,6
            vad = vad.to(self.device)  # b,c,t,f
            # nlen = self.stft.nLen(nlen).to(self.device)
            # nlen = nlen.to(self.device)  # B

            nB = mic.shape[0]

            xk_mic = self.stft.transform(mic)  # b,c,t,f
            with torch.no_grad():
                est_vad = self.net(xk_mic)  # B,T,1

            # B,T
            est_vad = pack_frames_vad(est_vad, self.nframe, self.nhop).cpu().numpy()
            N = min(torch.ones_like(nlen) * est_vad.shape[-1], nlen)
            for i in range(nB):
                audiowrite(
                    f"{outdir}/{idx}_mic.wav",
                    np.stack([mic[i].cpu().numpy()[:N], est_vad[:N]], axis=1),
                    self.fs,
                )
                idx += 1

    def valid_fn(
        self, sph: Tensor, enh: Tensor, nlen_list: Tensor, return_loss: bool = True
    ) -> Dict:
        """
        B,T
        """
        sisnr_l = []
        sdr_l = []
        np_l_sph, np_l_enh = [], []

        B = sph.size(0)

        for i in range(B):
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

        for mic, vad, nlen in pbar:
            mic = mic.to(self.device)  # B,T
            vad = vad_to_frames(vad, self.nframe, self.nhop)
            vad = vad.to(self.device)  # B,T,1
            nlen = self.stft.nLen(nlen).to(self.device)
            xk_mic = self.stft.transform(mic)  # b,c,t,f
            with torch.no_grad():
                enh = self.net(xk_mic)  # B,T,1

            metric_dict = self.valid_fn_vad(vad, enh, nlen)

            if draw is True:
                with torch.no_grad():
                    sxk = self.stft.transform(vad)
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
        # dirname = os.path.split(self.vtest_dset.dirname)[-1]
        dirname = self.vtest_dset.dirname
        pbar = tqdm(
            self.vtest_loader,
            ncols=120,
            leave=False,
            desc=f"vTest-{epoch}/{dirname}",
        )
        # vtest_outdir = os.path.join(self.vtest_outdir, dirname)
        # shutil.rmtree(vtest_outdir) if os.path.exists(vtest_outdir) else None

        for mic, vad, nlen in pbar:
            mic = mic.to(self.device)  # B,T,6
            vad = vad_to_frames(vad, self.nframe, self.nhop)
            vad = vad.to(self.device)  # b,c,t,f
            nlen = self.stft.nLen(nlen).to(self.device)
            xk_mic = self.stft.transform(mic)  # b,c,t,f

            with torch.no_grad():
                enh = self.net(xk_mic)

            metric_dict = self.valid_fn_vad(vad, enh, nlen)
            # record the loss
            metric_rec.update(metric_dict)
            # pbar.set_postfix(**metric_rec.state_dict())
            # break

        dirn = {}
        for k, v in metric_rec.state_dict().items():
            if "vtest" in self.raw_metrics and k in self.raw_metrics["vtest"]:
                dirn[k] = {"raw": self.raw_metrics["vtest"][k], "enh": v}
            else:
                dirn[k] = v
        out[dirname] = dirn
        return out

    def _net_flops(self) -> int:
        import copy

        from thop import profile

        x = torch.randn(1, 2, 250, self.nframe // 2 + 1)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="This API is being deprecated")
            flops, _ = profile(
                copy.deepcopy(self.net).cpu(),
                inputs=(x,),
                verbose=False,
            )
        return flops


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", help="test mode", action="store_true")
    parser.add_argument("--train", help="train mode", action="store_true", default=True)
    parser.add_argument("--pred", help="predict mode", action="store_true")

    parser.add_argument("--ckpt", help="ckpt path", type=str)
    parser.add_argument("--epoch", help="epoch", type=int)
    parser.add_argument("--src", help="input directory", type=str)
    parser.add_argument("--out", help="predicting output directory", type=str)
    parser.add_argument("--valid_first", help="valid first", action="store_true")
	parser.add_argument("--root_save_dir", help="root directory of all results", type=str)

    parser.add_argument("--valid", help="input directory", action="store_true")
    parser.add_argument("--vtest", help="input directory", action="store_true")
    parser.add_argument("--draw", help="input directory", action="store_true")

    parser.add_argument("--conf", help="config file")
    parser.add_argument("--name", help="name of the model")

    args = parser.parse_args()
    args.train = False if args.test or args.pred else True
    return args


@dataclass
class Eng_conf:
    name: str = "crnn_vad"
    epochs: int = 100
    desc: str = ""
    info_dir: str = "trained_infant_vad"
    resume: bool = True
    optimizer_name: str = "adam"
    scheduler_name: str = "stepLR"
    valid_per_epoch: int = 1
    vtest_per_epoch: int = 5  # 0 for disabled
    ## the output dir to store the predict files of `vtest_dset` during testing
    vtest_outdir: str = "vtest"
    dsets_raw_metrics: str = "dset_metrics.json"
    train_batch_sz: int = 14
    train_num_workers: int = 6
    valid_batch_sz: int = 24
    valid_num_workers: int = 4
    vtest_batch_sz: int = 24
    vtest_num_workers: int = 4


@dataclass
class Md_conf:
    # nframe: int = 512
    # nhop: int = 256
    feat_size: int = 257


@dataclass
class Conf:
    config: Eng_conf = Eng_conf()
    md_conf: Md_conf = Md_conf()

    train_dset: str = "/home/deepni/trunk/infant/train"
    valid_dset: str = "/home/deepni/trunk/infant/valid"
    vtest_dset: str = "/home/deepni/trunk/infant/test"
    vpred_dset: str = "/home/deepni/trunk/cry-vad/test"


def fetch_config(cfg_fname=None):
    if cfg_fname is None:
        return asdict(Conf())

    print("##", cfg_fname)
    if os.path.splitext(cfg_fname)[-1] == ".ini":
        cfg = read_ini(cfg_fname)
    elif os.path.splitext(cfg_fname)[-1] == ".yaml":
        with open(cfg_fname, "r") as fp:
            cfg = yaml.load(fp, Loader=yaml.FullLoader)
    else:
        raise RuntimeError("File not supported.")

    return cfg


def re_config(conf, args):
    def value(v, default_v):
        return v if v is not None else default_v

    conf["config"]["name"] = value(args.name, conf["config"]["name"])
    conf["config"]["epochs"] = value(args.epoch, conf["config"]["epochs"])

    return conf


if __name__ == "__main__":
    args = parse()

    cfg = fetch_config(args.conf)
    cfg = re_config(cfg, args)

    md_conf = cfg["md_conf"]
    md_name = cfg["config"]["name"]

    tables.print() if args.train else None
    cprint.r(f"current: {md_name}")

    model = tables.models.get(md_name)
    assert model is not None
    net = model(**md_conf)

    if args.train:
        init = cfg["config"]

        train_dset = VADTrunk(dirname=cfg["train_dset"], flist="infant_train.csv")
        valid_dset = VADTrunk(dirname=cfg["valid_dset"], flist="infant_valid.csv")
        vtest_dset = VADTrunk(dirname=cfg["vtest_dset"], flist="infant_test.csv")
        vpred_dset = VADTrunk(dirname=cfg["vpred_dset"], flist="infant_vpred.csv")

        init = cfg["config"]
        eng = Train(
            train_dset,
            valid_dset,
            vtest_dset,
            # vpred_dset=vpred_dset,
            net=net,
            valid_first=args.valid_first,
			root_save_dir=args.root_save_dir,
            nframe=512,
            nhop=256,
            **init,
        )
        print(eng)

        if args.test:
            eng.test()
        else:
            eng.fit()

    else:  # pred
        assert args.ckpt is not None
        assert args.src is not None
        net.load_state_dict(torch.load(args.ckpt))
        net.cuda()
        net.eval()
        d, fs = audioread(args.src)
        d = d[None, :]
        d = torch.from_numpy(d).float().cuda()
        with torch.no_grad():
            out = net(d)
        out = out.cpu().squeeze().detach()
        out = out.numpy()
        print(out.shape)
        audiowrite("bout.wav", out, sample_rate=fs)
