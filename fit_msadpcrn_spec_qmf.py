import argparse
import os
import sys
import warnings
from typing import Dict, List

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
from models.PQMF import PQMF

from models.APC_SNR.apc_snr import APC_SNR_multi_filter
from models.conv_stft import STFT
from models.MSA_DPCRN import MSA_DPCRN_SPEC, MSA_DPCRN_SPEC_online
from models.pase.models.frontend import wf_builder
from utils.audiolib import audioread, audiowrite
from utils.Engine import Engine
from utils.ini_opts import read_ini
from utils.logger import CPrint, cprint
from utils.losses import loss_compressed_mag, loss_pmsqe, loss_sisnr
from utils.record import REC, RECDepot
from utils.register import tables
from utils.stft_loss import MultiResolutionSTFTLoss
from torch.nn.utils.rnn import pad_sequence
from utils.gcc_phat import gcc_phat


def pad_to_longest(batch):
    """
    batch: [(mic, ref, label), (...), ...], B,T,C
    the input data, label must with shape (T,C) if time domain
    """
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)  # data length

    seq_len = [d.size(0) for d, _, _ in batch]
    mic, ref, label = zip(*batch)  # B,T,C
    mic = pad_sequence(mic, batch_first=True).float()
    ref = pad_sequence(ref, batch_first=True).float()
    label = pad_sequence(label, batch_first=True).float()

    # data = pack_padded_sequence(data, seq_len, batch_first=True, enforce_sorted=True)

    return mic, ref, label, torch.tensor(seq_len)


class Train(Engine):
    def __init__(
        self,
        train_dset: Dataset,
        valid_dset: Dataset,
        vtest_dset: Dataset,
        # net_ae: torch.nn.Module,
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

        self.stft = STFT(nframe=128, nhop=64, win="hann sqrt").to(self.device)
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

    def loss_fn(self, clean: Tensor, enh: Tensor) -> Dict:
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
        # pmsqe_score = 0.3 * loss_pmsqe(specs_sph, specs_enh, fs=self.fs)
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

        # loss = sc_loss + mag_loss + pmsqe_score  # + loss_pase  # + 0.05 * sdr_lv
        loss = sc_loss + mag_loss

        return {
            "loss": loss,
            "sc": sc_loss.detach(),
            "mag": mag_loss.detach(),
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
        for mic, ref, sph in pbar:
            mic = mic.to(self.device)
            ref = ref.to(self.device)
            mic_xk = self.stft.transform(mic)
            ref_xk = self.stft.transform(ref)
            sph = sph.to(self.device)

            self.optimizer.zero_grad()
            enh_xk, _ = self.net(mic_xk, ref_xk)
            enh = self.stft.inverse(enh_xk)
            loss_dict = self.loss_fn(sph[:, : enh.size(-1)], enh)

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
            ref = ref.to(self.device)
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
        # pesq_wb_sc = self._pesq(np_l_sph, np_l_enh, fs=self.fs).mean()
        pesq_nb_sc = self._pesq(np_l_sph, np_l_enh, fs=self.fs, mode="nb").mean()
        stoi_sc = self._stoi(np_l_sph, np_l_enh, fs=self.fs).mean()

        # composite = self._eval(clean, enh, 16000)
        # composite = {k: np.mean(v) for k, v in composite.items()}
        # pesq = composite.pop("pesq")

        state = {
            "score": float(pesq_nb_sc + stoi_sc) / 2.0,
            "sisnr": sisnr_sc,
            # "sisnr_pad": sisnr_sc_.cpu().detach().numpy(),
            "sdr": sdr_sc,
            "pesq_nb": pesq_nb_sc,
            "stoi": stoi_sc,
        }

        if return_loss:
            loss_dict = self.loss_fn(sph[..., : enh.size(-1)], enh)
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

        for mic, ref, sph, nlen in pbar:
            mic = mic.to(self.device)
            ref = ref.to(self.device)
            mic_xk = self.stft.transform(mic)
            ref_xk = self.stft.transform(ref)
            sph = sph.to(self.device)
            nlen = self.stft.nLen(nlen).to(self.device)

            with torch.no_grad():
                enh_xk, _ = self.net(mic_xk, ref_xk)
                enh = self.stft.inverse(enh_xk)

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

        for mic, ref, sph, nlen in pbar:
            mic = mic.to(self.device)
            ref = ref.to(self.device)
            mic_xk = self.stft.transform(mic)
            ref_xk = self.stft.transform(ref)
            sph = sph.to(self.device)
            nlen = self.stft.nLen(nlen).to(self.device)

            with torch.no_grad():
                enh_xk, _ = self.net(mic_xk, ref_xk)
                enh = self.stft.inverse(enh_xk)

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

    def _net_flops(self) -> int:
        import copy

        from thop import profile

        x = torch.randn(1, 2, 125, 65)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="This API is being deprecated")
            flops, _ = profile(
                copy.deepcopy(self.net).cpu(),
                inputs=(x, x),
                verbose=False,
            )
        return flops


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="train mode", action="store_true", default=True)
    parser.add_argument("--test", help="test mode", action="store_true")
    parser.add_argument("--pred", help="predict mode", action="store_true")

    parser.add_argument("--ckpt", help="ckpt path", type=str)
    parser.add_argument("--src", help="input directory", type=str, nargs="+")
    parser.add_argument("--out", help="predicting output directory", type=str)
    parser.add_argument("--valid_first", help="valid first", action="store_true")

    parser.add_argument("--vtest", help="input directory", action="store_true")
    parser.add_argument("--valid", help="input directory", action="store_true")

    parser.add_argument("--conf", help="config file")
    parser.add_argument("--name", help="name of the model")
    parser.add_argument("--online", help="frame 2 frame mode", action="store_true")

    args = parser.parse_args()
    args.train = False if args.test or args.pred else True
    return args


def split_to_frames(x, nframe, nhop):
    """
    input: B,T
    """
    pad = nframe // 2

    x = F.pad(x, (pad, pad))

    N = (x.size(-1) // nhop) * nhop
    print(N)

    idx = torch.arange(nframe)
    idx = torch.arange(0, N-nhop, nhop).unsqueeze(-1) + idx
    print(idx.shape)

def synthe_to_waves(x, nframe, nhop):
    pass


if __name__ == "__main__":
    args = parse()

    # ch must be multiply of 4,6
    cfg_fname = "config/yconf_msadpcrn.yaml" if args.conf is None else args.conf
    cprint.b(f"config: { cfg_fname }")

    if os.path.splitext(cfg_fname)[-1] == ".ini":
        cfg = read_ini(cfg_fname)
    elif os.path.splitext(cfg_fname)[-1] == ".yaml":
        with open(cfg_fname, "r") as fp:
            cfg = yaml.load(fp, Loader=yaml.FullLoader)
    else:
        raise RuntimeError("File not supported.")

    net = MSA_DPCRN_SPEC(**cfg["md_conf"])

    if args.train:
        cfg["config"]["info_dir"] = f'{cfg["config"]["info_dir"]}'
        train_dset, valid_dset, test_dset = get_datasets("AECChallenge8k")

        init = cfg["config"]
        eng = Train(
            train_dset,
            valid_dset,
            test_dset,
            net=net,
            batch_sz=2,
            valid_first=args.valid_first,
            fs=8000,
            **init,
        )
        print(eng)

        if args.test:
            eng.test()
        else:
            eng.fit()

    elif args.online:
        assert args.ckpt is not None
        assert args.out is not None
        net = MSA_DPCRN_SPEC_online(**cfg["md_conf"])
        net.load_state_dict(torch.load(args.ckpt))
        net.cuda()
        net.eval()

        qmf = PQMF(2)
        if len(args.src) == 2:
            mic, _ = audioread(args.src[0])
            ref, _ = audioread(args.src[1])
        elif len(args.src) == 1:
            data, _ = audioread(args.src[0])
            mic, ref = data[..., 0], data[..., 1]
        else:
            raise RuntimeError("input format error.")

        align = True
        if align:
            fs = 16000
            tau, _ = gcc_phat(mic, ref, fs=fs, interp=1)
            tau = max(0, int((tau - 0.001) * fs))
            ref = np.concatenate([np.zeros(tau), ref], axis=-1, dtype=np.float32)[
                : mic.shape[-1]
            ]
        else:
            N = min(len(mic), len(ref))
            N = 16000 * 5
            mic = mic[:N]
            ref = ref[:N]

        stft = STFT(nframe=128, nhop=64, win="hann sqrt").cuda()
        # B,2,T
        mic_lh = qmf.analysis(torch.from_numpy(mic).float())
        mic_l, mic_h = mic_lh[:, 0, ...], mic_lh[:, 1, ...]
        # mic_h = mic_h.float().cuda()
        ref_lh = qmf.analysis(torch.from_numpy(ref).float())
        ref_l, ref_h = ref_lh[:, 0, ...], ref_lh[:, 1, ...]

        d_mic = mic_l.cuda()
        d_ref = ref_l.cuda()
        d_mic = stft.transform(d_mic)  # b,2,t,f
        d_ref = stft.transform(d_ref)

        out_list = []
        state = None
        for nt in tqdm(range(d_mic.size(2))):
            mic_frame = d_mic[..., nt, :].unsqueeze(2)
            ref_frame = d_ref[..., nt, :].unsqueeze(2)

            with torch.no_grad():
                # * w: b,2,t,f
                out_frame, w, state = net(mic_frame, ref_frame, state)
                out_list.append(out_frame)

        out = torch.concat(out_list, dim=2)
        out = stft.inverse(out).cpu()  # B,T
        w = w.permute(2, 0, 1, 3).flatten(1, -1).mean(-1)
        print(out.shape, w.shape, mic_h.shape)  # B,T
        out_l = out.cpu().detach().squeeze().numpy()

        mic_h = mic_h[..., : out.size(-1)]
        # B,1,T
        out_full = qmf.synthesis(torch.stack([out, mic_h], dim=1))
        out_full = out_full.squeeze().cpu().numpy()
        print(out_full.shape)

        # split_to_frames(mic_h, 128, 64)

        fout = os.path.join(args.out, "enh_l_torch.wav")
        outd = os.path.dirname(fout)
        os.makedirs(outd) if not os.path.exists(outd) else None
        audiowrite(fout, out_l, sample_rate=8000)

        fout = os.path.join(args.out, "enh_full.wav")
        outd = os.path.dirname(fout)
        audiowrite(fout, out_full, sample_rate=16000)

    else:  # pred
        assert args.ckpt is not None
        assert args.out is not None
        net.load_state_dict(torch.load(args.ckpt))
        net.cuda()
        net.eval()

        # train_dset, valid_dset, test_dset = get_datasets("AECChallenge8k")
        # if args.vtest:
        #     dset = test_dset
        # elif args.valid:
        #     dset = valid_dset
        # else:
        #     raise RuntimeError("not supported.")
        qmf = PQMF(2)
        if len(args.src) == 2:
            mic, _ = audioread(args.src[0])
            ref, _ = audioread(args.src[1])
        elif len(args.src) == 1:
            data, _ = audioread(args.src[0])
            mic, ref = data[..., 0], data[..., 1]
        else:
            raise RuntimeError("input format error.")

        align = True
        if align:
            fs = 16000
            tau, _ = gcc_phat(mic, ref, fs=fs, interp=1)
            tau = max(0, int((tau - 0.001) * fs))
            ref = np.concatenate([np.zeros(tau), ref], axis=-1, dtype=np.float32)[
                : mic.shape[-1]
            ]
        else:
            N = min(len(mic), len(ref))
            mic = mic[:N]
            ref = ref[:N]

        stft = STFT(nframe=128, nhop=64, win="hann sqrt").cuda()
        # B,2,T
        mic_lh = qmf.analysis(torch.from_numpy(mic).float())
        mic_l, mic_h = mic_lh[:, 0, ...], mic_lh[:, 1, ...]
        # mic_h = mic_h.float().cuda()
        ref_lh = qmf.analysis(torch.from_numpy(ref).float())
        ref_l, ref_h = ref_lh[:, 0, ...], ref_lh[:, 1, ...]

        d_mic = mic_l.cuda()
        d_ref = ref_l.cuda()
        d_mic = stft.transform(d_mic)
        d_ref = stft.transform(d_ref)

        with torch.no_grad():
            # * w: b,2,t,f
            out, w = net(d_mic, d_ref)
        out = stft.inverse(out).cpu() # B,T
        w = w.permute(2, 0, 1, 3).flatten(1, -1).mean(-1)
        print(out.shape, w.shape, mic_h.shape) # B,T
        out_l = out.cpu().detach().squeeze().numpy()

        mic_h = mic_h[..., :out.size(-1)]
        # B,1,T
        out_full = qmf.synthesis(torch.stack([out, mic_h], dim=1))
        out_full = out_full.squeeze().cpu().numpy()
        print(out_full.shape)

        # split_to_frames(mic_h, 128, 64)

        fout = os.path.join(args.out, "enh_l.wav")
        outd = os.path.dirname(fout)
        os.makedirs(outd) if not os.path.exists(outd) else None
        audiowrite(fout, out_l, sample_rate=8000)

        fout = os.path.join(args.out, "enh_full.wav")
        outd = os.path.dirname(fout)
        audiowrite(fout, out_full, sample_rate=16000)
