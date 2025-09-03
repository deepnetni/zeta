import abc
import json
import os
import random
from collections import Counter
from itertools import repeat
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, cast

import matplotlib
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional.audio.sdr import signal_distortion_ratio as SDR
from tqdm import tqdm

from comps.conv_stft import STFT
from utils.audiolib import audiowrite
from utils.trunk_v2 import TrunkBasic

# import re
# from collections import Counter
# from utils.audiolib import audioread

matplotlib.use("Agg")

import numpy as np
import torch
import torch.nn as nn


class Status:
    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        print(self.msg, end="", flush=True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("\r" + " " * len(self.msg), end="\r", flush=True)


def setup_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # set seed for CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # set seed for current GPU
        torch.cuda.manual_seed_all(seed)  # set seed for all GPUs
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    # torch.use_deterministic_algorithms(True, warn_only=True)


setup_seed()

from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from torch.optim import Optimizer, lr_scheduler

# from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from utils.composite_metrics import eval_composite
from utils.logger import get_logger
from utils.metrics import *

log = get_logger("eng", mode="console")


class _EngOpts(object):
    def __init__(self) -> None:
        pass

    def _si_snr(
        self,
        sph: Union[torch.Tensor, np.ndarray],
        enh: Union[torch.Tensor, np.ndarray],
        zero_mean: bool = True,
    ) -> np.ndarray:
        """return numpy or torch according to the type of the input"""
        if isinstance(sph, torch.Tensor):
            sph = sph.cpu().detach().numpy()
        if isinstance(enh, torch.Tensor):
            enh = enh.cpu().detach().numpy()

        sph: np.ndarray
        enh: np.ndarray
        return compute_si_snr(sph, enh, zero_mean)

    def _snr(
        self,
        sph: Union[np.ndarray, list, torch.Tensor],
        enh: Union[np.ndarray, list, torch.Tensor],
        njobs: int = 10,
    ) -> np.ndarray:
        """element-wise parallel calculation"""
        scores = np.array(
            Parallel(n_jobs=njobs)(delayed(compute_snr)(s, e) for s, e in zip(sph, enh))
        )
        return scores

    def _pesq(
        self,
        sph: Union[np.ndarray, list, torch.Tensor],
        enh: Union[np.ndarray, list, torch.Tensor],
        fs: int,
        norm: bool = False,
        njobs: int = 10,
        mode: str = "wb",
    ) -> np.ndarray:
        """element-wise parallel calculation"""
        if isinstance(sph, torch.Tensor):
            sph = sph.cpu().detach().numpy()
        if isinstance(enh, torch.Tensor):
            enh = enh.cpu().detach().numpy()

        scores = np.array(
            Parallel(n_jobs=njobs)(
                delayed(compute_pesq)(s, e, fs, norm, mode) for s, e in zip(sph, enh)
            )
        )
        return scores

    def _stoi(
        self,
        sph: Union[np.ndarray, list, torch.Tensor],
        enh: Union[np.ndarray, list, torch.Tensor],
        fs: int,
        njobs: int = 10,
    ) -> np.ndarray:
        """element-wise parallel calculation"""
        if isinstance(sph, torch.Tensor):
            sph = sph.cpu().detach().numpy()
        if isinstance(enh, torch.Tensor):
            enh = enh.cpu().detach().numpy()

        scores = np.array(
            Parallel(n_jobs=njobs)(delayed(compute_stoi)(s, e, fs) for s, e in zip(sph, enh))
        )
        return scores

    def _eval(
        self,
        sph: Union[torch.Tensor, np.ndarray],
        enh: Union[torch.Tensor, np.ndarray],
        fs: int,
        njobs: int = 10,
    ) -> Dict:
        """Compute pesq, csig, cbak, and covl metrics"""
        if isinstance(sph, torch.Tensor):
            sph = sph.cpu().detach().numpy()
            enh = enh.cpu().detach().numpy()

        scores = Parallel(n_jobs=njobs)(
            delayed(eval_composite)(s, e, fs) for s, e in zip(sph, enh)
        )  # [{"pesq":..},{...}]
        # score = Counter(scores[0])
        # for s in scores[1:]:
        #     score += Counter(s)
        out = {}
        for _ in scores:
            for k, v in _.items():
                out.setdefault(k, []).append(v)
        return {k: np.array(v) for k, v in out.items()}  # {"pesq":,"csig":,"cbak","cvol"}

    def merge_metric(self, *args, **kwargs):
        """
        - Params:
            args: dict, a,b,c,d,...
            kwarge: tag, ...
            - tag: [ta,tb, ...]

        - Example
            {'a':x, 'b':y}, {'a':z, 'b':k}, tags=('na', 'nb')
            return {'a':{'na':x, 'nb':z}, 'b': {'na':y, 'nb':k}}


            if {'a':{...}, 'b':{...}}, {'a':{...}}, then
            return {'a_t1':{...}, 'a_t2':{...}, 'b':{...}}
        """
        tag_l = kwargs.get("tags", None)
        assert tag_l is None or len(tag_l) == len(args)
        keys_count = Counter(k for d in args for k in d)

        state_dict = {}
        for idx, meta in enumerate(args):
            for k, v in meta.items():
                # print(idx, k, v)
                if k not in state_dict and keys_count[k] == 1:
                    state_dict.setdefault(k, v)
                else:
                    t = idx if not tag_l else tag_l[idx]
                    if not isinstance(v, dict):
                        state_dict.setdefault(k, {}).update({t: v})
                    else:
                        # state_dict.setdefault(k, {}).update({f"{k}_{t}": v})
                        state_dict.setdefault(f"{k}_{t}", v)
                # print(state_dict)

        return state_dict


def pad_to_longest_each_element(batch):
    """
    batch: [(mic, ref, label), (...), ...]
    the input data, label must with shape (T,C) if time domain
    """
    # x[0] => mic => mic.shape[0] (T)
    # batch.sort(key=lambda x: x[0].shape[0], reverse=True)  # data length

    out = []
    seq_len = [ele[0].size(0) for ele in batch]
    for ele in zip(*batch):
        ele = pad_sequence(ele, batch_first=True).float()
        out.append(ele)

    return *out, torch.tensor(seq_len)


class Engine(_EngOpts):
    def __init__(
        self,
        name: str,
        train_dset: Dataset,
        valid_dset: Dataset,
        vtest_dset: Dataset,
        net: nn.Module,
        epochs: int,
        desc: str = "",
        info_dir: str = "",
        resume: bool = False,
        optimizer_name: str = "adam",
        scheduler_name: str = "stepLR",
        seed: int = 0,
        valid_per_epoch: int = 1,
        vtest_per_epoch: int = 0,
        valid_first: bool = False,
        dsets_raw_metrics: str = "",
        root_save_dir: Optional[str] = None,
        vpred_dset: Optional[Dataset] = None,
        **kwargs,
    ):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net: nn.Module = net.to(self.device)
        self.fs = kwargs.get("fs", 16000)
        self.lr = kwargs.get("lr", 5e-4)
        self.ncol = kwargs.get("ncol", 160)
        self.opt_lr_step_size = kwargs.get("step_size", 30)
        self.opt_lr_gamma = kwargs.get("gamma", 0.5)

        collate_fn = kwargs.get("dset_collate_fn", pad_to_longest_each_element)

        self.kwargs = kwargs

        self.train_loader = DataLoader(
            train_dset,
            batch_size=kwargs.get("train_batch_sz", 6),
            num_workers=kwargs.get("train_num_workers", 6),
            pin_memory=True,
            shuffle=True,
            worker_init_fn=self._worker_set_seed,
            generator=self._set_generator(),
            drop_last=True,
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
            collate_fn=collate_fn,
            # generator=g,
            drop_last=True,
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
            collate_fn=collate_fn,
            # generator=g,
            drop_last=True,
        )
        # vtest_dset = cast(TrunkBasic, vtest_dset)
        self.vtest_dset = vtest_dset

        if vpred_dset is None:
            self.vpred_dset = vtest_dset
        else:
            self.vpred_dset = vpred_dset

        # loss_dict: {}
        self.loss_list = kwargs.get("loss_list", {})
        for l_name, loss_dict in self.loss_list.items():
            assert "func" in loss_dict, f"loss {l_name} func is None"
            loss_dict["func"].cuda()
            loss_dict["func"].eval()

        self.optimizer = self._config_optimizer(
            optimizer_name, filter(lambda p: p.requires_grad, self.net.parameters())
        )
        self.scheduler = self._config_scheduler(scheduler_name, self.optimizer)
        self.epochs = epochs
        self.start_epoch = 1
        self.valid_per_epoch = valid_per_epoch
        self.vtest_per_epoch = vtest_per_epoch
        self.best_score = torch.finfo(torch.float32).min
        self.seed = seed
        # self.dsets_metrics = self._load_dset_valid_info()

        # checkpoints
        if os.path.isabs(info_dir):  # abspath
            self.info_dir = Path(info_dir)
        else:  # relative path
            self.info_dir = (
                Path(__file__).parent.parent / info_dir
                if info_dir != ""
                else Path(__file__).parent.parent / "trained"
            )

        name = name if not root_save_dir else root_save_dir
        self.base_dir = self.info_dir / name
        log.info(f"\033[92minfo dirname: {self.base_dir}\033[0m")
        self.ckpt_dir = self.info_dir / name / "checkpoints"
        self.ckpt_file = self.ckpt_dir / "ckpt.pth"
        self.ckpt_best_file = self.ckpt_dir / "best.pth"
        self.valid_first = valid_first
        self.vtest_first = kwargs.get("vtest_first", False)
        self.dsets_mfile = (
            self.info_dir / dsets_raw_metrics
            if dsets_raw_metrics != ""
            else self.info_dir / name / "dset_metrics.json"
        )

        self.ckpt_dir.mkdir(parents=True, exist_ok=True)  # create directory if not exists

        if resume is True:
            self._load_ckpt() if self.ckpt_file.exists() else log.warning(
                f"ckpt file: {self.ckpt_file} is not existed."
            )
            self.resume = True if self.ckpt_file.exists() else False

        # tensorboard
        self.tfb_dir = self.info_dir / name / "tfb"
        self.writer = SummaryWriter(log_dir=self.tfb_dir.as_posix())
        self.writer.add_text("Description", desc, global_step=1)
        # epoch results
        self.epoch_pred_dir = self.info_dir / name / "per_epoch"

        self.stft = STFT(512, 256).cuda()

        self.raw_metrics = self._load_dsets_metrics(self.dsets_mfile)

    @property
    def baseDir(self):
        return str(self.base_dir)

    @staticmethod
    def _worker_set_seed(worker_id):
        np.random.seed(worker_id)
        random.seed(worker_id)

    @staticmethod
    def _set_generator(seed: int = 0) -> torch.Generator:
        # make sure the dataloader return the same series under different PCs
        # torch.manual_seed(seed if seed is not None else self.seed)
        g = torch.Generator()
        g.manual_seed(seed)
        return g

    @staticmethod
    def flatten_dict(d, parent_key="", sep="_"):
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(Engine.flatten_dict(v, new_key, sep=sep))
            else:
                items[new_key] = v
        return items

    @staticmethod
    def pbar_postfix_color(pbar, show: Dict, color: str = "green"):
        cmap = dict(
            red="\033[91m",
            green="\033[92m",
            yellow="\033[93m",
            reset="\033[0m",
        )
        show = Engine.flatten_dict(show)
        pbar.set_postfix_str(
            ", ".join(f"{cmap[color]}{k}={v:>.3f}{cmap['reset']}" for k, v in show.items())
        )

    def loss_fn_list(self, sph, enh) -> Dict:
        loss = torch.tensor(0.0).to(sph.device)
        loss_dict = {}
        assert len(self.loss_list) != 0, "without loss funtion."

        sph_xk, enh_xk = None, None

        for name, item in self.loss_list.items():
            # name = item["name"]
            weight = item.get("w", 1.0)

            if weight == 0:
                continue

            if item["func"].domain == "time":
                ret = item["func"](sph, enh)
            else:
                if sph_xk is None:
                    sph_xk = self.stft.transform(sph)
                if enh_xk is None:
                    enh_xk = self.stft.transform(enh)
                ret = item["func"](sph_xk, enh_xk)

            if not isinstance(ret, Tuple):
                loss = loss + weight * ret
                loss_dict.update({name: (weight * ret).detach()})
            else:
                # return v, meta
                lv, meta = ret
                loss = loss + weight * lv
                loss_dict.update({name: (weight * lv).detach(), **meta})

        # loss_dict.update({"loss": loss})
        return dict(loss=loss, **loss_dict)

    def valid_fn_list(self, sph, enh, nlen_list, ret_loss: bool = True) -> Dict:
        nB = sph.size(0)
        # `numel()` return the total number of elements in a Tensor
        if nlen_list.unique().numel() == 1:
            sph = sph[..., : nlen_list[0]]
            enh = enh[..., : nlen_list[0]]
            np_l_sph = sph.cpu().numpy()
            np_l_enh = enh.cpu().numpy()

            sisnr_sc = self._si_snr(sph, enh).mean()
            sdr_sc = SDR(preds=enh, target=sph).mean().item()
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

        pesq_wb_sc = self._pesq(np_l_sph, np_l_enh, fs=16000).mean()
        pesq_nb_sc = self._pesq(np_l_sph, np_l_enh, fs=16000, mode="nb").mean()
        stoi_sc = self._stoi(np_l_sph, np_l_enh, fs=16000).mean()

        state = {
            "pesq_wb": pesq_wb_sc,
            "pesq_nb": pesq_nb_sc,
            "si-snr": sisnr_sc,
            "sdr": sdr_sc,
            "stoi": stoi_sc,
        }

        if ret_loss:
            N = nlen_list.min()
            loss_dict = self.loss_fn_list(sph[..., :N], enh[..., :N])
        else:
            loss_dict = {}

        return dict(state, vloss={k: v.item() for k, v in loss_dict.items()})

    def fit(self):
        for i in range(self.start_epoch, self.epochs + 1):
            if self.valid_first is False and self.vtest_first is False:
                self.net.train()

                loss = self._fit_each_epoch(i)
                torch.cuda.empty_cache()
                self.scheduler.step()
                self._print("Loss", loss, i)
                self._save_ckpt(i, is_best=False)

            with Status("Predicting ..."):
                self.prediction_per_epoch(i)

            self.valid_first = False
            if not self.vtest_first and self.valid_per_epoch != 0 and i % self.valid_per_epoch == 0:
                self.net.eval()
                score: Dict = self._valid_each_epoch(i)

                # vloss_vals = {k: score.pop(k) for k in list(score.keys()) if k.startswith("vloss")}
                # self._print("zEvaLoss", vloss_vals, i)

                vloss_keys = [k for k in score.keys() if k.startswith("vloss")]
                for k in vloss_keys:
                    vloss = score.pop(k)
                    self._print(f"zEval_{k}", vloss, i)
                self._print("Eval", score, i)
                if "score" in score and score["score"] > self.best_score:
                    self.best_score = score["score"]
                    self._save_ckpt(i, is_best=True)

            self.vtest_first = False
            if self.vtest_per_epoch != 0 and (i % self.vtest_per_epoch == 0 or i < 5):
                self.net.eval()
                scores = self._vtest_each_epoch(i)
                for name, score in scores.items():
                    out = ""
                    # score {"-5":{"pesq":v,"stoi":v},"0":{...}}
                    vloss_keys = [k for k in score.keys() if k.startswith("vloss")]
                    for k in vloss_keys:
                        vloss = score.pop(k)
                        self._print(f"zTest-{name}-Loss", vloss, i)

                    for k, v in score.items():
                        out += f"{k}:{v} " + "\n"
                    self.writer.add_text(f"Test-{name}", out, i)
                    self._print(f"Test-{name}", score, i)

    def test(self, epoch: Optional[int] = None):
        if epoch is None:
            epoch = self.start_epoch - 1 if self.valid_first is False else self.start_epoch
        self.net.eval()
        scores = self._vtest_each_epoch(epoch)
        for name, score in scores.items():
            self.writer.add_text(f"Test-{name}", json.dumps(score), epoch)
            self._print(f"Test-{name}", score, epoch)
            print(f"Test-{name}", score, epoch)

    def check_grads(self, loss_dict):
        for name, param in reversed(list(self.net.named_parameters())):
            if param.grad is not None and torch.isnan(param.grad).any():
                raise RuntimeError(f"{loss_dict}, {name}")

    def _config_optimizer(self, name: str, params, **kwargs) -> Optimizer:
        alpha = kwargs.get("alpha", 1.0)
        supported = {
            "adam": lambda p: torch.optim.Adam(p, lr=alpha * self.lr, amsgrad=False),
            "adamw": lambda p: torch.optim.AdamW(p, lr=alpha * self.lr, amsgrad=False),
            "rmsprop": lambda p: torch.optim.RMSprop(p, lr=alpha * self.lr),
        }
        return supported[name](params)

    def _config_scheduler(self, name: str, optimizer: Optimizer):
        supported = {
            "stepLR": lambda p: lr_scheduler.StepLR(
                p, step_size=self.opt_lr_step_size, gamma=self.opt_lr_gamma
            ),
            "reduceLR": lambda p: lr_scheduler.ReduceLROnPlateau(
                p, mode="min", factor=0.5, patience=1
            ),
        }
        return supported[name](optimizer)

    def _load_dsets_metrics(self, fname: Optional[Path] = None) -> Dict:
        """load dataset metrics provided by `self._valid_dsets`"""
        metrics = {}
        fname = self.dsets_mfile if fname is None else fname

        if fname.exists() is True:
            with open(str(fname), "r") as fp:
                metrics = json.load(fp)
        else:  # file not exists
            metrics = self._valid_dsets()
            with open(str(fname), "w+") as fp:
                json.dump(metrics, fp, indent=2)

        return metrics

    def _draw_spectrogram(self, epoch, *args, **kwargs):
        """
        draw spectrogram with args

        :param args: (xk, xk, ...), xk with shape (b,2,t,f) or (2,t,f)
        :param kwargs: fs
        :return:
        """
        N = len(args)
        fs = kwargs.get("fs", 0)
        titles = kwargs.get("titles", repeat(None))

        fig, ax = plt.subplots(N, 1, constrained_layout=True, figsize=(16.0, 9.0))
        for xk, axi, title in zip(args, ax.flat, titles):
            xk = xk.cpu().detach().numpy() if isinstance(xk, torch.Tensor) else xk

            if xk.ndim > 3:  # B,C,T,F
                r, i = xk[0, 0, ...], xk[0, 1, ...]  # r, i shape t,f
            else:  # C,T,F
                r, i = xk[0, ...], xk[1, ...]

            mag = (r**2 + i**2) ** 0.5
            spec = 10 * np.log10(mag**2 + 1e-10).transpose(1, 0)  # f,t

            if fs != 0:
                nbin = spec.shape[0]
                ylabel = np.arange(
                    0, fs // 2 + 1, 1000 if fs <= 16000 else 3000
                )  # 1000, 2000, ..., Frequency
                yticks = nbin * ylabel * 2 // fs
                axi.set_yticks(yticks)
                axi.set_yticklabels(ylabel)

            axi.set_title(title) if title is not None else None
            axi.imshow(spec, origin="lower", aspect="auto", cmap="jet")

        self.writer.add_figure(
            f"spectrogram/{epoch}", fig, global_step=None, close=True, walltime=None
        )

    def __str__(self):
        content = "\n"
        ncol = 6
        total, trainable, total_sz = self._net_info()
        content += "=" * 60 + "\n"
        content += f"{'ckpt':<{ncol}}: {self.ckpt_file}\n"
        content += f"{'Total':<{ncol}}: {total_sz/1024**2:.3f} MB\n"
        content += f"{'nTotal':<{ncol}}: {total:<{ncol},d}\n"
        content += f"nTrainable: {trainable: <{ncol},d}, "
        content += f"nNon-Trainable: {total-trainable: <{ncol},d}\n"

        try:
            flops = self._net_flops()
            # content += f"FLOPS: {flops / 1024**3:.3f}G\n"
            content += f"{'FLOPS':<{ncol}}: {flops / 1e9:.3f}G\n"
        except NotImplementedError:
            # content += "\n"
            pass

        content += "=" * 60

        if not self.resume:
            self.writer.add_text("Params", content, global_step=1)

        return content

    def _net_info(self):
        total = sum(p.numel() for p in self.net.parameters())
        trainable = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        size = sum(p.numel() * p.element_size() for p in self.net.parameters())
        return total, trainable, size

    def _save_ckpt(self, epoch, is_best=False):
        """Could be overwritten by the subclass"""

        if is_best:
            torch.save(self.net.state_dict(), self.ckpt_best_file)
        else:
            state_dict = {
                "epoch": epoch,
                "best_score": self.best_score,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "net": self.net.state_dict(),
            }
            state_dict = self.post_save_ckpt(state_dict)
            torch.save(state_dict, self.ckpt_file)
            torch.save(state_dict, self.ckpt_dir / f"epoch_{str(epoch).zfill(4)}.pth")

    def _load_ckpt(self, epoch=None):
        if not epoch:  # epoch is None
            ckpt = torch.load(self.ckpt_file, map_location=self.device)
        else:
            ckpt_file = self.ckpt_dir / f"epoch_{str(epoch).zfill(4)}.pth"
            log.info(f"Loading {ckpt_file}")
            ckpt = torch.load(ckpt_file, map_location=self.device)

        if self.valid_first or self.vtest_first:
            self.start_epoch = ckpt["epoch"]
        else:
            self.start_epoch = ckpt["epoch"] + 1

        self.best_score = ckpt["best_score"]
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.net.load_state_dict(ckpt["net"])

    def _print(self, tag: str, state_dict: Dict, epoch: int):
        """
        :param state_dict: {"loss1":1, "loss2":2} or {"i1":{"k1":v1,"k2":v2},"i2":{..}}
        :param epoch:
        :return:
        """
        for k, v in state_dict.items():
            if isinstance(v, dict):
                self.writer.add_scalars(f"{tag}/{k}", v, epoch)
            else:
                self.writer.add_scalar(f"{tag}/{k}", v, epoch)

    def post_save_ckpt(self, ckpt_dict):
        return ckpt_dict

    def eval_epoch(self, epoch: int | List):
        """Validate a given epoch ckpt result."""
        if isinstance(epoch, List):
            score = {}
            for i in epoch:
                self._load_ckpt(i)
                ret: Dict = self._valid_each_epoch(i)
                score.update({i: ret})
        else:
            self._load_ckpt(epoch)
            score: Dict = self._valid_each_epoch(epoch)

        return score

    @abc.abstractmethod
    def _net_flops(self) -> int:
        # from thop import profile
        # import copy
        # x = torch.randn(1, 16000)
        # flops, _ = profile(copy.deepcopy(self.net), inputs=(x,), verbose=False)
        # return flops
        pass
        # raise NotImplementedError

    def _valid_dsets(self) -> Dict:
        """return metrics of valid & test dataset
        Return:
            {'valid':{"pesq":xx, "STOI":xx,...}, "test":{...}}
        """
        print("!! valid dset funsion not defined.")
        return {}

    def prediction_per_epoch(self, epoch):
        self.net.eval()
        return str(self.epoch_pred_dir / str(epoch))

    @abc.abstractmethod
    def _fit_each_epoch(self, epoch: int) -> Dict:
        pass
        # raise NotImplementedError
        # return {"loss": 0}

    @abc.abstractmethod
    def _valid_each_epoch(self, epoch: int) -> Dict:
        pass
        # raise NotImplementedError
        # return {"score": 0}

    @abc.abstractmethod
    def _vtest_each_epoch(self, epoch: int) -> Dict[str, Dict[str, Dict]]:
        """
        {"dir1":{"metric":v,..}, "d2":{..}} or
        {"dir1":{"subd1":{"metric":v,...},"sub2":{...}}, "dir2":{...}}
        """
        pass


class EngineGAN(Engine):
    def __init__(
        self,
        name: str,
        train_dset: Dataset,
        valid_dset: Dataset,
        vtest_dset: Dataset,
        net: nn.Module,
        net_D: nn.Module,
        epochs: int,
        desc: str = "",
        info_dir: str = "",
        resume: bool = False,
        optimizer_name: str = "adam",
        scheduler_name: str = "stepLR",
        seed: int = 0,
        valid_per_epoch: int = 1,
        vtest_per_epoch: int = 0,
        valid_first: bool = False,
        dsets_raw_metrics: str = "",
        root_save_dir: Optional[str] = None,
        vpred_dset: Optional[Dataset] = None,
        *args,
        **kwargs,
    ):
        # fmt: off
        super().__init__(name, train_dset, valid_dset, vtest_dset, net, epochs,
            desc, info_dir, resume, optimizer_name, scheduler_name, seed, valid_per_epoch,
            vtest_per_epoch, valid_first, dsets_raw_metrics, root_save_dir, vpred_dset,
            *args, **kwargs,
        )
        # fmt: on

        self.net_D = net_D.to(self.device)
        self.optimizer_D = self._config_optimizer(
            optimizer_name, filter(lambda p: p.requires_grad, self.net_D.parameters())
        )
        self.scheduler_D = self._config_scheduler(scheduler_name, self.optimizer_D)

        self.post_load_ckpt() if self.ckpt_file.exists() else False

    def post_save_ckpt(self, ckpt_dict):
        # dict.update is a inplace operation.
        ckpt_dict.update(
            {
                "net_D": self.net_D.state_dict(),
                "optimizer_D": self.optimizer_D.state_dict(),
                "scheduler_D": self.scheduler_D.state_dict(),
            }
        )
        return ckpt_dict

    def post_load_ckpt(self):
        ckpt = torch.load(self.ckpt_file, map_location=self.device)
        self.optimizer_D.load_state_dict(ckpt["optimizer_D"])
        self.scheduler_D.load_state_dict(ckpt["scheduler_D"])
        self.net_D.load_state_dict(ckpt["net_D"])

    def fit(self):
        for i in range(self.start_epoch, self.epochs + 1):
            if self.valid_first is False and self.vtest_first is False:
                self.net.train()
                self.net_D.train()

                loss = self._fit_each_epoch(i)
                self.scheduler.step()
                self.scheduler_D.step()
                self._print("Loss", loss, i)
                self._save_ckpt(i, is_best=False)

                self.prediction_per_epoch(i)

            self.valid_first = False
            if not self.vtest_first and self.valid_per_epoch != 0 and i % self.valid_per_epoch == 0:
                self.net.eval()
                self.net_D.eval()
                score = self._valid_each_epoch(i)
                vloss = score.pop("vloss", None)
                self._print("zEvaLoss", vloss, i) if vloss is not None else None
                self._print("Eval", score, i)
                if "score" in score and score["score"] > self.best_score:
                    self.best_score = score["score"]
                    self._save_ckpt(i, is_best=True)

            self.vtest_first = False
            if self.vtest_per_epoch != 0 and (i % self.vtest_per_epoch == 0 or i < 5):
                self.net.eval()
                scores = self._vtest_each_epoch(i)
                for name, score in scores.items():
                    out = ""
                    # score {"-5":{"pesq":v,"stoi":v},"0":{...}}
                    vloss = score.pop("vloss", None)
                    self._print(f"zTest-{name}-Loss", vloss, i) if vloss is not None else None
                    for k, v in score.items():
                        out += f"{k}:{v} " + "\n"
                    self.writer.add_text(f"Test-{name}", out, i)
                    self._print(f"Test-{name}", score, i)


class PredEngine(object):
    def __init__(
        self, name, net: nn.Module, ckpt: str, dset: Dataset, info_dir: str = "", fs=16000, **kwargs
    ) -> None:
        """Predictor engine, revised the run API if unavilable.

        :param net:
        :param ckpt: str type for the epoch, or abs path for the specific checkpoint.
        :param dset:
        :param info_dir: the directory of the trained results.
        :param fs:
        :returns:

        """
        root_save_dir = kwargs.get("root_save_dir", "")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net: nn.Module = net
        # self.dset: TrunkBasic = dset
        self.dset: TrunkBasic = cast(TrunkBasic, dset)
        # not root_save_dir, [], {}, None, 0, False, ""
        name = name if not root_save_dir else root_save_dir
        self.info_dir = Path(info_dir) / name
        # self.fout_dir = self.info_dir / "output"
        self.fout_dir = self.info_dir / "output"
        log.info(f"\033[92mpred dirname: {self.fout_dir/ self.dset.dirname}\033[0m")
        self.ckpt = str(ckpt) if isinstance(ckpt, int) else ckpt
        self.fs = fs

        self.load_ckpt()

    def load_ckpt(self):
        if os.path.isabs(self.ckpt) and os.path.isfile(self.ckpt):
            ckpt_f = self.ckpt
        else:
            ckpt_f = self.info_dir / "checkpoints" / f"epoch_{self.ckpt:0>4}.pth"
        log.info(f"\033[92mload ckpt: {ckpt_f}\033[0m")

        # print(ckpt_f, "@@@")
        self.net.load_state_dict(torch.load(ckpt_f)["net"])
        self.net.to(self.device)
        self.net.eval()

    def run(self):
        """
        revise if not suitable.
        """
        pbar = tqdm(self.dset, ncols=100)
        for mic, fname in pbar:
            mic = mic.cuda()

            fout = os.path.join(str(self.fout_dir), fname)

            with torch.no_grad():
                enh = self.net(mic)

            enh = enh.cpu().detach().squeeze().numpy()
            audiowrite(fout, enh, sample_rate=self.fs)

            # try:
            # outd = os.path.dirname(fout)
            # except


if __name__ == "__main__":
    i = {"a": 1, "b": 2, "c": 3}
    j = {"b": 22, "c": 33}
    k = {"e": {"1": 1, "2": 2}, "f": {"3": 4}}
    l = {"e": {"1": 3, "2": 4}, "f": {"3": 4}}

    obj = _EngOpts()
    # met = obj.merge_metric(i, j, tags=("ii", "jj"))
    met = obj.merge_metric(i, j, k, l, tags=("ii", "jj", "kk", "ll"))
    # print(met)
    met = Engine.flatten_dict(met)
    print(met)
