import glob
import logging
import os
import random
import pickle

import numpy as np
import rir_generator as rir
import scipy.signal as sps
import yaml
import json
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

from typing import Dict, Optional, List

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return super().default(obj)


def sph2cart(azimuth, elevation, r):
    """Spherical to Cartesian Conversion

    :param azimuth:
    :param elevation:
    :param r:
    :returns:

    """
    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)  # N,
    return np.stack([x, y, z], axis=-1).round(2)


class RIRGenerator:
    """
    revised from https://github.com/tomkocse/sim-rir-preparation/
    """

    def __init__(
        self,
        array: Dict,
        samples=4096,
        fs=16000,
        **kwargs,
    ) -> None:
        self.conf_rooms = {}
        self.fs = fs
        self.samples = samples
        self.conf_mics = array

        self.noise_num = kwargs.get("noise_num", 1)

        for rtype, conf in kwargs.get("rooms", {}).items():
            # rtype: normal, small, medium
            if conf["prop"] <= 0:
                continue

            # {'w': a, 'h': b, 'prop': c}
            self.conf_rooms[rtype] = conf

        props = np.array([x["prop"] for _, x in self.conf_rooms.items()], dtype="float")
        props /= sum(props)

        self.props = props
        self.rtypes = list(self.conf_rooms.keys())

    @classmethod
    def from_yaml(
        cls,
        conf=os.path.join(os.path.dirname(__file__), "rir_conf.yaml"),
        key: Optional[str] = None,
    ):
        with open(conf) as f:
            conf_dict = yaml.load(f, Loader=yaml.CLoader)

        if key is None:
            return cls(**conf_dict)
        else:
            return cls(**conf_dict[key])

    def _sample_mic_pos(self, room_xyz: np.ndarray):
        if self.conf_mics["mic_position"] == "center":
            mic_xyz = room_xyz / 2 + self._uniform(self.conf_mics["displacement"], size=3)
        elif self.conf_mics["mic_position"] == "random":
            mic_xyz = room_xyz * self._uniform([0, 1], size=3)
        else:
            raise RuntimeError(f'{self.conf_mics["mic_position"]} not supported.')

        mic_xyz[2] = self._uniform(self.conf_mics["mic_h"])

        return (mic_xyz).round(2)

    def _rand(self, size=None):
        # u is (N,) within [0,1)
        return np.random.rand(self.conf_mics["rirs_per_room"] if size is None else size)

    def _uniform(self, dist: List, size=1):
        if size == 1:
            return round(np.random.uniform(*dist), 2)
        else:
            return np.random.uniform(*dist, size=size).round(2)

    def _check_source_pos(self, room_xyz, src_xyz):
        return False if (src_xyz > room_xyz[None, :]).any() or np.any(src_xyz < 0) else True

    def sample(self):
        choice = np.random.choice(self.rtypes, p=self.props)
        conf = self.conf_rooms[choice]

        # * step1. sample room size
        room_x = self._uniform(conf["l"])
        room_y = self._uniform(conf["w"])
        room_z = self._uniform(conf["h"])
        room_xyz = np.array([room_x, room_y, room_z])
        mic_xyz = self._sample_mic_pos(room_xyz)

        if self.conf_mics["absorption_bound"] is not None:
            absorption = self._uniform(self.conf_mics["absorption_bound"])
            # assume all the walls are built with the same material.
            reflection = [np.sqrt(1 - absorption).round(2)] * 6
            rt60 = None
        else:
            rt60 = self._uniform(self.conf_mics["rt60"])
            reflection = None

        meta = dict(
            L=room_xyz,  # room dimensions [x,y,z](m)
            r=mic_xyz,  # receiver position [x,y,z](m); if multi-micros [[m1], [m2], ...]
            beta=reflection,  # using this if configured with rt60
            reverberation_time=rt60,
            nsample=self.samples,  # number of output samples
            order=self.conf_mics["order"],
            hp_filter=self.conf_mics["hp_filter"],
        )

        # step2. sample a source within the sphere.
        pos = []
        dist_l = []
        hrir_l = []
        hrir_norm_l = []
        while True:
            if self.conf_mics["elevation"] is None:
                elevation = np.arcsin(2 * self._rand(1) - 1)  # (N,) [-pi/2, pi/2)
            else:
                elevation = np.arcsin(self._uniform(self.conf_mics["elevation"]))

            azimuth = 2 * np.pi * self._rand(1)
            # radii = self._uniform(self.conf_mics["SMD_bound"])
            # ! Since volume proportional to r^3, use r^(1/3) for uniform sphere fill
            if isinstance(self.conf_mics["SMD_bound"], (float, int)):
                radii = self.conf_mics["SMD_bound"] * self._rand(1) ** (1 / 3)
                print(radii)
            else:
                radii = np.array([self._uniform(self.conf_mics["SMD_bound"])])

            offset_xyz = sph2cart(azimuth, elevation, radii)
            src_xyz = mic_xyz[None, :] + offset_xyz
            src_xyz = src_xyz.round(2)
            if not self._check_source_pos(room_xyz, src_xyz):
                # not valid
                continue

            # * step3.1. generate the rir samples
            hrir = rir.generate(
                c=340,  # sound velocity
                fs=self.fs,
                s=src_xyz.squeeze(),
                **meta,
            )  # (samples, mics)
            # np.power(sh_norm, 2).sum(0) == [1,1,1...]
            hrir_norm = hrir / np.linalg.norm(hrir, axis=0, keepdims=True)

            pos.append(src_xyz)
            dist_l.append(radii)
            hrir_l.append(hrir.squeeze())
            hrir_norm_l.append(hrir_norm.squeeze())
            if len(pos) >= self.conf_mics["rirs_per_room"]:
                break

        meta.update(
            {
                "N": self.conf_mics["rirs_per_room"],
                "source_xyz": np.concatenate(pos, axis=0),
                "SMDist": np.concatenate(dist_l, axis=0).round(2),
                "h": hrir_l,
                "h_norm": hrir_norm_l,
            }
        )

        return meta

    def _draw(self, meta, fname):
        h = meta.pop("h")
        _ = meta.pop("h_norm")
        N = meta["N"]

        fig = plt.figure(figsize=(16, 9))
        gs = gridspec.GridSpec((N + 1) // 2 + 5, 2, figure=fig)

        dist = meta["SMDist"]
        for i in range(N):
            # plt.subplot(N // 2 + 1, 2, i + 1)
            ax = fig.add_subplot(gs[i // 2, i % 2])
            # fidn max
            max_idx = np.argmax(h[i])
            y_max = h[i][max_idx]

            ax.plot(h[i], label=dist[i])
            ax.scatter(max_idx, y_max, color="r")
            ax.text(max_idx, y_max, f"{max_idx}: {y_max:.2f}", color="black", va="bottom")
            ax.set_title(i + 1)
            # ax.set_ylim(top=y_max + 1)
            ax.legend()

        ax3d = fig.add_subplot(gs[-5:, :], projection="3d")  # 4-5, full columns
        mic_p = meta["r"]
        src_p = meta["source_xyz"]

        x, y, z = zip(*src_p)
        ax3d.scatter(x, y, z, color="royalblue", marker="o")
        for i, (x, y, z) in enumerate(src_p, start=1):
            # ax3d.text(x, y, z, f"{i}({x:.2f},{y:.2f})", color="black", fontsize=10)
            ax3d.text(x, y, z, f"{i}({dist[i-1]:.2f})", color="black", fontsize=10)

        ax3d.scatter(*mic_p, color="red", marker="x")

        ax3d.set_title("3D spatial point cloud")
        ax3d.set_xlabel("X")
        ax3d.set_ylabel("Y")
        ax3d.set_zlabel("Z")

        fname = fname.split(".")[0] + ".svg"
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()

    def save(self, meta, fname="out.pkl"):
        with open(fname, "wb") as fp:
            pickle.dump(meta, fp)

        self._draw(meta, fname)

        # print(meta)
        fname = fname.split(".")[0] + ".json"
        with open(str(fname), "w+") as fp:
            json.dump(meta, fp, indent=2, cls=NumpyEncoder)


class RIRDict:
    def __init__(self, config):
        datasets = dict()
        weights = dict()

        for dataset_name, data_config in config.items():
            if "weight" in data_config and data_config["weight"] <= 0:
                continue

            if dataset_name in datasets:
                raise ValueError(f"Duplicate dataset '{dataset_name}'")

            files = list(glob.glob(os.path.join(data_config["dir"], "**/*.pkl"), recursive=True))
            if len(files) > 0:
                datasets[dataset_name] = files
                weights[dataset_name] = data_config["weight"]
            else:
                # raise RuntimeWarning(f"Empty dataset, ignoring: {dataset_name}")
                logger.info(f"Empty dataset, ignoring: {dataset_name}")

        # if len(weights) == 0:
        #     raise ValueError("No datasets are defined.")

        names = list(weights.keys())
        props = np.array(list(weights.values()), dtype="float")
        props /= sum(props)
        self.names = names  # name list
        self.props = props  # weights list
        self.datasets = datasets  # rir list

        # self.rng = np.random.default_rng(random_seed)

    def sample(self):
        """
        return: source h, noise h, meta; all with format `C,L`.
        """
        name = np.random.choice(self.names, p=self.props)
        rir_list = self.datasets[name]
        # sample one rir from the dataset
        rir_p = random.sample(rir_list, 1)[0]

        try:
            # rirs_meta = np.load(str(rir_meta_path), allow_pickle=True)
            with open(rir_p, "rb") as f:
                rirs_meta = pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load {rir_p} from rir dataset")
            raise e

        return rirs_meta["h"], rirs_meta["noise_h"], rirs_meta


if __name__ == "__main__":
    gene = RIRGenerator.from_yaml()
    meta = gene.sample()

    gene.save(meta)
