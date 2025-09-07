import json
import random
from pathlib import Path

import mat73
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, random_split

from bci_brake.constants import dist_data_dir

X_DROP_FEATS = ("EOGv", "EOGh", "EMGf", "lead_gas", "lead_brake", "dist_to_lead", "wheel_X", "wheel_Y", "gas", "brake")
# x_drop_feats = ["lead_gas", "lead_brake", "dist_to_lead", "wheel_X", "wheel_Y", "gas", "brake"]

with open(dist_data_dir.joinpath("x_features.json")) as f:
    X_FEATS = json.load(f)


class BrakeDset(Dataset):

    def __init__(
            self,
            x: np.ndarray,
            times_lead_brakes: np.ndarray,
            reaction_times: np.ndarray,
            freq: float,
            p_tps: float = 0.5,
            window_length: int = 2000,
            predict_horizon_length: tuple[int, int] = (100, 500),
            x_alt_feats: tuple[str, ...] | list[str] = X_DROP_FEATS
    ):
        """

        Args:
            x: (X, F)
            times_lead_brakes: (Y, )
            reaction_times: (Y, )
            freq: frequency
            window_length: milliseconds in window
            predict_horizon_length: milliseconds
            p_tps: proportion of TPs
        """

        feat_idxs, self.x_features = zip(*[(idx, feat) for idx, feat in enumerate(X_FEATS) if feat not in x_alt_feats])
        self.x = x[:, feat_idxs]  # iterations

        alt_feat_idxs, self.alt_x_features = zip(*[(idx, feat) for idx, feat in enumerate(X_FEATS) if feat in x_alt_feats])
        self.alt_x = x[:, alt_feat_idxs]

        self.times_lead_brakes = times_lead_brakes
        self.reaction_times = np.minimum(reaction_times, window_length)  # ms
        predict_horizon_length = np.random.randint(predict_horizon_length[0], predict_horizon_length[1], self.reaction_times.shape[0])
        adj_reaction_times = np.minimum(self.reaction_times + predict_horizon_length, window_length)
        self.padding = window_length - adj_reaction_times

        self.n_tps = times_lead_brakes.shape[0]
        self.n = int(self.n_tps / p_tps) if p_tps > 0 else self.n_tps
        self.p_tps = p_tps

        self.freq = freq  # 1 / s
        self.period = 1000 / self.freq  # ms
        self.samples_per_window = int(window_length / self.period)

        self.T = window_length
        self.H = predict_horizon_length


    @staticmethod
    def init_from_p(
            p: Path,
            p_tps: float | list[float],
            seq_split: list[float] | None = None,
            window_length: int = 2000,
            predict_horizon_length: tuple[int, int] = (100, 500),
            x_alt_feats: tuple[str, ...] = X_DROP_FEATS
    ):

        data = mat73.loadmat(p)
        cnt = data["cnt"]
        mrk = data["mrk"]
        freq = cnt["fs"]  # 1 / s

        x = cnt["x"]

        y = np.argmax(mrk['y'], axis=0)
        class_map = mrk["className"]
        y = [class_map[cls_idx] for cls_idx in y]
        times = mrk["time"]
        reaction_times_ = mrk["event"]["react"]  # ms

        times_lead_brakes = []
        reaction_times = []

        for i, cls in enumerate(y):
            n_brake = len(times_lead_brakes)
            n_react = len(reaction_times)

            if n_brake == n_react:
                if cls == "car_brake":
                    times_lead_brakes.append(times[i])

            else:
                if cls == "react_emg":
                    reaction_times.append(reaction_times_[i])

        times_lead_brakes = np.array(times_lead_brakes)
        reaction_times = np.array(reaction_times)

        if len(times_lead_brakes) != len(reaction_times):
            n = min(len(times_lead_brakes), len(reaction_times))
            times_lead_brakes = times_lead_brakes[:n]
            reaction_times = reaction_times[:n]

        n = len(times_lead_brakes)
        dsets = []

        if seq_split is not None:
            if isinstance(p_tps, float):
                p_tps = [p_tps] * len(seq_split)
            if len(p_tps) != len(seq_split):
                raise IndexError("if seq_split is used p_tps must be a list of same size as seq_split")
            seq_split = np.array(seq_split)
            if seq_split.sum() != 1:
                raise ValueError(f"seq_split.sum() != 1, got {seq_split.sum()}")

            ns = (seq_split * n).astype(int)
            ns[0] = ns[0] + n - ns.sum()

            for i in range(ns.shape[0]):
                start = ns[:i].sum()
                stop = start + ns[i]
                p_tps_ = p_tps[i]

                dsets.append(
                    BrakeDset(
                        x=x,
                        times_lead_brakes=times_lead_brakes[start: stop],
                        reaction_times=reaction_times[start: stop],
                        freq=freq,
                        p_tps=p_tps_,
                        window_length=window_length,
                        predict_horizon_length=predict_horizon_length,
                        x_alt_feats=x_alt_feats
                    )
                )

            return tuple(dsets)

        else:
            return BrakeDset(
                x=x,
                times_lead_brakes=times_lead_brakes,
                reaction_times=reaction_times,
                freq=freq,
                p_tps=p_tps,
                window_length=window_length,
                predict_horizon_length=predict_horizon_length
            )

    def __len__(self):
        return self.n

    def _get_random(self):
        n_x = self.x.shape[0]
        while True:
            start = np.random.randint(n_x - self.T / self.period)

            # if start falls between when the lead car brakes and the participant brakes,
            # get a new start and check again

            diff_start_brakes = start * self.period - self.times_lead_brakes
            mask = diff_start_brakes > 0
            diff_start_brakes = diff_start_brakes[mask]
            react_times = self.reaction_times[mask]

            if len(mask) and np.any(diff_start_brakes < react_times):
                continue
            else:
                break

        stop = int(start + self.samples_per_window)
        x = self.x[start: stop, :]
        alt_x = self.alt_x[start: stop, :]

        ms100 = int(100 / self.period)
        # tmp_x = x[:ms100]
        # x = (x - tmp_x.mean(0, keepdims=True)) / tmp_x.std(0, keepdims=True)
        x = x - x[:ms100].mean(0, keepdims=True)

        y = np.zeros(self.samples_per_window)
        reaction_time = np.nan
        return x, y, reaction_time, alt_x

    def __getitem__(self, item):
        if item >= self.n:
            raise StopIteration

        if item >= self.n_tps:
            return self._get_random()

        brake_time = self.times_lead_brakes[item]  # ms
        react_time = self.reaction_times[item]  # ms
        padding = self.padding[item]  # ms
        left_padding = int(random.random() * padding) + self.H[item]  # ms

        start = (brake_time - left_padding) / self.period  # idx
        stop = start + self.T / self.period  # idx

        start = int(start)
        stop = int(stop)

        x_n = self.x.shape[0]
        if stop > x_n:
            offset = stop - x_n
            start -= offset
            stop -= offset

        x = self.x[start: stop, :]
        ms100 = int(100 / self.period)
        # tmp_x = x[:ms100]
        # x = (x - tmp_x.mean(0, keepdims=True)) / tmp_x.std(0, keepdims=True)
        x = x - x[:ms100].mean(0, keepdims=True)

        alt_x = self.alt_x[start: stop, :]

        n_left_padding = int(left_padding / self.period)
        n_right_padding = int((self.T / self.period) - n_left_padding)
        y = np.concat(([0] * n_left_padding, [1] * n_right_padding))

        return x, y, react_time, alt_x
