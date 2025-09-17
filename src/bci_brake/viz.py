import random
from pathlib import Path

import mat73
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from ray import tune
import plotly.express as px

from bci_brake.data import BrakeDset, X_DROP_FEATS
from bci_brake.model import EEGBrakeSeqLSTM


PathLike = str | Path

def _plot_save_show(f: plt.Figure, save_p: PathLike | None = None, show: bool = True):
    f.tight_layout()

    if save_p is not None:
        f.savefig(save_p)

    if show:
        f.show()

    return f


def plot_reaction_time(mat_p: PathLike, save_p: PathLike | None = None, plt_kwargs: dict | None = None, *, show: bool = True):
    if plt_kwargs is None:
        plt_kwargs = {}

    data = mat73.loadmat(mat_p)
    mrk = data["mrk"]

    reaction_time = mrk["event"]["react"]
    time = mrk["time"]

    f, ax = plt.subplots(**plt_kwargs)

    ax.scatter(time, reaction_time)

    return _plot_save_show(f, save_p, show)


def plot_braking_event(model: torch.nn.Module, device, dset: BrakeDset, i: int | None = None, plt_kwargs: dict | None = None, *, show: bool = True):
    alt_x_features = list(dset.alt_x_features)

    dist_to_lead_idx = alt_x_features.index("dist_to_lead")
    lead_brake_idx = alt_x_features.index("lead_brake")
    participant_brake_idx = alt_x_features.index("brake")
    participant_gas_idx = alt_x_features.index("gas")
    emg_idx = alt_x_features.index("EMGf")

    if plt_kwargs is None:
        plt_kwargs = {}

    if i is None:
        i = random.randint(0, len(dset))

    x, target, react_time, alt_x = dset[i]
    # x = np.random.standard_normal((400, 59))

    dist_to_lead = alt_x[:, dist_to_lead_idx] * -1
    lead_brake = alt_x[:, lead_brake_idx]
    participant_brake = alt_x[:, participant_brake_idx]
    emg = alt_x[:, emg_idx]

    x = torch.tensor(x, device=device, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        pred = model(x)
        pred = F.sigmoid(pred)[0].cpu().numpy()

    f, (ax0, ax1, ax2) = plt.subplots(3, **plt_kwargs)

    n = len(target)
    time = np.linspace(0, val_dset.period * n, n)
    stim_idx = np.argmax(target)

    stim_time = time[stim_idx]
    response_time = stim_time + react_time

    ax0.plot(time, pred, label="AI brake probability", color="green")
    ax0.vlines(stim_time, ymin=0, ymax=1, label="lead brakes", color="red")
    ax0.vlines(response_time, ymin=0, ymax=1, label="participant brakes", color='purple')
    ax0.legend(loc="upper left")

    ax1.plot(time, lead_brake, label="lead brake deflection", color="red")
    ax1.plot(time, participant_brake, label="participant brake deflection", color="purple")
    ax1.legend()

    ax2.plot(time, dist_to_lead, label="distance to lead", color="blue")
    ax2.legend()

    f.tight_layout()
    plt.legend()
    plt.show()

def plt_breaking_start(dset: BrakeDset):
    brake_times = []

    for x, target, react_time in dset:
        pass
        # brake_star


    # x, target, react_time = ds

def parallel_coords_tune(tune_path):
    from bci_brake.main import train_from_config

    tuner = tune.Tuner.restore(tune_path, trainable=train_from_config)

    df = tuner.get_results().get_dataframe(
        filter_metric="validation_iou",
        filter_mode="max"
    )

    cols = df.columns
    cols = [c.split("/")[-1] for c in cols]
    df.columns = cols

    df["kernel"] = list(zip(
        df["kernel_0"],
        df["kernel_1"],
        df["kernel_2"]
    ))

    df["dilation"] = list(zip(
        df["dilation_0"],
        df["dilation_1"],
        df["dilation_2"]
    ))

    # labels = ["window_length", "cnn_branch_out", "kernel", "dilation", "lstm_layers", "dropout", "lr", "weight_decay",
    #           "validation_iou"]

    dilations = [f"dilation_{i}" for i in range(3)]
    kernels = [f"kernel_{i}" for i in range(3)]

    xs = ["window_length"]
    xs.extend(dilations)
    xs.extend(kernels)

    y = df["validation_iou"]

    for x in xs:
        x_ = df[x]

        plt.scatter(x_, y)
        plt.title(x)
        plt.tight_layout()
        plt.show()




if __name__ == "__main__":
    import sys
    p = "/home/josh/projects/python/BCIBrake/data/trainings/tune/train_from_config_2025-09-11_21-23-57"
    parallel_coords_tune(p)

    sys.exit()

    mat_p = "/home/josh/projects/python/BCIBrake/data/mats/VPih.mat"
    model_p = "/home/josh/projects/python/BCIBrake/data/trainings/focal_loss_2/models/best.pt"
    device = 0
    # plot_reaction_time(mat_p, plt_kwargs={"figsize": (12, 6)})

    model = EEGBrakeSeqLSTM(59)
    model.load_state_dict(torch.load(model_p))
    model.to(device)
    model.eval()

    train_dset, val_dset = BrakeDset.init_from_p(
        mat_p,
        p_tps=(1., 1.),
        seq_split=(0.8, 0.2),
        window_length=2000,
    )

    plt_kwargs = dict(figsize=(12, 12))

    for _ in range(10):
        plot_braking_event(model, device, val_dset, plt_kwargs=plt_kwargs)





