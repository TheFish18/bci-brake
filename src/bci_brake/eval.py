import random
from pathlib import Path

import mat73
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from bci_brake.data import BrakeDset
from bci_brake.model import EEGBrakeSeqLSTM
from bci_brake.train import load_data_train_all, calc_brake_metrics


def eval_model(model: nn.Module, dset: Dataset, device=0, batch_size=50):
    dloader = DataLoader(dset, batch_size=batch_size, shuffle=False)
    metrics = calc_brake_metrics(model, dloader, device=device)



if __name__ == "__main__":
    mat_p = "/home/josh/projects/python/BCIBrake/data/mats/VPae.mat"
    mat_dir = Path("/home/josh/projects/python/BCIBrake/data/mats")
    model_p = "/home/josh/projects/python/BCIBrake/data/trainings/focal_loss/models/best.pt"
    device = 0
    # plot_reaction_time(mat_p, plt_kwargs={"figsize": (12, 6)})

    model = EEGBrakeSeqLSTM(59)
    model.load_state_dict(torch.load(model_p))
    model.to(device)
    model.eval()

    train_dset, val_dset = load_data_train_all(mat_dir)

    for i in range(5):
        eval_model(model, val_dset, device=device)
        print()


