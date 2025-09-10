import random
from pathlib import Path

import mat73
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, PrecisionRecallDisplay, RocCurveDisplay
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from bci_brake.data import BrakeDset
from bci_brake.model import EEGBrakeSeqLSTM
from bci_brake.train import load_data_train_all, calc_brake_metrics


def eval_model(model: nn.Module, dset: Dataset, device=0, batch_size=50):
    dloader = DataLoader(dset, batch_size=batch_size, shuffle=False)
    metrics = calc_brake_metrics(model, dloader, device=device)

def scores(model: nn.Module, dset: Dataset, device=0, batch_size=50):
    dloader = DataLoader(dset, batch_size=batch_size, shuffle=False)

    targets = []
    preds = []

    for x, target, react_time, alt_x in dloader:
        x = x.type(torch.float32).to(device)

        targets.append(target.cpu().numpy())

        with torch.no_grad():
            pred = F.sigmoid(model(x))

        preds.append(pred.cpu().numpy())

    targets = np.vstack(targets)
    preds = np.vstack(preds)

    targets = np.any(targets, axis=1).astype(np.uint8)
    preds = np.max(preds, axis=1)

    auc = roc_auc_score(targets, preds)
    # roc_c = roc_curve(targets, preds)
    pr_curve = precision_recall_curve(targets, preds)
    #
    # plt.plot(roc_c)
    # plt.show()
    #
    PrecisionRecallDisplay.from_predictions(targets, preds)
    plt.tight_layout()
    plt.show()

    RocCurveDisplay.from_predictions(targets, preds)
    plt.tight_layout()
    plt.show()
    pass






if __name__ == "__main__":
    mat_p = "/home/josh/projects/python/BCIBrake/data/mats/VPae.mat"
    mat_dir = Path("/home/josh/projects/python/BCIBrake/data/mats")
    model_p = "/home/josh/projects/python/BCIBrake/data/trainings/focal_loss_2/models/best.pt"
    device = 0
    # plot_reaction_time(mat_p, plt_kwargs={"figsize": (12, 6)})

    model = EEGBrakeSeqLSTM(59)
    model.load_state_dict(torch.load(model_p))
    model.to(device)
    model.eval()

    # train_dset, val_dset = load_data_train_all(mat_dir)
    train_dset, val_dset = BrakeDset.init_from_p(mat_p, p_tps=(0.5, 0.5), seq_split=(0.8, 0.2))
    scores(model, val_dset, device=device)

    # for i in range(5):
    #     eval_model(model, val_dset, device=device)
    #     print()


