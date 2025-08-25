from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW

from bci_brake.model import EEGBrakeSeqLSTM
from bci_brake.train import (
    load_data_train_all, load_data_holdout_participant,
    train_bci_model, calc_brake_metrics
)


learning_rate = 1e-3
batch_size = 25
epochs = 50
device = 0

window_length = 2000
predict_horizon_length = 500

training_name = "first"

eeg_features = 59
data_dir = Path(__file__).parent.parent.parent.joinpath("data")
mat_dir = data_dir / "mats"
log_dir = data_dir / f"trainings/{training_name}"


model = EEGBrakeSeqLSTM(eeg_features)
optim = AdamW(model.parameters(), lr=learning_rate)
loss_fn = nn.BCEWithLogitsLoss()
train_dset, val_dset = load_data_train_all(
    mat_dir=mat_dir,
    train_val_split=(0.8, 0.2),
    p_tps=(0.5, 0.5),
    window_length=window_length,
    predict_horizon_length=predict_horizon_length,
)

train_bci_model(
    model=model,
    optim=optim,
    loss_fn=loss_fn,
    train_dset=train_dset,
    val_dset=val_dset,
    log_dir=log_dir,
    device=device,
    epochs=epochs,
    batch_size=batch_size,
    metrics_freq=5,
    verbose=True
)
