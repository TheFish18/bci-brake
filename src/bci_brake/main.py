import uuid
import json
from pathlib import Path

import torch
from ray.tune.schedulers import ASHAScheduler
from torch import nn
from torch.nn.functional import dropout
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ray import tune

from bci_brake.model import EEGBrakeSeqLSTM
from bci_brake.train import (
    load_data_train_all, load_data_holdout_participant,
    train_bci_model, calc_brake_metrics
)

DATA_DIR = Path(__file__).parent.parent.parent.joinpath("data")
TUNE_PATH = DATA_DIR / "trainings/tune"

N_EEG_FEATS = 59
DEVICE = 0


def standard_train():
    learning_rate = 1e-3
    batch_size = 25
    epochs = 200

    window_length = 2000
    predict_horizon_length = (100, 500)

    training_name = "focal_loss_2"

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"BCIBrake/data not found")

    mat_dir = DATA_DIR / "mats"
    log_dir = DATA_DIR / f"trainings/{training_name}"

    model = EEGBrakeSeqLSTM(N_EEG_FEATS)
    # model = EEGBrakeLinv1(400, eeg_features)

    optim = AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()
    scheduler = ReduceLROnPlateau(
        optim,
        mode="max",
        factor=0.1,
        patience=20,
        threshold=0.05,
        cooldown=2
    )

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
        scheduler=scheduler,
        loss_fn=loss_fn,
        train_dset=train_dset,
        val_dset=val_dset,
        log_dir=log_dir,
        device=DEVICE,
        epochs=epochs,
        batch_size=batch_size,
        verbose=True
    )

def train_from_config(config: dict, *, tuning=True):
    trial_name = "first"
    mat_dir = DATA_DIR / "mats"

    # data
    epochs = 100
    batch_size = config["batch_size"]
    window_length = config["window_length"]
    predict_horizon_length = (100, 500)

    train_dset, val_dset = load_data_train_all(
        mat_dir=mat_dir,
        train_val_split=(0.8, 0.2),
        p_tps=(0.5, 0.5),
        window_length=window_length,
        predict_horizon_length=predict_horizon_length,
    )

    # ------- model
    kernels = [config[f"kernel_{i}"] for i in range(3)]
    dilations = [config[f"dilation_{i}"] for i in range(3)]

    model = EEGBrakeSeqLSTM(
        N_EEG_FEATS,
        cnn_branch_out=config["cnn_branch_out"],
        kernels=kernels,
        dilations=dilations,
        lstm_hidden=config["lstm_hidden"],
        lstm_layers=config["lstm_layers"],
        dropout=config["dropout"]
    )

    # ------- Optimizer ---------
    learning_rate = config["lr"]
    weight_decay = config["weight_decay"]

    optim = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    loss_fn = nn.BCEWithLogitsLoss()
    scheduler = ReduceLROnPlateau(
        optim,
        mode="max",
        factor=0.1,
        patience=20,
        threshold=0.05,
        cooldown=2
    )


    if tuning:
        log_dir = None
        verbose = False
    else:
        log_dir = DATA_DIR / "trainings" / str(uuid.uuid4())[-5:]
        verbose = True

    train_bci_model(
        model=model,
        optim=optim,
        scheduler=scheduler,
        loss_fn=loss_fn,
        train_dset=train_dset,
        val_dset=val_dset,
        log_dir=log_dir,
        device=DEVICE,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        tuning=tuning
    )



def tune_train():
    sample_space = {
        # data
        "batch_size": tune.randint(4, 60),
        "window_length": tune.qloguniform(1000, 4000, q=500),

        # model
        "cnn_branch_out": tune.qrandint(16, 64, 8),
        "kernel_0": tune.randint(3, 15),
        "kernel_1": tune.randint(15, 50),
        "kernel_2": tune.randint(50, 100),
        "dilation_0": tune.choice([1, 2, 4]),
        "dilation_1": tune.choice([1, 2, 4]),
        "dilation_2": tune.choice([1, 2, 4]),
        "lstm_hidden": tune.qrandint(64, 256, 8),
        "lstm_layers": tune.randint(1, 4),
        "dropout": tune.quniform(0, 0.5, 0.05),

        # optim
        "lr": tune.loguniform(1e-5, 1e-2),
        "weight_decay": tune.loguniform(1e-1, 1e-5),
    }

    asha_scheduler = ASHAScheduler(
        metric="validation_iou",
        mode="max",
        max_t=100,
        grace_period=10,
    )

    trainable = tune.with_resources(
        train_from_config,
        {"cpu": 4, "gpu": 1}
    )

    tuner = tune.Tuner(
        trainable,
        param_space=sample_space,
        tune_config=tune.TuneConfig(
            scheduler=asha_scheduler,
            num_samples=20,
        ),
        run_config=tune.RunConfig(
            storage_path=TUNE_PATH
        )
    )

    results = tuner.fit()


if __name__ == "__main__":
    # Run tune
    # tune_train()

    # Load Tune
    # tuner_p = "/home/josh/projects/python/BCIBrake/data/trainings/tune/train_from_config_2025-09-11_21-23-57"
    # restored_tuner = tune.Tuner.restore(tuner_p, trainable=train_from_config)
    #
    # result_grid = restored_tuner.get_results()

    config_p = Path("/home/josh/projects/python/BCIBrake/data/trainings/tune/train_from_config_2025-09-11_21-23-57/train_from_config_283e9_00017_17_batch_size=47,cnn_branch_out=16,dilation_0=1,dilation_1=4,dilation_2=4,dropout=0.2000,kernel_0=5,_2025-09-11_21-23-58/params.json")
    config = json.load(config_p.open())
    train_from_config(config, tuning=False)



"""
Questions:
- median human react time for train/val do not seem equal. Rough avg for train is 480 ms and 520 for val.
- avg human react time for train/val does not seem equal. Rough avg for train is 480 ms and 500 for val
- x vals all over the place
"""
