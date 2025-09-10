import random
from pathlib import Path
from typing import Callable

import numpy as np
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.ops import sigmoid_focal_loss

from bci_brake.data import BrakeDset


def load_data_train_all(
        mat_dir: Path,
        train_val_split=(0.8, 0.2),
        p_tps=(0.5, 0.5),
        window_length=2000,
        predict_horizon_length=(100, 500),
) -> tuple[BrakeDset, BrakeDset]:
    """
    Train/Validate split each participant
    Args:
        mat_dir: directory that contains *.mat
        train_val_split: (train_proportion, validate_proportion)
        p_tps: see bci_brake.data.BrakeDset
        window_length: see bci_brake.data.BrakeDset
        predict_horizon_length: see bci_brake.data.BrakeDset

    Returns:
        train dataset, validate dataset
    """
    train_dsets, val_dsets = [], []

    for p in mat_dir.glob("*.mat"):
        train_dset, val_dset = BrakeDset.init_from_p(
            p,
            p_tps=p_tps,
            seq_split=train_val_split,
            window_length=window_length,
            predict_horizon_length=predict_horizon_length
        )
        train_dsets.append(train_dset)
        val_dsets.append(val_dset)

    train_dset = ConcatDataset(train_dsets)
    val_dset = ConcatDataset(val_dsets)

    return train_dset, val_dset


def load_data_holdout_participant(
        mat_dir: Path,
        n_holdouts: int = 1,
        p_tps=(0.5, 0.5),
        window_length=2000,
        predict_horizon_length=500,
) -> tuple[BrakeDset, BrakeDset]:
    """
    Train/Validate split each participant
    Args:
        mat_dir: directory that contains *.mat
        n_holdouts: number of participant to holdout for validation
        p_tps: see bci_brake.data.BrakeDset
        window_length: see bci_brake.data.BrakeDset
        predict_horizon_length: see bci_brake.data.BrakeDset

    Returns:
        train dataset, validate dataset
    """
    paths = list(mat_dir.glob("*.mat"))
    holdouts = random.choices(paths, k=n_holdouts)

    train_dsets, val_dsets = [], []

    for p in paths:
        if p in holdouts:
            p_tp = p_tps[1]
        else:
            p_tp = p_tps[0]

        dset = BrakeDset.init_from_p(
            p,
            p_tp,
            window_length=window_length,
            predict_horizon_length=predict_horizon_length
        )

        if p in holdouts:
            val_dsets.append(dset)
        else:
            train_dsets.append(dset)

    train_dset = ConcatDataset(train_dsets)
    val_dset = ConcatDataset(val_dsets)

    return train_dset, val_dset


def calc_brake_metrics(
        model: nn.Module,
        data_loader: DataLoader,
        writer: SummaryWriter | None = None,
        writer_main_tag: str | None = None,
        global_step: int | None = None,
        *,
        verbose: bool = True,
        device = 0

):
    n_correct = []
    totals = []

    n_correct_brakes = []
    n_pred_brake = []
    n_actual_brake = []

    human_react_times = []
    bci_react_times = []

    model.to(device)
    for x, target, react_time, alt_x in data_loader:
        x = x.type(torch.float32).to(device)
        target = target.type(torch.float32).to(device)

        with torch.no_grad():
            pred = model(x)
            pred = (F.sigmoid(pred) > 0.45).type(torch.uint8)

        actual_brake = torch.any(target, 1).type(torch.uint8)
        predicts_brake = torch.any(pred, 1).type(torch.uint8)

        tps = (actual_brake == predicts_brake).sum()
        n_correct.append(tps.item())
        totals.append(actual_brake.shape[0])

        both_brake = (actual_brake * predicts_brake).type(torch.bool)
        n_correct_brakes.append(both_brake.sum().item())
        n_pred_brake.append(predicts_brake.sum().item())
        n_actual_brake.append(actual_brake.sum().item())

        if both_brake.sum() > 0:
            human_react_time = react_time[both_brake.detach().cpu().numpy()]

            target_start = torch.argmax(target, 1)[both_brake]
            pred_start = torch.argmax(pred, 1)[both_brake]

            offset = pred_start - target_start
            bci_react_time = (offset * 5).type(torch.float32)

            human_react_times.extend(human_react_time.tolist())
            bci_react_times.extend(bci_react_time.tolist())

    acc = sum(n_correct) / sum(totals)
    iou = sum(n_correct_brakes) / (sum(n_pred_brake) + sum(n_actual_brake) - sum(n_correct_brakes) + 1e-12)
    recall = sum(n_correct_brakes) / (sum(n_actual_brake) + 1e-12)
    precision = sum(n_correct_brakes) / (sum(n_pred_brake) + 1e-12)

    human_react_times = np.array(human_react_times)
    avg_human_react = human_react_times.mean()
    std_human_react = human_react_times.std()
    med_human_react = np.median(human_react_times)

    bci_react_times = np.array(bci_react_times)
    avg_bci_react = bci_react_times.mean()
    std_bci_react = bci_react_times.std()
    med_bci_react = np.median(bci_react_times)

    tag_scalar_dict = {
        "accuracy": acc,
        "recall": recall,
        "precision": precision,
        "iou": iou,
        "avg human react": avg_human_react,
        "std human react": std_human_react,
        "median human react": med_human_react,
        "avg bci react": avg_bci_react,
        "std bci react": std_bci_react,
        "median bci react": med_bci_react
    }

    for k, v in tag_scalar_dict.items():
        if writer is not None:
            if writer_main_tag is None or global_step is None:
                raise ValueError("if writer is provided, must pass values of writer_main_tag and global_step")

            writer.add_scalars(
                k,
                {
                    writer_main_tag: v
                },
                global_step=global_step
            )

        if verbose:
            if " " in k:
                vs = str(v)
            else:
                vs = f"{v:0.2%}"

            print(f"{k}: {vs}")

    return tag_scalar_dict


def train_bci_model(
        model: nn.Module,
        optim: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        train_dset: BrakeDset,
        val_dset: BrakeDset,
        log_dir: Path,
        *,
        device=0,
        epochs: int = 50,
        batch_size: int = 25,
        verbose: bool = True
):
    writer = SummaryWriter(log_dir=log_dir)
    model.to(device)
    train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dset, batch_size=batch_size, shuffle=False)

    model_save_dir = log_dir / "models"
    model_save_dir.mkdir(exist_ok=True)

    best_val_iou = 0

    print(f"STARTING TRAINING")
    for epoch in range(1, epochs + 1):

        train_losses = []
        val_losses = []

        model.train()
        for x, target, react_time, alt_x in tqdm(train_loader):
            x = x.to(device).type(torch.float32)
            target = target.to(device).type(torch.float32)
            pred = model(x)

            optim.zero_grad()
            # loss = loss_fn(pred, target)
            loss = sigmoid_focal_loss(pred, target, reduction="mean")
            loss.backward()
            optim.step()

            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            for x, target, react_time, alt_x in tqdm(val_loader):
                x = x.to(device).type(torch.float32)
                target = target.to(device).type(torch.float32)

                x = x.to(device)
                target = target.to(device)
                pred = model(x)

                # loss = loss_fn(pred, target)
                loss = sigmoid_focal_loss(pred, target, reduction="mean")

                val_losses.append(loss.item())

        calc_brake_metrics(
            model=model,
            data_loader=train_loader,
            writer=writer,
            writer_main_tag="train",
            global_step=epoch,
            verbose=verbose,
            device=device
        )
        val_metrics = calc_brake_metrics(
            model=model,
            data_loader=val_loader,
            writer=writer,
            writer_main_tag="validation",
            global_step=epoch,
            verbose=verbose,
            device=device
        )

        val_iou = val_metrics["iou"]
        scheduler.step(val_iou)
        curr_lr = scheduler.get_last_lr()[0]

        train_loss = np.mean(train_losses).item()
        val_loss = np.mean(val_losses).item()

        if verbose:
            print(f"EPOCH: {epoch}: ")
            print(f"\tTrain: ")
            print(f"\t\tloss: {train_loss}")
            print(f"\tValidation: ")
            print(f"\t\tloss: {val_loss}")

        writer.add_scalars(
            "loss",
            {
                'train': train_loss,
                "validation": val_loss
            },
            global_step=epoch
        )

        writer.add_scalar("lr", curr_lr, epoch)

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), model_save_dir.joinpath("best.pt"))
            with open(model_save_dir.joinpath("best.txt"), "w") as f:
                f.write(str(epoch))

    torch.save(model.state_dict(), model_save_dir.joinpath("last.pt"))
