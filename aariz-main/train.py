"""
train.py — End-to-End Training Script for Aariz Landmark Detection
"""

import os
import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import AarizDataset
from model   import build_model
from utils   import (
    decode_heatmaps,
    heatmap_coords_to_input,
    save_checkpoint,
    load_checkpoint,
    aggregate_metrics,
    plot_metrics_history,
)


# ── Loss functions ────────────────────────────────────────────────────────────

class MSEHeatmapLoss(nn.Module):
    def forward(self, pred, target):
        return nn.functional.mse_loss(pred, target)


class AdaptiveWingLoss(nn.Module):
    """
    Vectorised Adaptive Wing Loss — MPS compatible (no boolean indexing).
    Reference: Wang et al. ICCV 2019.
    """
    def __init__(self, omega=14.0, theta=0.5, epsilon=1.0, alpha=2.1):
        super().__init__()
        self.omega   = omega
        self.theta   = theta
        self.epsilon = epsilon
        self.alpha   = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        delta = (target - pred).abs()

        A = self.omega \
            * (1.0 / (1.0 + (self.theta / self.epsilon) ** (self.alpha - target))) \
            * (self.alpha - target) \
            * ((self.theta / self.epsilon) ** (self.alpha - target - 1.0)) \
            / self.epsilon
        C = self.theta * A - self.omega * torch.log(
            1.0 + (self.theta / self.epsilon) ** (self.alpha - target)
        )

        loss_small = self.omega * torch.log(
            1.0 + (delta / self.epsilon) ** (self.alpha - target)
        )
        loss_large = A * delta - C

        # Smooth mask instead of boolean indexing (MPS-safe)
        mask = (delta < self.theta).float()
        return (mask * loss_small + (1.0 - mask) * loss_large).mean()


# ── One training epoch ────────────────────────────────────────────────────────

def train_one_epoch(
    model, loader, optimizer, hm_criterion, cvm_criterion,
    device, heatmap_size, input_size,
    cvm_weight=0.1, epoch=0, log_interval=20,
) -> float:
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(loader):
        images, hm_targets, lm_targets, cvm_targets, _, _ = batch
        images      = images.to(device)
        hm_targets  = hm_targets.to(device)
        cvm_targets = cvm_targets.to(device)

        optimizer.zero_grad()

        pred_hm, pred_cvm = model(images)

        if pred_hm.shape[-2:] != hm_targets.shape[-2:]:
            pred_hm = nn.functional.interpolate(
                pred_hm, size=hm_targets.shape[-2:],
                mode="bilinear", align_corners=True
            )

        loss_hm  = hm_criterion(pred_hm, hm_targets)
        loss_cvm = cvm_criterion(pred_cvm, cvm_targets)
        loss     = loss_hm + cvm_weight * loss_cvm

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()

        if (step + 1) % log_interval == 0:
            print(f"  [Ep {epoch} | step {step+1}/{len(loader)}] "
                  f"loss={loss.item():.4f}  hm={loss_hm.item():.4f}  "
                  f"cvm={loss_cvm.item():.4f}")

    return total_loss / len(loader)


# ── Validation loop ───────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    model, loader, hm_criterion, cvm_criterion,
    device, heatmap_size, input_size, cvm_weight=0.1,
) -> tuple:
    model.eval()
    total_loss = 0.0
    all_pred, all_gt = [], []

    for batch in loader:
        images, hm_targets, lm_targets, cvm_targets, orig_sizes, scales = batch
        images      = images.to(device)
        hm_targets  = hm_targets.to(device)
        cvm_targets = cvm_targets.to(device)

        pred_hm, pred_cvm = model(images)

        if pred_hm.shape[-2:] != hm_targets.shape[-2:]:
            pred_hm = nn.functional.interpolate(
                pred_hm, size=hm_targets.shape[-2:],
                mode="bilinear", align_corners=True
            )

        loss_hm  = hm_criterion(pred_hm, hm_targets)
        loss_cvm = cvm_criterion(pred_cvm, cvm_targets)
        total_loss += (loss_hm + cvm_weight * loss_cvm).item()

        # Decode heatmap coords → input image pixel space
        pred_hm_size = pred_hm.shape[-1]
        coords_hm    = decode_heatmaps(pred_hm)
        coords_input = heatmap_coords_to_input(coords_hm, pred_hm_size, input_size)

        all_pred.append(coords_input.cpu().numpy())
        all_gt.append(lm_targets.numpy())

    all_pred = np.concatenate(all_pred, axis=0)
    all_gt   = np.concatenate(all_gt,   axis=0)

    metrics = aggregate_metrics(all_pred, all_gt)
    return total_loss / len(loader), metrics["mre"]


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"\n{'='*60}")
    print(f"  Aariz Training Pipeline")
    print(f"  Backbone : {args.backbone}")
    print(f"  Device   : {device}")
    print(f"{'='*60}\n")

    train_ds = AarizDataset(args.data, "TRAIN",
                            input_size=args.input_size,
                            heatmap_size=args.heatmap_size,
                            sigma=args.sigma)
    val_ds   = AarizDataset(args.data, "VALID",
                            input_size=args.input_size,
                            heatmap_size=args.heatmap_size,
                            sigma=args.sigma)

    pin = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers,
                              pin_memory=pin, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=pin)

    print(f"  Train samples : {len(train_ds)}")
    print(f"  Val   samples : {len(val_ds)}\n")

    model = build_model(args.backbone, pretrained=not args.no_pretrain).to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model params  : {total_params:.2f} M\n")

    hm_criterion  = MSEHeatmapLoss() if args.loss == "mse" else AdaptiveWingLoss()
    cvm_criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    start_epoch = 1
    best_mre    = float("inf")
    history     = {"train_loss": [], "val_loss": [], "val_mre": []}

    if args.resume:
        load_checkpoint(args.resume, model, optimizer, device=str(device))

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer,
            hm_criterion, cvm_criterion, device,
            args.heatmap_size, args.input_size,
            cvm_weight=args.cvm_weight,
            epoch=epoch,
            log_interval=args.log_interval,
        )

        val_loss, val_mre = validate(
            model, val_loader,
            hm_criterion, cvm_criterion, device,
            args.heatmap_size, args.input_size,
            cvm_weight=args.cvm_weight,
        )

        scheduler.step()

        elapsed = time.time() - t0
        print(f"\n[Epoch {epoch:3d}/{args.epochs}] "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_mre={val_mre:.4f} px  ({elapsed:.1f}s)\n")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mre"].append(val_mre)

        is_best = val_mre < best_mre
        if is_best:
            best_mre = val_mre

        ckpt_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch:03d}.pth")
        save_checkpoint(model, optimizer, epoch,
                        {"val_mre": val_mre, "val_loss": val_loss},
                        ckpt_path, is_best=is_best)

    with open(os.path.join(args.checkpoint_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    plot_metrics_history(history,
                         save_path=os.path.join(args.checkpoint_dir, "metrics.png"))
    print(f"\n  Training complete.  Best val MRE = {best_mre:.4f} px\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",           type=str,   required=True)
    p.add_argument("--backbone",       type=str,   default="hrnet",
                   choices=["hrnet","unet","resnet"])
    p.add_argument("--input_size",     type=int,   default=512)
    p.add_argument("--heatmap_size",   type=int,   default=128)
    p.add_argument("--sigma",          type=float, default=6.0)
    p.add_argument("--epochs",         type=int,   default=100)
    p.add_argument("--batch_size",     type=int,   default=4)
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--weight_decay",   type=float, default=1e-4)
    p.add_argument("--loss",           type=str,   default="awing",
                   choices=["mse","awing"])
    p.add_argument("--cvm_weight",     type=float, default=0.1)
    p.add_argument("--num_workers",    type=int,   default=4)
    p.add_argument("--log_interval",   type=int,   default=20)
    p.add_argument("--checkpoint_dir", type=str,   default="checkpoints/")
    p.add_argument("--resume",         type=str,   default=None)
    p.add_argument("--no_pretrain",    action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())