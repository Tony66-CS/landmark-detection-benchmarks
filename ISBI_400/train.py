"""

Cephalometric Landmark Detection - Training Pipeline
=====================================================
Model: EfficientNet-B3 backbone + Coordinate regression head
Input: Raw cephalometric X-ray image (any size → resized to 512x512)
Output: 35 landmark (x, y) coordinates (normalized 0–1)

Usage:
    python train.py --images_dir ./images --csv all_landmarks.csv
"""

import os
import argparse
import math
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageEnhance
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import json

# ── Config ──────────────────────────────────────────────────────────────────
IMG_SIZE      = 512
BATCH_SIZE    = 8
EPOCHS        = 100
LR            = 3e-4
WEIGHT_DECAY  = 1e-4
TRAIN_SPLIT   = 0.85
SEED          = 42
NUM_WORKERS   = 2
EARLY_STOP    = 15          # patience epochs
CHECKPOINT    = "best_model.pth"


# ── Dataset ──────────────────────────────────────────────────────────────────
class CephDataset(Dataset):
    """
    Returns image tensor + flat landmark vector [x0,y0, x1,y1, …] (normalized).
    Handles images that have fewer landmarks than the full set by masking.
    """
    def __init__(self, df, img_dir, landmark_names, transform=None):
        self.df             = df
        self.img_dir        = Path(img_dir)
        self.landmark_names = landmark_names
        self.n_landmarks    = len(landmark_names)
        self.transform      = transform
        self.image_files    = df["image"].unique().tolist()

    def _find_image(self, fname):
        """Resolve image path regardless of extension (.png / .jpg / .jpeg)."""
        candidate = self.img_dir / fname
        if candidate.exists():
            return candidate
        stem = Path(fname).stem
        for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
            p = self.img_dir / (stem + ext)
            if p.exists():
                return p
        raise FileNotFoundError(
            f"Image '{fname}' not found in {self.img_dir} "
            f"(tried .png / .jpg / .jpeg)"
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        fname  = self.image_files[idx]
        rows   = self.df[self.df["image"] == fname]

        # ── Load & convert image (handles mixed .png / .jpg / .jpeg) ────────
        img_path = self._find_image(fname)
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        # ── Build coordinate tensor ───────────────────────────────────────
        coords = torch.full((self.n_landmarks * 2,), -1.0)   # -1 = missing
        for _, row in rows.iterrows():
            name = row["name"]
            if name in self.landmark_names:
                i = self.landmark_names.index(name)
                coords[2 * i]     = float(row["x_norm"])
                coords[2 * i + 1] = float(row["y_norm"])

        return img, coords, fname


# ── Augmentation ─────────────────────────────────────────────────────────────
def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.0),   # X-rays: no horizontal flip
            transforms.RandomAffine(degrees=5, translate=(0.03, 0.03), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ])


# ── Model ─────────────────────────────────────────────────────────────────────
class CephLandmarkNet(nn.Module):
    """
    EfficientNet-B3 backbone → global avg pool → regression head.
    Output: 2*N coordinates in [0,1].
    """
    def __init__(self, n_landmarks, pretrained=True):
        super().__init__()
        weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b3(weights=weights)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()

        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_landmarks * 2),
            nn.Sigmoid()      # outputs in (0,1)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


# ── Loss ──────────────────────────────────────────────────────────────────────
class MaskedWingLoss(nn.Module):
    """
    Wing loss for robust landmark regression, masked for missing points (-1).
    """
    def __init__(self, w=10.0, eps=2.0):
        super().__init__()
        self.w   = w
        self.eps = eps
        self.C   = w - w * math.log(1 + w / eps)

    def forward(self, pred, target):
        mask   = (target >= 0).float()          # valid landmarks
        diff   = (pred - target).abs() * mask
        loss   = torch.where(
            diff < self.w,
            self.w * torch.log(1 + diff / self.eps),
            diff - self.C
        )
        n_valid = mask.sum().clamp(min=1)
        return loss.sum() / n_valid


# ── MRE metric ───────────────────────────────────────────────────────────────
def mean_radial_error(pred, target, img_size=IMG_SIZE):
    """Mean Radial Error in pixels (on IMG_SIZE × IMG_SIZE grid)."""
    pred_px   = pred.view(-1, 2) * img_size
    target_px = target.view(-1, 2) * img_size
    valid     = (target.view(-1, 2) >= 0).all(dim=1)
    if valid.sum() == 0:
        return 0.0
    dist = ((pred_px[valid] - target_px[valid]) ** 2).sum(dim=1).sqrt()
    return dist.mean().item()


# ── Training loop ─────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_mre, n = 0, 0, 0
    for imgs, coords, _ in loader:
        imgs, coords = imgs.to(device), coords.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss  = criterion(preds, coords)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        bs          = imgs.size(0)
        total_loss += loss.item() * bs
        total_mre  += mean_radial_error(preds.detach(), coords) * bs
        n          += bs
    return total_loss / n, total_mre / n


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, total_mre, n = 0, 0, 0
    for imgs, coords, _ in loader:
        imgs, coords = imgs.to(device), coords.to(device)
        preds       = model(imgs)
        loss        = criterion(preds, coords)
        bs          = imgs.size(0)
        total_loss += loss.item() * bs
        total_mre  += mean_radial_error(preds, coords) * bs
        n          += bs
    return total_loss / n, total_mre / n


# ── Main ──────────────────────────────────────────────────────────────────────
def main(args):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load CSV ──────────────────────────────────────────────────────────
    df = pd.read_csv(args.csv)
    landmark_names = df["name"].unique().tolist()
    n_landmarks    = len(landmark_names)
    print(f"Landmarks: {n_landmarks}  |  Images: {df['image'].nunique()}")

    # Save landmark order for inference
    with open("landmark_names.json", "w") as f:
        json.dump(landmark_names, f)

    # ── Split ─────────────────────────────────────────────────────────────
    all_imgs = df["image"].unique()
    np.random.shuffle(all_imgs)
    n_train  = int(len(all_imgs) * TRAIN_SPLIT)
    train_imgs, val_imgs = all_imgs[:n_train], all_imgs[n_train:]

    train_df = df[df["image"].isin(train_imgs)].reset_index(drop=True)
    val_df   = df[df["image"].isin(val_imgs)].reset_index(drop=True)
    print(f"Train: {len(train_imgs)}  |  Val: {len(val_imgs)}")

    # ── Datasets & loaders ────────────────────────────────────────────────
    train_ds = CephDataset(train_df, args.images_dir, landmark_names, get_transforms(True))
    val_ds   = CephDataset(val_df,   args.images_dir, landmark_names, get_transforms(False))
    train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    val_dl   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # ── Model, loss, optimizer ────────────────────────────────────────────
    model     = CephLandmarkNet(n_landmarks, pretrained=True).to(device)
    criterion = MaskedWingLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # ── Training loop ─────────────────────────────────────────────────────
    history        = {"train_loss": [], "val_loss": [], "train_mre": [], "val_mre": []}
    best_val_mre   = float("inf")
    no_improve     = 0

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_mre = train_one_epoch(model, train_dl, optimizer, criterion, device)
        vl_loss, vl_mre = validate(model, val_dl, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_mre"].append(tr_mre)
        history["val_mre"].append(vl_mre)

        improved = vl_mre < best_val_mre
        if improved:
            best_val_mre = vl_mre
            no_improve   = 0
            torch.save({
                "epoch":          epoch,
                "model_state":    model.state_dict(),
                "landmark_names": landmark_names,
                "img_size":       IMG_SIZE,
            }, CHECKPOINT)
            tag = "✓ saved"
        else:
            no_improve += 1
            tag = f"  (no improve {no_improve}/{EARLY_STOP})"

        print(f"Epoch {epoch:3d} | "
              f"TrLoss {tr_loss:.4f}  TrMRE {tr_mre:.2f}px | "
              f"VlLoss {vl_loss:.4f}  VlMRE {vl_mre:.2f}px {tag}")

        if no_improve >= EARLY_STOP:
            print("Early stopping triggered.")
            break

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"],   label="val")
    axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot(history["train_mre"], label="train")
    axes[1].plot(history["val_mre"],   label="val")
    axes[1].set_title("MRE (px)"); axes[1].legend()
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=120)
    print(f"\nBest Val MRE: {best_val_mre:.2f} px  |  Checkpoint: {CHECKPOINT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True, help="Folder with raw X-ray images")
    parser.add_argument("--csv",        required=True, help="Path to all_landmarks.csv")
    main(parser.parse_args())
