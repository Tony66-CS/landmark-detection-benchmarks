"""
utils.py — Helper Functions
"""

import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from config import ANATOMICAL_LANDMARKS, LANDMARK_IDS


# ── Heatmap decoding ──────────────────────────────────────────────────────────

def decode_heatmaps(heatmaps: torch.Tensor) -> torch.Tensor:
    """
    Hard argmax decoding: find the (x, y) position of the peak value.
    More reliable than soft-argmax when heatmaps are peaked (small sigma).

    Args:
        heatmaps : (B, N, Hm, Wm)
    Returns:
        coords   : (B, N, 2) — (x, y) in heatmap pixel coordinates
    """
    B, N, Hm, Wm = heatmaps.shape
    flat  = heatmaps.view(B, N, -1)
    idx   = flat.argmax(dim=-1)           # (B, N)
    y     = (idx // Wm).float()           # row
    x     = (idx %  Wm).float()           # col
    return torch.stack([x, y], dim=-1)    # (B, N, 2)


def heatmap_coords_to_input(
    coords_hm: torch.Tensor,
    heatmap_size: int,
    input_size: int,
) -> torch.Tensor:
    """Scale heatmap-space coords → input-image-space coords."""
    scale = input_size / heatmap_size
    return coords_hm * scale


def input_coords_to_original(
    coords_in: torch.Tensor,
    scale_x: float,
    scale_y: float,
) -> torch.Tensor:
    inv = torch.tensor([1.0 / scale_x, 1.0 / scale_y],
                       dtype=coords_in.dtype, device=coords_in.device)
    return coords_in * inv.view(1, 1, 2)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_mre(pred, gt, pixel_spacing_mm=1.0):
    diff   = pred - gt
    errors = np.sqrt((diff ** 2).sum(axis=-1)) * pixel_spacing_mm
    return float(errors.mean()), errors


def compute_sdr(errors, thresholds=(2.0, 2.5, 3.0, 4.0)):
    return {t: float((errors <= t).mean() * 100.0) for t in thresholds}


def aggregate_metrics(all_pred, all_gt, pixel_spacing_mm=1.0,
                      thresholds=(2.0, 2.5, 3.0, 4.0)):
    M, N, _ = all_pred.shape
    all_errors = np.zeros((M, N), dtype=np.float32)
    for i in range(M):
        _, errs = compute_mre(all_pred[i], all_gt[i], pixel_spacing_mm)
        all_errors[i] = errs
    flat = all_errors.flatten()
    return {
        "mre":              float(flat.mean()),
        "std":              float(flat.std()),
        "sdr":              compute_sdr(flat, thresholds),
        "per_landmark_mre": all_errors.mean(axis=0),
    }


# ── Checkpoint utilities ──────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch, metrics, path, is_best=False):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = {
        "epoch":      epoch,
        "state_dict": model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "metrics":    metrics,
    }
    torch.save(state, path)
    if is_best:
        best_path = path.replace(".pth", "_best.pth")
        torch.save(state, best_path)
        print(f"  ★ Best checkpoint saved → {best_path}")


def load_checkpoint(path, model, optimizer=None, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    print(f"  Loaded checkpoint from epoch {checkpoint.get('epoch','?')} [{path}]")
    return checkpoint.get("metrics", {})


# ── Visualisation ─────────────────────────────────────────────────────────────

_COLOURS = {
    "skeletal":    (255, 50,  50),
    "dental":      (50,  200, 50),
    "soft_tissue": (50,  100, 255),
}
_LM_CATEGORY = ["skeletal"] * 16 + ["dental"] * 8 + ["soft_tissue"] * 5


def draw_landmarks(image, landmarks, pred_landmarks=None, radius=5):
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    N = landmarks.shape[0]
    for i in range(N):
        cat    = _LM_CATEGORY[i] if i < len(_LM_CATEGORY) else "skeletal"
        colour = _COLOURS[cat]
        cx, cy = int(round(landmarks[i, 0])), int(round(landmarks[i, 1]))
        cv2.circle(vis, (cx, cy), radius, colour, -1)
        if pred_landmarks is not None:
            px, py = int(round(pred_landmarks[i, 0])), int(round(pred_landmarks[i, 1]))
            cv2.circle(vis, (px, py), radius, (255, 255, 0), 2)
            cv2.line(vis, (cx, cy), (px, py), (200, 200, 200), 1)
    return vis


def plot_metrics_history(history, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if "train_loss" in history:
        axes[0].plot(history["train_loss"], label="Train Loss")
    if "val_loss" in history:
        axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(True)
    if "val_mre" in history:
        axes[1].plot(history["val_mre"], label="Val MRE (px)", color="orange")
    axes[1].set_title("MRE (px)"); axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved metrics plot → {save_path}")
    else:
        plt.show()
    plt.close()


def print_landmark_mre(per_lm_mre):
    print(f"\n{'Landmark':<30} {'Symbol':<8} {'MRE (mm)':>10}")
    print("-" * 52)
    lm_info = list(ANATOMICAL_LANDMARKS.values())
    for i, mre in enumerate(per_lm_mre):
        title  = lm_info[i]["title"]  if i < len(lm_info) else f"LM{i}"
        symbol = lm_info[i]["symbol"] if i < len(lm_info) else "?"
        print(f"  {title:<28} {symbol:<8} {mre:>10.4f}")
    print("-" * 52)
    print(f"  {'MEAN':<36} {per_lm_mre.mean():>10.4f}\n")