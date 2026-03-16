"""
eval.py — Evaluation Script: MRE and SDR on Validation or Test Set

Usage:
    python eval.py --data Aariz --checkpoint checkpoints/epoch_100_best.pth --mode TEST
"""

import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import AarizDataset
from model   import build_model
from utils   import (
    decode_heatmaps,
    heatmap_coords_to_input,
    load_checkpoint,
    aggregate_metrics,
    print_landmark_mre,
    draw_landmarks,
)
import cv2


@torch.no_grad()
def evaluate(model, loader, device, input_size, vis_dir=None,
             thresholds=(2.0, 2.5, 3.0, 4.0)):
    """
    Both predictions and GT are kept in **input-image pixel space** (after
    resize+pad to input_size). This ensures a fair apples-to-apples comparison.
    pixel_spacing is applied per-sample when converting px → mm.
    """
    model.eval()
    all_pred, all_gt, all_spacing = [], [], []
    sample_idx = 0

    for batch in loader:
        images, hm_targets, lm_targets, cvm_targets, orig_sizes, scales = batch
        images = images.to(device)

        pred_hm, _ = model(images)

        # Decode heatmap → input image pixel space
        pred_hm_size = pred_hm.shape[-1]
        coords_hm    = decode_heatmaps(pred_hm)                               # (B, N, 2)
        coords_input = heatmap_coords_to_input(coords_hm, pred_hm_size, input_size)  # (B, N, 2)

        B = images.shape[0]
        for b in range(B):
            pred_np = coords_input[b].cpu().numpy()    # (N, 2) input px
            gt_np   = lm_targets[b].numpy()            # (N, 2) input px  ← same space

            all_pred.append(pred_np)
            all_gt.append(gt_np)

            # pixel spacing: how many mm per pixel in the *input* image
            # scale_x = input_size / orig_w  →  mm_per_input_px = mm_per_orig_px / scale_x
            # We default orig mm/px = 1.0 (dataset CSV not required for basic eval)
            scale_x = float(scales[0][b])
            all_spacing.append(1.0 / scale_x)   # approx mm/px in input space

            if vis_dir is not None:
                _save_vis(images[b].cpu(), pred_np, gt_np,
                          input_size, sample_idx, vis_dir)
            sample_idx += 1

    all_pred    = np.stack(all_pred,    axis=0)   # (M, N, 2)
    all_gt      = np.stack(all_gt,      axis=0)
    avg_spacing = float(np.mean(all_spacing))

    results = aggregate_metrics(all_pred, all_gt,
                                pixel_spacing_mm=avg_spacing,
                                thresholds=thresholds)
    return results


def _save_vis(img_tensor, pred, gt, input_size, idx, vis_dir):
    os.makedirs(vis_dir, exist_ok=True)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_np = img_tensor.permute(1, 2, 0).numpy()
    img_np = (img_np * std + mean).clip(0, 1)
    img_u8 = (img_np * 255).astype(np.uint8)
    vis = draw_landmarks(img_u8, gt, pred)
    cv2.imwrite(os.path.join(vis_dir, f"sample_{idx:04d}.png"),
                cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


def print_results(results, mode):
    print(f"\n{'='*60}")
    print(f"  Evaluation Results  [{mode}]")
    print(f"{'='*60}")
    print(f"  MRE  : {results['mre']:.4f} ± {results['std']:.4f} mm")
    print(f"\n  Success Detection Rate (SDR):")
    for t, sdr in results["sdr"].items():
        print(f"    @ {t} mm : {sdr:.2f} %")
    print_landmark_mre(results["per_landmark_mre"])


def main(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    ds = AarizDataset(args.data, args.mode,
                      input_size=args.input_size,
                      heatmap_size=args.heatmap_size,
                      sigma=args.sigma)
    loader = DataLoader(ds, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"))

    print(f"  Evaluating {len(ds)} samples from {args.mode} split …\n")

    model = build_model(args.backbone).to(device)
    load_checkpoint(args.checkpoint, model, device=str(device))

    results = evaluate(model, loader, device,
                       input_size=args.input_size,
                       vis_dir=args.vis_dir,
                       thresholds=(2.0, 2.5, 3.0, 4.0))

    print_results(results, args.mode)

    if args.output_json:
        out = {
            "mode":    args.mode,
            "mre":     results["mre"],
            "std":     results["std"],
            "sdr":     {str(k): v for k, v in results["sdr"].items()},
            "per_landmark_mre": results["per_landmark_mre"].tolist(),
        }
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  Results saved → {args.output_json}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",         type=str, required=True)
    p.add_argument("--checkpoint",   type=str, required=True)
    p.add_argument("--backbone",     type=str, default="hrnet",
                   choices=["hrnet","unet","resnet"])
    p.add_argument("--mode",         type=str, default="VALID",
                   choices=["VALID","TEST"])
    p.add_argument("--input_size",   type=int,   default=512)
    p.add_argument("--heatmap_size", type=int,   default=128)
    p.add_argument("--sigma",        type=float, default=6.0)
    p.add_argument("--batch_size",   type=int,   default=4)
    p.add_argument("--num_workers",  type=int,   default=0)
    p.add_argument("--vis_dir",      type=str,   default=None)
    p.add_argument("--output_json",  type=str,   default=None)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())