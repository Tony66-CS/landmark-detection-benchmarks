"""
predict.py — Inference Script: Predict Landmarks on a Single Image

Usage:
    python predict.py --image path/to/xray.png --checkpoint checkpoints/best.pth
    python predict.py --image path/to/xray.png --checkpoint checkpoints/best.pth --vis
"""

import os
import argparse
import json
import numpy as np
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import build_model
from utils import (
    decode_heatmaps,
    heatmap_coords_to_input,
    input_coords_to_original,
    load_checkpoint,
    draw_landmarks,
)
from config import ANATOMICAL_LANDMARKS, LANDMARK_IDS


# ── Single-image inference ────────────────────────────────────────────────────

def predict_single(
    image_path: str,
    model: torch.nn.Module,
    device: torch.device,
    input_size: int = 512,
    heatmap_size: int = 128,
) -> dict:
    """
    Run landmark detection on one image.

    Returns:
        {
            "landmarks": {symbol: {"x": ..., "y": ...}, ...},  # original pixel coords
            "cvm_stage": int,                                    # 1-indexed predicted stage
        }
    """
    # ── Load & preprocess ─────────────────────────────────────────────────
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")

    orig_h, orig_w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    transform = A.Compose([
        A.LongestMaxSize(max_size=input_size),
        A.PadIfNeeded(min_height=input_size, min_width=input_size,
                      border_mode=cv2.BORDER_CONSTANT, value=0),
        A.Normalize(mean=[0.485], std=[0.229]),
        ToTensorV2(),
    ])

    result    = transform(image=img_rgb)
    img_tensor = result["image"].unsqueeze(0).to(device)   # (1, 3, H, W)

    scale_x = input_size / orig_w
    scale_y = input_size / orig_h

    # ── Model forward pass ────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        pred_hm, pred_cvm = model(img_tensor)

    # ── Decode landmarks ──────────────────────────────────────────────────
    pred_hm_size = pred_hm.shape[-1]
    coords_hm    = decode_heatmaps(pred_hm)                                 # (1, N, 2)
    coords_input = heatmap_coords_to_input(coords_hm, pred_hm_size, input_size)
    coords_orig  = input_coords_to_original(coords_input, scale_x, scale_y)  # → orig px

    coords_np = coords_orig.squeeze(0).cpu().numpy()   # (N, 2)

    # ── CVM stage ─────────────────────────────────────────────────────────
    cvm_probs    = torch.softmax(pred_cvm, dim=-1).squeeze(0).cpu().numpy()
    cvm_stage    = int(np.argmax(cvm_probs)) + 1   # 1-indexed

    # ── Format output ─────────────────────────────────────────────────────
    lm_info   = list(ANATOMICAL_LANDMARKS.values())
    landmarks = {}
    for i, lm_id in enumerate(LANDMARK_IDS):
        info = ANATOMICAL_LANDMARKS[lm_id]
        landmarks[info["symbol"]] = {
            "title": info["title"],
            "x":     float(coords_np[i, 0]),
            "y":     float(coords_np[i, 1]),
        }

    return {
        "landmarks": landmarks,
        "cvm_stage": cvm_stage,
        "cvm_probs": cvm_probs.tolist(),
        "original_size": (orig_h, orig_w),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    model = build_model(args.backbone).to(device)
    load_checkpoint(args.checkpoint, model, device=str(device))

    prediction = predict_single(
        args.image, model, device,
        input_size=args.input_size,
        heatmap_size=args.heatmap_size,
    )

    # ── Console output ────────────────────────────────────────────────────
    print(f"\n  Image         : {args.image}")
    print(f"  CVM Stage     : CVM-S{prediction['cvm_stage']}")
    print(f"\n  Landmarks (x, y in original pixel space):")
    for sym, info in prediction["landmarks"].items():
        print(f"    {sym:<8} {info['title']:<30} x={info['x']:.1f}  y={info['y']:.1f}")

    # ── Save JSON ─────────────────────────────────────────────────────────
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(prediction, f, indent=2)
        print(f"\n  Saved predictions → {args.output_json}")

    # ── Visualisation ─────────────────────────────────────────────────────
    if args.vis:
        img_bgr = cv2.imread(args.image)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        coords = np.array(
            [[v["x"], v["y"]] for v in prediction["landmarks"].values()],
            dtype=np.float32,
        )
        vis = draw_landmarks(img_rgb, coords)

        out_vis = args.vis_out or (os.path.splitext(args.image)[0] + "_pred.png")
        cv2.imwrite(out_vis, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print(f"  Visualisation saved → {out_vis}")


def parse_args():
    p = argparse.ArgumentParser(description="Aariz landmark prediction on a single image")
    p.add_argument("--image",        type=str, required=True,  help="Path to input X-ray image.")
    p.add_argument("--checkpoint",   type=str, required=True,  help="Path to trained .pth checkpoint.")
    p.add_argument("--backbone",     type=str, default="hrnet",
                   choices=["hrnet", "unet", "resnet"])
    p.add_argument("--input_size",   type=int, default=512)
    p.add_argument("--heatmap_size", type=int, default=128)
    p.add_argument("--vis",          action="store_true",       help="Save visualised prediction.")
    p.add_argument("--vis_out",      type=str, default=None,    help="Output path for visualisation.")
    p.add_argument("--output_json",  type=str, default=None,    help="Save predictions as JSON.")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
