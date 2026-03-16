"""
Cephalometric Landmark Detection - Inference
=============================================
Predicts landmark coordinates on a single raw cephalometric X-ray image.

Usage:
    python predict.py --image patient.png --checkpoint best_model.pth
    python predict.py --image patient.png --checkpoint best_model.pth --visualize
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


IMG_SIZE = 512


# ── Model (same arch as train.py) ─────────────────────────────────────────────
class CephLandmarkNet(nn.Module):
    def __init__(self, n_landmarks):
        super().__init__()
        backbone = models.efficientnet_b3(weights=None)
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
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.head(self.backbone(x))


def load_model(checkpoint_path, device):
    ckpt           = torch.load(checkpoint_path, map_location=device)
    landmark_names = ckpt["landmark_names"]
    n_landmarks    = len(landmark_names)
    model          = CephLandmarkNet(n_landmarks).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, landmark_names


def preprocess(image_path):
    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225]),
    ])
    # Resolve path regardless of extension
    image_path = Path(image_path)
    if not image_path.exists():
        stem = image_path.stem
        parent = image_path.parent
        for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
            candidate = parent / (stem + ext)
            if candidate.exists():
                image_path = candidate
                break
        else:
            raise FileNotFoundError(f"Image not found: {image_path}")
    img = Image.open(image_path).convert("RGB")
    return img, tfm(img).unsqueeze(0)


@torch.no_grad()
def predict(model, tensor, device):
    tensor = tensor.to(device)
    coords = model(tensor).squeeze(0).cpu().numpy()   # shape: (2*N,)
    return coords


def visualize(original_img, coords_norm, landmark_names, out_path="prediction.png"):
    """Draw predicted landmarks on the original image at its native resolution."""
    W, H    = original_img.size
    img_arr = np.array(original_img.convert("RGB"))

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img_arr, cmap="gray")

    colors = plt.cm.tab20.colors
    patches = []
    for i, name in enumerate(landmark_names):
        x_norm = coords_norm[2 * i]
        y_norm = coords_norm[2 * i + 1]
        x_px   = x_norm * W
        y_px   = y_norm * H
        c      = colors[i % len(colors)]
        ax.scatter(x_px, y_px, s=40, color=c, zorder=5, edgecolors="white", linewidths=0.5)
        ax.annotate(name, (x_px, y_px), fontsize=5, color="white",
                    xytext=(3, 3), textcoords="offset points")
        patches.append(mpatches.Patch(color=c, label=name))

    ax.legend(handles=patches, loc="upper right", fontsize=5,
              ncol=3, framealpha=0.7)
    ax.set_title("Predicted Cephalometric Landmarks", fontsize=11)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Visualization saved → {out_path}")
    plt.show()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    model, landmark_names = load_model(args.checkpoint, device)
    original_img, tensor  = preprocess(args.image)
    coords_norm           = predict(model, tensor, original_img.size)

    W, H = original_img.size
    print(f"\n{'Landmark':<12} {'x_norm':>8} {'y_norm':>8} {'x_px':>7} {'y_px':>7}")
    print("-" * 48)
    results = {}
    for i, name in enumerate(landmark_names):
        xn, yn = float(coords_norm[2*i]), float(coords_norm[2*i+1])
        xp, yp = xn * W, yn * H
        print(f"{name:<12} {xn:8.4f} {yn:8.4f} {xp:7.1f} {yp:7.1f}")
        results[name] = {"x_norm": xn, "y_norm": yn, "x_px": xp, "y_px": yp}

    if args.visualize:
        visualize(original_img, coords_norm, landmark_names,
                  out_path=args.image.replace(".png","_predicted.png")
                                     .replace(".jpg","_predicted.jpg"))

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",      required=True, help="Path to input X-ray image")
    parser.add_argument("--checkpoint", default="best_model.pth")
    parser.add_argument("--visualize",  action="store_true", help="Save overlay image")
    main(parser.parse_args())
