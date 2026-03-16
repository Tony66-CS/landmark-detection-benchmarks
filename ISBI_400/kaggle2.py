# # ── Run this first to inspect your dataset ──
# import os
# import pandas as pd
#
# path = "/Users/tonyshome/Downloads/SBME_CV_CephalometricLandmarks-main"
#
# # Show all files
# for root, dirs, files in os.walk(path):
#     dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git']]
#     level = root.replace(path, '').count(os.sep)
#     indent = '  ' * level
#     print(f"{indent}{os.path.basename(root)}/")
#     for f in files[:5]:  # show first 5 files per folder
#         print(f"  {indent}{f}")
#
# # Read CSV files
# csv_dir = os.path.join(path, "data_csv")
# for f in os.listdir(csv_dir):
#     if f.endswith(".csv"):
#         print(f"\n── {f} ──")
#         df = pd.read_csv(os.path.join(csv_dir, f))
#         print(df.shape)
#         print(df.head(3))
#         print(df.columns.tolist())
#         break  # just show first CSV
#
# model = load_trained_model("best_cephalometric_model.pth")
# image, landmarks = predict(model, "/Users/tonyshome/Downloads/HRNet-Image-Classification-master/1.jpg")
# visualize(image, landmarks)


import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import timm
import os

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CHECKPOINT  = "best_cephalometric_model.pth"
INPUT_SIZE  = 512
NUM_LANDMARKS = 19

device = torch.device("mps"  if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

LANDMARK_NAMES = [
    "Sella", "Nasion", "Orbitale", "Porion", "Subspinale (A)",
    "Supramentale (B)", "Pogonion", "Menton", "Gnathion", "Gonion",
    "Lower Incisor Tip", "Upper Incisor Tip", "Upper Lip", "Lower Lip",
    "Subnasale", "Soft Tissue Pogonion", "Posterior Nasal Spine",
    "Anterior Nasal Spine", "Articulare"
]

# ─────────────────────────────────────────────
# MODEL — must match training architecture exactly
# ─────────────────────────────────────────────
def build_model():
    model = timm.create_model("hrnet_w32.ms_in1k", pretrained=False)
    model.classifier = nn.Sequential(
        nn.Linear(2048, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, NUM_LANDMARKS * 2),
        nn.Sigmoid()
    )
    return model

def load_model(checkpoint_path=CHECKPOINT):
    model = build_model().to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"✅ Model loaded — epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.6f}")
    return model

# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ─────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────
def predict(model, image_path):
    image        = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    tensor       = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(tensor).cpu().numpy().squeeze()  # (38,)

    landmarks = preds.reshape(-1, 2)   # (19, 2) normalized [0,1]
    landmarks[:, 0] *= orig_w          # scale x to pixel space
    landmarks[:, 1] *= orig_h          # scale y to pixel space
    return image, landmarks

# ─────────────────────────────────────────────
# VISUALIZE
# ─────────────────────────────────────────────
def visualize(image, landmarks, save_path="output.png"):
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.imshow(np.array(image))
    cmap = plt.cm.get_cmap("tab20", NUM_LANDMARKS)

    for i, (x, y) in enumerate(landmarks):
        color = cmap(i)
        ax.scatter(x, y, color=color, s=80, zorder=5,
                   edgecolors="white", linewidths=0.8)
        ax.annotate(
            str(i + 1), (x, y),
            xytext=(6, 6), textcoords="offset points",
            fontsize=8, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.2", fc=color, alpha=0.85, ec="none")
        )

    legend = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=cmap(i), markersize=8,
                   label=f"{i+1}. {LANDMARK_NAMES[i]}")
        for i in range(NUM_LANDMARKS)
    ]
    ax.legend(handles=legend, loc="upper right",
              fontsize=7, framealpha=0.85, ncol=2)
    ax.axis("off")
    ax.set_title("Cephalometric Landmark Detection", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ Saved to {save_path}")

# ─────────────────────────────────────────────
# PRINT RESULTS
# ─────────────────────────────────────────────
def print_results(landmarks):
    print("\n📍 Detected Landmarks:")
    print("─" * 48)
    for i, (x, y) in enumerate(landmarks):
        print(f"  {i+1:2}. {LANDMARK_NAMES[i]:<30} ({x:.1f}, {y:.1f})")
    print("─" * 48)

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # ── Change this to any X-ray image path ──
    IMAGE_PATH = "/Users/tonyshome/Downloads/HRNet-Image-Classification-master/15.jpg"

    model = load_model(CHECKPOINT)
    image, landmarks = predict(model, IMAGE_PATH)
    print_results(landmarks)
    visualize(image, landmarks, save_path="output.png")