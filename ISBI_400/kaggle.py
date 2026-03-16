import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
import matplotlib.pyplot as plt
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_PATH   = "/Users/tonyshome/Downloads/SBME_CV_CephalometricLandmarks-main"
IMG_DIR     = os.path.join(BASE_PATH, "cepha400")
TRAIN_CSV   = os.path.join(BASE_PATH, "data_csv", "train_senior.csv")
TEST_CSV    = os.path.join(BASE_PATH, "data_csv", "test1_senior.csv")
SAVE_PATH   = "best_cephalometric_model.pth"

INPUT_SIZE    = 512
NUM_LANDMARKS = 19
BATCH_SIZE    = 4
EPOCHS        = 50
LR            = 1e-4

device = torch.device("mps"  if torch.backends.mps.is_available()  else
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
# DATASET
# ─────────────────────────────────────────────
class CephalometricDataset(Dataset):
    def __init__(self, csv_path, img_dir, input_size=INPUT_SIZE, augment=False):
        self.df      = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.augment = augment

        # Build coordinate column names: 1_x, 1_y, 2_x, 2_y, ...
        self.coord_cols = []
        for i in range(1, NUM_LANDMARKS + 1):
            self.coord_cols += [f"{i}_x", f"{i}_y"]

        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_path"])
        image    = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        # Extract landmarks and normalize to [0, 1]
        coords = row[self.coord_cols].values.astype(np.float32)  # (38,)
        coords[0::2] /= orig_w   # x coords
        coords[1::2] /= orig_h   # y coords

        image_tensor = self.transform(image)
        target       = torch.tensor(coords, dtype=torch.float32)

        return image_tensor, target, row["image_path"]

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
def build_model():
    model = timm.create_model("hrnet_w32.ms_in1k", pretrained=True)

    # Find the actual classifier
    for name, module in model.named_children():
        print(f"  {name} → {type(module).__name__}")

    # The real head in this HRNet version is model.classifier
    in_features = model.num_features  # 2048
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, NUM_LANDMARKS * 2),
        nn.Sigmoid()
    )

    # Verify
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE)
        out = model(dummy)
        print(f"Output shape: {out.shape}")  # must be [1, 38]

    return model
# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────
def train():
    train_ds = CephalometricDataset(TRAIN_CSV, IMG_DIR, augment=True)
    val_ds   = CephalometricDataset(TEST_CSV,  IMG_DIR, augment=False)

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model     = build_model().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──
        model.train()
        running = 0.0
        for images, targets, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]", leave=False):
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
            running += loss.item()
        train_loss = running / len(train_loader)

        # ── Validate ──
        model.eval()
        running = 0.0
        with torch.no_grad():
            for images, targets, _ in val_loader:
                images, targets = images.to(device), targets.to(device)
                running += criterion(model(images), targets).item()
        val_loss = running / len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step()

        print(f"Epoch {epoch:3}/{EPOCHS} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
            }, SAVE_PATH)
            print(f"  ✅ Best model saved (val_loss={val_loss:.6f})")

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses,   label="Val")
    plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
    plt.title("Training Loss"); plt.legend()
    plt.tight_layout()
    plt.savefig("training_loss.png")
    plt.show()
    print(f"\nDone! Best val loss: {best_val_loss:.6f}")
    return model

# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────
infer_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_trained_model(checkpoint_path=SAVE_PATH):
    model = build_model().to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"✅ Loaded from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.6f})")
    return model

def predict(model, image_path):
    image  = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    tensor = infer_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(tensor).cpu().numpy().squeeze()  # (38,)

    landmarks = preds.reshape(-1, 2)   # (19, 2) in [0,1]
    landmarks[:, 0] *= orig_w
    landmarks[:, 1] *= orig_h
    return image, landmarks

def visualize(image, landmarks, save_path="output.png"):
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.imshow(np.array(image))
    cmap = plt.cm.get_cmap("tab20", NUM_LANDMARKS)

    for i, (x, y) in enumerate(landmarks):
        color = cmap(i)
        ax.scatter(x, y, color=color, s=60, zorder=5,
                   edgecolors="white", linewidths=0.5)
        ax.annotate(str(i + 1), (x, y), xytext=(5, 5),
                    textcoords="offset points", fontsize=7,
                    fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.2",
                              fc=color, alpha=0.85, ec="none"))

    legend = [plt.Line2D([0], [0], marker='o', color='w',
                         markerfacecolor=cmap(i), markersize=7,
                         label=f"{i+1}. {LANDMARK_NAMES[i]}")
              for i in range(NUM_LANDMARKS)]
    ax.legend(handles=legend, loc="upper right",
              fontsize=6, framealpha=0.8, ncol=2)
    ax.axis("off")
    ax.set_title("Cephalometric Landmark Detection", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ Saved to {save_path}")

def evaluate(model, csv_path=TEST_CSV, img_dir=IMG_DIR):
    """Compute Mean Radial Error in pixels."""
    df   = pd.read_csv(csv_path)
    errors = []

    model.eval()
    for _, row in df.iterrows():
        img_path = os.path.join(img_dir, row["image_path"])
        image, preds = predict(model, img_path)
        orig_w, orig_h = image.size

        gt = []
        for i in range(1, NUM_LANDMARKS + 1):
            gt.append([row[f"{i}_x"], row[f"{i}_y"]])
        gt = np.array(gt, dtype=np.float32)

        radial_errors = np.sqrt(((preds - gt) ** 2).sum(axis=1))
        errors.append(radial_errors)

    errors = np.array(errors)  # (N_images, 19)
    mre    = errors.mean()
    print(f"\n📊 Evaluation on {len(df)} images:")
    print(f"   Mean Radial Error (MRE): {mre:.2f} px")
    print(f"   SDR @ 2mm (≈8px):  {(errors < 8).mean()*100:.1f}%")
    print(f"   SDR @ 4mm (≈16px): {(errors < 16).mean()*100:.1f}%")
    for i, name in enumerate(LANDMARK_NAMES):
        print(f"   {i+1:2}. {name:<30} MRE={errors[:, i].mean():.1f}px")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # ── Step 1: Train ──
    model = train()

    # ── Step 2: Evaluate on test set ──
    model = load_trained_model()
    evaluate(model)

    # ── Step 3: Predict on a single image ──
    image, landmarks = predict(model, os.path.join(IMG_DIR, "289.jpg"))
    visualize(image, landmarks, save_path="289_output.png")



