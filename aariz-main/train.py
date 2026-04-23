import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from my_dataset import MyLandmarkDataset
import numpy as np

# ---- CONFIG ----
DATASET_DIR   = "C:/Users/sanda/OneDrive/Desktop/Archive"
NUM_LANDMARKS = 21
IMAGE_SIZE    = (512, 512)
BATCH_SIZE    = 2        # heatmaps use more VRAM, keep at 2 for 4GB GPU
EPOCHS        = 150
LEARNING_RATE = 1e-4
PATIENCE      = 15
MODEL_SAVE_PATH = "trained_model.pth"
# ----------------

# ---- U-Net Model ----
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=21):
        super().__init__()
        # Encoder
        self.enc1 = DoubleConv(in_channels, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.enc4 = DoubleConv(128, 256)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(256, 512)

        # Decoder
        self.up4    = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4   = DoubleConv(512, 256)
        self.up3    = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3   = DoubleConv(256, 128)
        self.up2    = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2   = DoubleConv(128, 64)
        self.up1    = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1   = DoubleConv(64, 32)

        # Output
        self.out_conv = nn.Conv2d(32, out_channels, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)  # [B, NUM_LANDMARKS, H, W]

# ---- Setup ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

full_dataset = MyLandmarkDataset(DATASET_DIR, image_size=IMAGE_SIZE, augment=False)
train_size = int(0.8 * len(full_dataset))
val_size   = len(full_dataset) - train_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
train_ds.dataset.augment = True

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model     = UNet(in_channels=3, out_channels=NUM_LANDMARKS).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
criterion = nn.MSELoss()

# ---- Training Loop ----
best_val_loss    = float('inf')
patience_counter = 0

for epoch in range(EPOCHS):
    # Training
    model.train()
    train_loss = 0
    for images, heatmaps, _, names in train_loader:
        images, heatmaps = images.to(device), heatmaps.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, heatmaps)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, heatmaps, _, names in val_loader:
            images, heatmaps = images.to(device), heatmaps.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, heatmaps).item()

    train_loss /= len(train_loader)
    val_loss   /= len(val_loader)
    scheduler.step()

    print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    if val_loss < best_val_loss:
        best_val_loss    = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"  ✅ Model saved (val_loss: {val_loss:.6f})")
    else:
        patience_counter += 1
        print(f"  ⏳ No improvement ({patience_counter}/{PATIENCE})")
        if patience_counter >= PATIENCE:
            print(f"\n🛑 Early stopping at epoch {epoch+1}. Best: {best_val_loss:.6f}")
            break

print("\nTraining complete! Best model saved to:", MODEL_SAVE_PATH)