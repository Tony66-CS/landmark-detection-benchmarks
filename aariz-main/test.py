import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os

# ---- CONFIG ----
NUM_LANDMARKS   = 21
IMAGE_SIZE      = (512, 512)
MODEL_SAVE_PATH = "trained_model.pth"
# ----------------

SELECTED_LANDMARKS = [
    'S', 'Na', 'Po', 'Or', 'ANS', 'PNS', 'Ar', 'Go', 'Me',
    'Mx-ABAM', 'Mx-ALAM', 'Mx-PBAM', 'Mx-PLAM',
    'Md-ABAM', 'Md-ALAM', 'Md-PBAM', 'Md-PLAM',
    'U1IncisalTip', 'U1RootTip', 'L1IncisalTip', 'L1RootTip'
]

# ---- U-Net (must match train.py exactly) ----
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
        self.enc1 = DoubleConv(in_channels, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.enc4 = DoubleConv(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(256, 512)
        self.up4  = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = DoubleConv(512, 256)
        self.up3  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        self.up2  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        self.up1  = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(64, 32)
        self.out_conv = nn.Conv2d(32, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)

def load_model(path, device):
    model = UNet(in_channels=3, out_channels=NUM_LANDMARKS).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model

def heatmaps_to_coords(heatmaps, orig_w, orig_h):
    """Find peak of each heatmap and convert to original image coordinates."""
    N, H, W = heatmaps.shape
    points = []
    for i in range(N):
        hm = heatmaps[i]
        idx = np.unravel_index(np.argmax(hm), hm.shape)
        y_px, x_px = idx
        # Scale back to original image size
        x_orig = x_px * (orig_w / W)
        y_orig = y_px * (orig_h / H)
        points.append((x_orig, y_orig))
    return points

def load_ground_truth(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df['Name'].isin(SELECTED_LANDMARKS)]
    df = df.set_index('Name').loc[SELECTED_LANDMARKS].reset_index()
    return df['X'].values.astype(np.float32), df['Y'].values.astype(np.float32)

def estimate_pixel_to_mm(points, gt_x=None, gt_y=None):
    S_N_REAL_MM = 71.0
    s_idx  = SELECTED_LANDMARKS.index('S')
    na_idx = SELECTED_LANDMARKS.index('Na')
    if gt_x is not None:
        sx, sy   = gt_x[s_idx],  gt_y[s_idx]
        nax, nay = gt_x[na_idx], gt_y[na_idx]
    else:
        sx,  sy  = points[s_idx]
        nax, nay = points[na_idx]
    dist_px = np.sqrt((nax - sx)**2 + (nay - sy)**2)
    if dist_px < 1:
        return 0.1
    px_to_mm = S_N_REAL_MM / dist_px
    print(f"  Calibration: S-N = {dist_px:.1f}px → {px_to_mm:.4f} mm/px")
    return px_to_mm

def compute_metrics(points, gt_x, gt_y, px_to_mm):
    pred_x = np.array([p[0] for p in points])
    pred_y = np.array([p[1] for p in points])
    errors_px = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
    errors_mm = errors_px * px_to_mm
    mre  = np.mean(errors_mm)
    sd   = np.std(errors_mm)
    sdr2  = np.mean(errors_mm <= 2.0) * 100
    sdr25 = np.mean(errors_mm <= 2.5) * 100
    sdr3  = np.mean(errors_mm <= 3.0) * 100
    sdr4  = np.mean(errors_mm <= 4.0) * 100
    return mre, sd, sdr2, sdr25, sdr3, sdr4, errors_mm

def predict_and_show(image_path, model, device, gt_csv=None):
    original = Image.open(image_path).convert("RGB")
    orig_w, orig_h = original.size

    resized    = original.resize(IMAGE_SIZE)
    img_tensor = transforms.ToTensor()(resized).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)  # [1, N, H, W]

    heatmaps = output[0].cpu().numpy()  # [N, H, W]
    points   = heatmaps_to_coords(heatmaps, orig_w, orig_h)

    # Ground truth
    has_gt = gt_csv is not None and os.path.exists(gt_csv)
    gt_x, gt_y = None, None
    if has_gt:
        gt_x, gt_y = load_ground_truth(gt_csv)

    # Draw
    draw_img = original.copy()
    draw     = ImageDraw.Draw(draw_img)

    for i, (x, y) in enumerate(points):
        r     = 6
        label = SELECTED_LANDMARKS[i]
        draw.ellipse([x-r, y-r, x+r, y+r], fill='red', outline='yellow')
        draw.text((x+8, y-8), label, fill='white')
        if has_gt:
            gx, gy = gt_x[i], gt_y[i]
            draw.ellipse([gx-r, gy-r, gx+r, gy+r], fill='green', outline='white')
            draw.line([x, y, gx, gy], fill='yellow', width=1)

    # Metrics
    if has_gt:
        px_to_mm = estimate_pixel_to_mm(points, gt_x, gt_y)
        mre, sd, sdr2, sdr25, sdr3, sdr4, errors_mm = compute_metrics(
            points, gt_x, gt_y, px_to_mm
        )
        print("\n" + "="*55)
        print(f"  ACCURACY METRICS — {os.path.basename(image_path)}")
        print("="*55)
        print(f"  MRE ± SD     : {mre:.2f} ± {sd:.2f} mm")
        print(f"  SDR @ 2.0mm  : {sdr2:.2f}%")
        print(f"  SDR @ 2.5mm  : {sdr25:.2f}%")
        print(f"  SDR @ 3.0mm  : {sdr3:.2f}%")
        print(f"  SDR @ 4.0mm  : {sdr4:.2f}%")
        print("="*55)
        print(f"\n  Per-landmark errors:")
        print(f"  {'#':<4} {'Name':<20} {'Error (mm)':>12}")
        print(f"  {'-'*38}")
        for i, err in enumerate(errors_mm):
            flag = " ✗" if err > 4.0 else (" ~" if err > 2.0 else " ✓")
            print(f"  {i+1:<4} {SELECTED_LANDMARKS[i]:<20} {err:>10.2f}{flag}")

        draw.rectangle([10, 10, 330, 170], fill=(0, 0, 0))
        draw.text((15, 15),  f"MRE: {mre:.2f} +/- {sd:.2f} mm",  fill='white')
        draw.text((15, 35),  f"SDR 2.0mm : {sdr2:.1f}%",          fill='lime')
        draw.text((15, 55),  f"SDR 2.5mm : {sdr25:.1f}%",         fill='lime')
        draw.text((15, 75),  f"SDR 3.0mm : {sdr3:.1f}%",          fill='lime')
        draw.text((15, 95),  f"SDR 4.0mm : {sdr4:.1f}%",          fill='lime')
        draw.text((15, 120), f"RED = Predicted",                   fill='red')
        draw.text((15, 140), f"GREEN = Ground Truth",              fill='lime')

    plt.figure(figsize=(14, 14))
    plt.imshow(draw_img)
    title = f"Detected {NUM_LANDMARKS} Landmarks — {os.path.basename(image_path)}"
    if has_gt:
        title += f"  |  MRE: {mre:.2f} ± {sd:.2f} mm"
    plt.title(title, fontsize=13)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if len(sys.argv) < 2:
        print("Usage:")
        print("  No evaluation : python test.py image.jpg")
        print("  With metrics  : python test.py image.jpg ground_truth.csv --eval")
        sys.exit(1)

    image_path = sys.argv[1]
    csv_path   = sys.argv[2] if len(sys.argv) >= 3 else None
    eval_mode  = "--eval" in sys.argv
    gt_csv     = csv_path if eval_mode else None

    model = load_model(MODEL_SAVE_PATH, device)
    predict_and_show(image_path, model, device, gt_csv)