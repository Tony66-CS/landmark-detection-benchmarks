import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch

SELECTED_LANDMARKS = [
    'S', 'Na', 'Po', 'Or', 'ANS', 'PNS', 'Ar', 'Go', 'Me',
    'Mx-ABAM', 'Mx-ALAM', 'Mx-PBAM', 'Mx-PLAM',
    'Md-ABAM', 'Md-ALAM', 'Md-PBAM', 'Md-PLAM',
    'U1IncisalTip', 'U1RootTip', 'L1IncisalTip', 'L1RootTip'
]

def generate_heatmap(height, width, x, y, sigma=10):
    """Generate a 2D Gaussian heatmap centered at (x, y)."""
    xv = np.arange(width)
    yv = np.arange(height)
    xx, yy = np.meshgrid(xv, yv)
    heatmap = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
    return heatmap.astype(np.float32)

class MyLandmarkDataset(Dataset):
    def __init__(self, root_dir, image_size=(512, 512), augment=False, sigma=10):
        self.image_dir = os.path.join(root_dir, "images")
        self.excel_dir = os.path.join(root_dir, "Excel")
        self.image_size = image_size
        self.augment = augment
        self.sigma = sigma
        self.landmark_names = SELECTED_LANDMARKS.copy()

        self.samples = []
        skipped = 0
        for fname in os.listdir(self.image_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                case_id = os.path.splitext(fname)[0]
                csv_path = os.path.join(self.excel_dir, f"{case_id}.csv")
                img_path = os.path.join(self.image_dir, fname)
                if os.path.exists(csv_path):
                    df_check = pd.read_csv(csv_path)
                    missing = [l for l in SELECTED_LANDMARKS if l not in df_check['Name'].values]
                    if missing:
                        print(f"  Skipping {fname}: missing {missing}")
                        skipped += 1
                    else:
                        self.samples.append((img_path, csv_path))

        print(f"[Dataset] Found {len(self.samples)} valid pairs. Skipped {skipped} incomplete cases.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, csv_path = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        image = image.resize(self.image_size)

        # Augmentation
        if self.augment:
            enhancer = transforms.ColorJitter(brightness=0.3, contrast=0.3)
            image = enhancer(image)

        img_tensor = transforms.ToTensor()(image)  # [3, H, W]

        # Load landmarks
        df = pd.read_csv(csv_path)
        df = df[df['Name'].isin(SELECTED_LANDMARKS)]
        df = df.set_index('Name').loc[SELECTED_LANDMARKS].reset_index()

        x_coords = df['X'].values.astype(np.float32)
        y_coords = df['Y'].values.astype(np.float32)

        # Scale coords to resized image space
        x_scaled = x_coords * (self.image_size[0] / orig_w)
        y_scaled = y_coords * (self.image_size[1] / orig_h)

        # Generate heatmaps — one per landmark
        H, W = self.image_size[1], self.image_size[0]
        num_landmarks = len(SELECTED_LANDMARKS)
        heatmaps = np.zeros((num_landmarks, H, W), dtype=np.float32)
        for i in range(num_landmarks):
            heatmaps[i] = generate_heatmap(H, W, x_scaled[i], y_scaled[i], sigma=self.sigma)

        heatmaps_tensor = torch.from_numpy(heatmaps)  # [N, H, W]

        return img_tensor, heatmaps_tensor, img_path, self.landmark_names