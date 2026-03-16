"""
dataset.py — Aariz Dataset Loader with Preprocessing and Augmentation

Loads lateral cephalometric X-ray images, landmark annotations (averaged from
senior and junior orthodontists), and CVM stage labels. Returns images, Gaussian
heatmaps (for heatmap regression), raw landmark coordinates, and CVM one-hot vectors.
"""

import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ── Config ────────────────────────────────────────────────────────────────────
from config import ANATOMICAL_LANDMARKS, CVM_STAGES, NUM_LANDMARKS, NUM_CVM_STAGES

# Ordered list of landmark IDs (preserves consistent ordering across samples)
LANDMARK_IDS = list(ANATOMICAL_LANDMARKS.keys())


# ── Gaussian heatmap helper ───────────────────────────────────────────────────

def generate_heatmap(size: tuple, center: tuple, sigma: float = 6.0) -> np.ndarray:
    """
    Create a 2-D Gaussian heatmap for a single landmark.

    Args:
        size    : (H, W) of the output heatmap.
        center  : (x, y) landmark position in heatmap coordinates.
        sigma   : Standard deviation of the Gaussian kernel.

    Returns:
        heatmap : np.ndarray of shape (H, W) with values in [0, 1].
    """
    H, W = size
    heatmap = np.zeros((H, W), dtype=np.float32)

    cx, cy = int(round(center[0])), int(round(center[1]))

    # Bounding box to speed up computation
    radius = int(3 * sigma)
    x0, x1 = max(0, cx - radius), min(W, cx + radius + 1)
    y0, y1 = max(0, cy - radius), min(H, cy + radius + 1)

    xs = np.arange(x0, x1) - cx
    ys = np.arange(y0, y1) - cy
    xx, yy = np.meshgrid(xs, ys)
    patch = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))

    heatmap[y0:y1, x0:x1] = patch
    return heatmap


def generate_heatmaps(
    landmarks_px: np.ndarray,
    heatmap_size: tuple,
    sigma: float = 6.0,
) -> np.ndarray:
    """
    Generate stacked heatmaps for all landmarks.

    Args:
        landmarks_px  : (N, 2) array of (x, y) in heatmap pixel space.
        heatmap_size  : (H, W).
        sigma         : Gaussian sigma.

    Returns:
        heatmaps : np.ndarray of shape (N, H, W).
    """
    N = landmarks_px.shape[0]
    H, W = heatmap_size
    heatmaps = np.zeros((N, H, W), dtype=np.float32)
    for i in range(N):
        heatmaps[i] = generate_heatmap((H, W), (landmarks_px[i, 0], landmarks_px[i, 1]), sigma)
    return heatmaps


# ── Augmentation pipelines ────────────────────────────────────────────────────

def get_train_transforms(input_size: int) -> A.Compose:
    """Stronger augmentation pipeline to reduce overfitting on 700 samples."""
    return A.Compose(
        [
            A.LongestMaxSize(max_size=input_size),
            A.PadIfNeeded(min_height=input_size, min_width=input_size,
                          border_mode=cv2.BORDER_CONSTANT),
            A.HorizontalFlip(p=0.0),          # Disabled: laterality matters
            A.Affine(
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                scale=(0.85, 1.15),
                rotate=(-15, 15),
                p=0.8
            ),
            A.RandomBrightnessContrast(brightness_limit=0.3,
                                       contrast_limit=0.3, p=0.7),
            A.GaussNoise(p=0.4),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.CLAHE(clip_limit=3.0, p=0.4),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(
            format="xy",
            remove_invisible=False,   # keep all 29 even if outside bounds
        ),
    )


def get_val_transforms(input_size: int) -> A.Compose:
    """Albumentations pipeline for validation / test (deterministic)."""
    return A.Compose(
        [
            A.LongestMaxSize(max_size=input_size),
            A.PadIfNeeded(min_height=input_size, min_width=input_size,
                          border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


# ── Dataset class ─────────────────────────────────────────────────────────────

class AarizDataset(Dataset):
    """
    PyTorch Dataset for the Aariz cephalometric benchmark.

    Directory layout expected (mirrors the official Figshare release):
        <root>/
          Train/
            Cephalograms/          *.png  (or *.jpg)
            Annotations/
              Cephalometric Landmarks/
                Senior Orthodontists/  *.json
                Junior Orthodontists/  *.json
              CVM Stages/              *.json
          Valid/   (same structure)
          Test/    (same structure)

    Args:
        dataset_folder_path : Root directory of the dataset.
        mode                : "TRAIN", "VALID", or "TEST".
        input_size          : Square resolution fed to the network (e.g. 512).
        heatmap_size        : Spatial size of the heatmap output (e.g. 128).
        sigma               : Gaussian sigma for heatmap generation.
        transforms          : Albumentations Compose object (auto-selected if None).
    """

    def __init__(
        self,
        dataset_folder_path: str,
        mode: str,
        input_size: int = 512,
        heatmap_size: int = 128,
        sigma: float = 2.0,
        transforms=None,
    ):
        # Actual on-disk folders are lowercase: train / valid / test
        mode_map = {"TRAIN": "train", "VALID": "valid", "TEST": "test"}
        if mode.upper() not in mode_map:
            raise ValueError("mode must be TRAIN, VALID, or TEST")
        folder = mode_map[mode.upper()]

        self.images_root = os.path.join(dataset_folder_path, folder, "Cephalograms")
        labels_root = os.path.join(dataset_folder_path, folder, "Annotations")

        self.senior_root = os.path.join(
            labels_root, "Cephalometric Landmarks", "Senior Orthodontists"
        )
        self.junior_root = os.path.join(
            labels_root, "Cephalometric Landmarks", "Junior Orthodontists"
        )
        self.cvm_root = os.path.join(labels_root, "CVM Stages")

        self.images_list = sorted(os.listdir(self.images_root))
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma

        # Load pixel spacing (mm/px) from CSV for physical MRE computation
        csv_path = os.path.join(dataset_folder_path, "cephalogram_machine_mappings.csv")
        self.pixel_spacing = self._load_pixel_spacing(csv_path)

        if transforms is None:
            is_train = mode.upper() == "TRAIN"
            self.transforms = (
                get_train_transforms(input_size)
                if is_train
                else get_val_transforms(input_size)
            )
        else:
            self.transforms = transforms

    # ── internal helpers ───────────────────────────────────────────────────

    def _load_pixel_spacing(self, csv_path: str) -> dict:
        """
        Load per-image pixel spacing (mm/px) from cephalogram_machine_mappings.csv.
        Returns dict mapping image stem → pixel_spacing_mm (float).
        Falls back to 1.0 if CSV not found or column missing.
        """
        spacing = {}
        if not os.path.exists(csv_path):
            return spacing
        import csv
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Try common column name variants
                stem = row.get("cephalogram_id") or row.get("image_id") or row.get("filename", "")
                stem = os.path.splitext(stem)[0]
                px   = row.get("pixel_size") or row.get("pixel_spacing") or row.get("resolution", "1.0")
                try:
                    spacing[stem] = float(px)
                except ValueError:
                    spacing[stem] = 1.0
        return spacing

    def get_pixel_spacing(self, image_stem: str) -> float:
        """Return mm/px for a given image, defaulting to 1.0 if unknown."""
        return self.pixel_spacing.get(image_stem, 1.0)

    def _load_image(self, file_name: str) -> np.ndarray:
        """Read image as single-channel uint8 and convert to 3-channel for Albumentations."""
        path = os.path.join(self.images_root, file_name)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # (H, W, 3)
        return img

    def _load_landmarks_raw(self, json_name: str) -> dict:
        """
        Load raw landmark coordinates from a JSON annotation file.
        Returns a dict mapping landmark_id -> {"x": float, "y": float}.
        """
        def _parse(root: str) -> dict:
            path = os.path.join(root, json_name)
            with open(path, "r") as f:
                data = json.load(f)
            return {lm["landmark_id"]: {"x": lm["value"]["x"], "y": lm["value"]["y"]}
                    for lm in data["landmarks"]}

        senior = _parse(self.senior_root)
        junior = _parse(self.junior_root)

        averaged = {}
        for lm_id in LANDMARK_IDS:
            sx, sy = senior[lm_id]["x"], senior[lm_id]["y"]
            jx, jy = junior[lm_id]["x"], junior[lm_id]["y"]
            averaged[lm_id] = {
                "x": np.ceil(0.5 * (sx + jx)),
                "y": np.ceil(0.5 * (sy + jy)),
            }
        return averaged

    def _landmarks_to_array(self, lm_dict: dict, img_h: int, img_w: int) -> np.ndarray:
        """
        Convert landmark dict to (N, 2) pixel array.
        Aariz stores coordinates as absolute pixel values (not normalised).
        """
        arr = np.zeros((NUM_LANDMARKS, 2), dtype=np.float32)
        for i, lm_id in enumerate(LANDMARK_IDS):
            arr[i, 0] = lm_dict[lm_id]["x"]   # already in pixels
            arr[i, 1] = lm_dict[lm_id]["y"]
        return arr

    def _load_cvm(self, json_name: str) -> np.ndarray:
        """Return one-hot CVM stage vector of length NUM_CVM_STAGES."""
        path = os.path.join(self.cvm_root, json_name)
        with open(path, "r") as f:
            data = json.load(f)
        stage_val = data["cvm_stage"]["value"]          # 1-indexed
        one_hot = np.zeros(NUM_CVM_STAGES, dtype=np.float32)
        one_hot[stage_val - 1] = 1.0
        return one_hot

    # ── public interface ───────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.images_list)

    def __getitem__(self, index: int):
        """
        Returns:
            image       : Tensor (3, input_size, input_size)   – normalised
            heatmaps    : Tensor (NUM_LANDMARKS, H/s, W/s)     – Gaussian targets
            landmarks   : Tensor (NUM_LANDMARKS, 2)            – pixel coords (augmented)
            cvm_stage   : Tensor (NUM_CVM_STAGES,)             – one-hot
            orig_size   : tuple (orig_H, orig_W)               – for metric rescaling
            scale       : tuple (scale_x, scale_y)             – input_size / orig_size
        """
        img_name = self.images_list[index]
        json_name = os.path.splitext(img_name)[0] + ".json"

        # ── load ─────────────────────────────────────────────────────────
        img = self._load_image(img_name)
        orig_h, orig_w = img.shape[:2]

        lm_dict = self._load_landmarks_raw(json_name)
        landmarks_px = self._landmarks_to_array(lm_dict, orig_h, orig_w)  # (N,2) pixels
        cvm_one_hot = self._load_cvm(json_name)

        # ── augment ───────────────────────────────────────────────────────
        # Albumentations expects keypoints as list of (x, y)
        keypoints = [(float(landmarks_px[i, 0]), float(landmarks_px[i, 1]))
                     for i in range(NUM_LANDMARKS)]

        result = self.transforms(image=img, keypoints=keypoints)
        img_tensor = result["image"]           # (3, H, W) float tensor

        aug_kps = result["keypoints"]          # list of (x, y) in augmented space
        aug_landmarks = np.array(
            [[kp[0], kp[1]] for kp in aug_kps], dtype=np.float32
        )
        # Clamp to valid image bounds so out-of-bounds keypoints don't break heatmaps
        aug_landmarks[:, 0] = aug_landmarks[:, 0].clip(0, self.input_size - 1)
        aug_landmarks[:, 1] = aug_landmarks[:, 1].clip(0, self.input_size - 1)

        # ── heatmaps ──────────────────────────────────────────────────────
        # Scale augmented landmarks to heatmap resolution
        scale_h = self.heatmap_size / self.input_size
        scale_w = self.heatmap_size / self.input_size
        lm_heatmap = aug_landmarks.copy()
        lm_heatmap[:, 0] *= scale_w
        lm_heatmap[:, 1] *= scale_h

        heatmaps = generate_heatmaps(
            lm_heatmap, (self.heatmap_size, self.heatmap_size), self.sigma
        )

        # ── scaling metadata (for metric evaluation) ──────────────────────
        scale_x = self.input_size / orig_w
        scale_y = self.input_size / orig_h

        return (
            img_tensor,                                     # (3, H, W)
            torch.from_numpy(heatmaps),                     # (N, Hm, Wm)
            torch.from_numpy(aug_landmarks),                # (N, 2)  px in input space
            torch.from_numpy(cvm_one_hot),                  # (NUM_CVM_STAGES,)
            (orig_h, orig_w),
            (scale_x, scale_y),
        )