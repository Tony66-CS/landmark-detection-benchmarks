# Aariz Cephalometric Landmark Detection — Training Pipeline

A complete, modular training and evaluation pipeline for the **Aariz** benchmark dataset
(1000 lateral cephalometric X-rays, 29 anatomical landmarks, CVM stages).

---

## File Overview

| File | Purpose |
|---|---|
| `config.py` | Landmark IDs, CVM stages, constants |
| `dataset.py` | Dataset loader, augmentation, heatmap generation |
| `model.py` | HRNet / U-Net / ResNet backbone models |
| `utils.py` | Decoding, metrics (MRE, SDR), checkpoints, visualisation |
| `train.py` | End-to-end training script |
| `eval.py` | Evaluation: MRE & SDR on val/test split |
| `predict.py` | Inference on a single image |

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Dataset Setup

Download the Aariz dataset from [Figshare](https://doi.org/10.6084/m9.figshare.27986417.v1)
and extract it. The expected directory structure is:

```
aariz_dataset/
  Train/
    Cephalograms/           *.png
    Annotations/
      Cephalometric Landmarks/
        Senior Orthodontists/   *.json
        Junior Orthodontists/   *.json
      CVM Stages/               *.json
  Valid/   (same layout)
  Test/    (same layout)
```

---

## Training

```bash
# HRNet backbone (recommended), 100 epochs, batch size 4
python train.py \
    --data /path/to/aariz_dataset \
    --backbone hrnet \
    --epochs 100 \
    --batch_size 4 \
    --lr 1e-4 \
    --loss awing \
    --checkpoint_dir checkpoints/hrnet_run1

# U-Net backbone
python train.py --data /path/to/aariz_dataset --backbone unet --epochs 100

# ResNet-50 backbone with ImageNet weights
python train.py --data /path/to/aariz_dataset --backbone resnet --epochs 100

# Resume from a checkpoint
python train.py --data /path/to/aariz_dataset --resume checkpoints/hrnet_run1/epoch_050.pth
```

**Key hyperparameters:**

| Flag | Default | Description |
|---|---|---|
| `--input_size` | 512 | Network input resolution (px) |
| `--heatmap_size` | 128 | Heatmap output resolution (px) |
| `--sigma` | 6.0 | Gaussian sigma for heatmap targets |
| `--loss` | awing | `mse` or `awing` (Adaptive Wing Loss) |
| `--cvm_weight` | 0.1 | Weight of auxiliary CVM classification loss |
| `--lr` | 1e-4 | Initial learning rate (cosine annealed) |
| `--weight_decay` | 1e-4 | AdamW weight decay |

---

## Evaluation

```bash
# Evaluate on validation set
python eval.py \
    --data /path/to/aariz_dataset \
    --checkpoint checkpoints/hrnet_run1/epoch_100_best.pth \
    --mode VALID \
    --output_json results_valid.json

# Evaluate on test set with visualisations
python eval.py \
    --data /path/to/aariz_dataset \
    --checkpoint checkpoints/hrnet_run1/epoch_100_best.pth \
    --mode TEST \
    --vis_dir vis_outputs/ \
    --output_json results_test.json
```

Sample output:
```
============================================================
  Evaluation Results  [TEST]
============================================================
  MRE  : 1.74 ± 3.45 mm

  Success Detection Rate (SDR):
    @ 2.0 mm : 79.41 %
    @ 2.5 mm : 85.93 %
    @ 3.0 mm : 90.12 %
    @ 4.0 mm : 94.55 %
```

---

## Inference on a Single Image

```bash
python predict.py \
    --image /path/to/xray.png \
    --checkpoint checkpoints/hrnet_run1/epoch_100_best.pth \
    --vis \
    --output_json prediction.json
```

---

## Model Architecture Notes

### HRNet (default, recommended)
- Maintains high-resolution representations throughout the network
- 3 parallel resolution branches (W32 variant: 32 / 64 / 128 channels)
- Multi-scale feature fusion after each stage
- Heatmaps predicted at 1/4 input resolution

### U-Net
- Classic encoder-decoder with skip connections
- Lightweight; good baseline for limited GPU memory
- Heatmaps at full encoder input resolution

### ResNet-50
- Pretrained ImageNet encoder + FPN-style decoder
- Heatmaps at 1/4 input resolution

---

## Citation

If you use this pipeline with the Aariz dataset, please cite:

```bibtex
@article{khalid2025benchmark,
  title={A Benchmark Dataset for Automatic Cephalometric Landmark Detection and CVM Stage Classification},
  author={Khalid, Muhammad Anwaar and Zulfiqar, Kanwal and Bashir, Ulfat and
          Shaheen, Areeba and Iqbal, Rida and Rizwan, Zarnab and
          Rizwan, Ghina and Fraz, Muhammad Moazam},
  journal={Scientific Data},
  volume={12}, number={1}, pages={1336}, year={2025},
  publisher={Nature Publishing Group UK London}
}
```
