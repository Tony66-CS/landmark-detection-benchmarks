"""
model.py — Heatmap-Based Cephalometric Landmark Detection Model

Backbones:
  • "hrnet"   — Real HRNet-W32 via timm (29M params, ImageNet pretrained) ← recommended
  • "unet"    — Lightweight encoder-decoder U-Net
  • "resnet"  — ResNet-50 encoder + FPN decoder (ImageNet pretrained)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from config import NUM_LANDMARKS, NUM_CVM_STAGES


# ── Shared building blocks ────────────────────────────────────────────────────

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1, groups=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            ConvBNReLU(in_ch, out_ch),
        )
    def forward(self, x):
        return self.up(x)


# ── HRNet-W32 via timm (recommended) ─────────────────────────────────────────

class HRNetW32Model(nn.Module):
    """
    Real HRNet-W32 backbone from timm with ImageNet pretrained weights.
    Output: heatmaps (B, NUM_LANDMARKS, H/4, W/4) + cvm_logits (B, NUM_CVM_STAGES)
    """

    def __init__(self, num_landmarks: int, num_cvm: int, pretrained: bool = True):
        super().__init__()
        import timm

        # Load pretrained HRNet-W32 — remove the classification head
        self.backbone = timm.create_model(
            "hrnet_w32",
            pretrained=pretrained,
            features_only=True,
        )

        # timm HRNet-W32 features_only returns 5 scales with these channels:
        # [64, 128, 256, 512, 1024]  — actual total when upsampled = 1984
        # We discover this at runtime so we use a lazy init approach:
        # Just use a 1×1 conv to project whatever channels we get → 256
        self._agg_built = False
        self.agg = None

        # Landmark heatmap head (built after first forward to know channels)
        self.heatmap_head = nn.Sequential(
            ConvBNReLU(256, 256),
            nn.Conv2d(256, num_landmarks, kernel_size=1),
        )

        # CVM classification head
        self.cvm_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_cvm),
        )
        self._num_landmarks = num_landmarks
        self._num_cvm = num_cvm

    def forward(self, x):
        features = self.backbone(x)

        # Upsample all feature maps to the finest resolution
        h0, w0 = features[0].shape[-2:]
        upsampled = [features[0]] + [
            F.interpolate(f, size=(h0, w0), mode="bilinear", align_corners=True)
            for f in features[1:]
        ]
        cat = torch.cat(upsampled, dim=1)   # (B, total_ch, H/4, W/4)

        # Build aggregation conv on first forward (lazy — adapts to any channel count)
        if self.agg is None:
            total_ch = cat.shape[1]
            self.agg = ConvBNReLU(total_ch, 256).to(cat.device)

        fused      = self.agg(cat)
        heatmaps   = self.heatmap_head(fused)
        cvm_logits = self.cvm_head(fused)
        return heatmaps, cvm_logits


# ── U-Net backbone ────────────────────────────────────────────────────────────

class UNetEncoder(nn.Module):
    def __init__(self, in_ch=3):
        super().__init__()
        self.enc1 = nn.Sequential(ConvBNReLU(in_ch, 64),  ConvBNReLU(64, 64))
        self.enc2 = nn.Sequential(ConvBNReLU(64, 128),    ConvBNReLU(128, 128))
        self.enc3 = nn.Sequential(ConvBNReLU(128, 256),   ConvBNReLU(256, 256))
        self.enc4 = nn.Sequential(ConvBNReLU(256, 512),   ConvBNReLU(512, 512))
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = nn.Sequential(ConvBNReLU(512, 1024), ConvBNReLU(1024, 512))

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        return b, [e4, e3, e2, e1]

class UNetDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up4  = UpBlock(512, 512);  self.conv4 = ConvBNReLU(512+512, 256)
        self.up3  = UpBlock(256, 256);  self.conv3 = ConvBNReLU(256+256, 128)
        self.up2  = UpBlock(128, 128);  self.conv2 = ConvBNReLU(128+128, 64)
        self.up1  = UpBlock(64,  64);   self.conv1 = ConvBNReLU(64+64,   64)

    def forward(self, b, skips):
        e4, e3, e2, e1 = skips
        x = self.conv4(torch.cat([self.up4(b),  e4], dim=1))
        x = self.conv3(torch.cat([self.up3(x),  e3], dim=1))
        x = self.conv2(torch.cat([self.up2(x),  e2], dim=1))
        x = self.conv1(torch.cat([self.up1(x),  e1], dim=1))
        return x

class UNetModel(nn.Module):
    def __init__(self, num_landmarks, num_cvm):
        super().__init__()
        self.encoder = UNetEncoder()
        self.decoder = UNetDecoder()
        self.heatmap_head = nn.Conv2d(64, num_landmarks, kernel_size=1)
        self.cvm_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(512, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(128, num_cvm),
        )

    def forward(self, x):
        b, skips = self.encoder(x)
        feat = self.decoder(b, skips)
        return self.heatmap_head(feat), self.cvm_head(b)


# ── ResNet-50 backbone ────────────────────────────────────────────────────────

class ResNetModel(nn.Module):
    def __init__(self, num_landmarks, num_cvm, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.resnet50(weights=weights)
        self.stem   = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1   # 256ch @ H/4
        self.layer2 = base.layer2   # 512ch @ H/8
        self.layer3 = base.layer3   # 1024ch @ H/16
        self.layer4 = base.layer4   # 2048ch @ H/32

        self.dec4 = UpBlock(2048, 256)
        self.dec3 = UpBlock(256+1024, 128)
        self.dec2 = UpBlock(128+512, 64)
        self.dec1 = UpBlock(64+256, 64)

        self.heatmap_head = nn.Conv2d(64, num_landmarks, kernel_size=1)
        self.cvm_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(2048, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(256, num_cvm),
        )

    def forward(self, x):
        s  = self.stem(x)
        l1 = self.layer1(s)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        x  = self.dec4(l4)
        x  = self.dec3(torch.cat([x, l3], dim=1))
        x  = self.dec2(torch.cat([x, l2], dim=1))
        x  = self.dec1(torch.cat([x, l1], dim=1))
        return self.heatmap_head(x), self.cvm_head(l4)


# ── Factory ───────────────────────────────────────────────────────────────────

def build_model(backbone: str = "hrnet", pretrained: bool = True) -> nn.Module:
    """
    Args:
        backbone  : "hrnet" | "unet" | "resnet"
        pretrained: Use ImageNet weights (HRNet and ResNet).
    """
    b = backbone.lower()
    if b == "hrnet":
        return HRNetW32Model(NUM_LANDMARKS, NUM_CVM_STAGES, pretrained=pretrained)
    elif b == "unet":
        return UNetModel(NUM_LANDMARKS, NUM_CVM_STAGES)
    elif b == "resnet":
        return ResNetModel(NUM_LANDMARKS, NUM_CVM_STAGES, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown backbone '{backbone}'. Choose hrnet, unet, or resnet.")