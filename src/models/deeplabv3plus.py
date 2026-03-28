"""
DeepLabV3+ with MobileNetV2 backbone.

Architecture overview
---------------------
* **Backbone** : MobileNetV2 (ImageNet pre-trained, optional).
  - Low-level features  @ stride 4  — 24 channels  → decoder low-level path
  - High-level features @ stride 16 — 96 channels  → ASPP

* **ASPP** (Atrous Spatial Pyramid Pooling):
  Parallel atrous convolutions at rates [1, 6, 12, 18] plus global average
  pooling, projected to 256 channels.  Captures multi-scale context without
  additional downsampling.

* **Decoder** :
  Low-level features (24 ch) are projected to 48 ch, then concatenated with
  the up-sampled ASPP output (256 ch) and refined with two 3×3 convolutions
  before the final 1×1 classification head.

* **Multi-channel input** :
  When ``in_channels != 3`` (e.g. 6 for pre+post), the first MobileNetV2
  convolution is re-initialised.  If ``in_channels`` is a multiple of 3,
  pre-trained weights are tiled across the extra channels and rescaled so
  that the initial activation magnitude is preserved.

References
----------
* Chen et al., "Encoder-Decoder with Atrous Separable Convolution for
  Semantic Image Segmentation", ECCV 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


# ─── ASPP components ──────────────────────────────────────────────────────────

class _ASPPConv(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, dilation: int):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class _ASPPPooling(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(self.pool(x), size=x.shape[-2:], mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling (rates: 1, 6, 12, 18)."""

    def __init__(self, in_ch: int, out_ch: int = 256):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ),
            _ASPPConv(in_ch, out_ch, dilation=6),
            _ASPPConv(in_ch, out_ch, dilation=12),
            _ASPPConv(in_ch, out_ch, dilation=18),
            _ASPPPooling(in_ch, out_ch),
        ])
        self.project = nn.Sequential(
            nn.Conv2d(len(self.branches) * out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([b(x) for b in self.branches], dim=1))


# ─── Decoder ──────────────────────────────────────────────────────────────────

class Decoder(nn.Module):
    """
    DeepLabV3+ decoder: fuses ASPP output (stride-16) with low-level
    features (stride-4) via 1×1 projection + concatenation + 3×3 refinement.
    """

    def __init__(self, low_level_ch: int, aspp_ch: int = 256, num_classes: int = 5):
        super().__init__()
        self.low_proj = nn.Sequential(
            nn.Conv2d(low_level_ch, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(aspp_ch + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(256, num_classes, 1)

    def forward(self, aspp_out: torch.Tensor, low_feat: torch.Tensor) -> torch.Tensor:
        low  = self.low_proj(low_feat)
        up   = F.interpolate(aspp_out, size=low.shape[-2:], mode="bilinear", align_corners=False)
        x    = self.refine(torch.cat([up, low], dim=1))
        return self.head(x)


# ─── Full model ───────────────────────────────────────────────────────────────

class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ with MobileNetV2 backbone.

    Parameters
    ----------
    num_classes : int
        Number of output segmentation classes.
    in_channels : int
        Input channels.  6 = pre+post concatenated (recommended), 3 = post only.
    pretrained_backbone : bool
        Load ImageNet weights for MobileNetV2.
    """

    def __init__(
        self,
        num_classes: int = 5,
        in_channels: int = 6,
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        weights  = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        backbone = mobilenet_v2(weights=weights).features

        # ── Adapt first convolution for non-standard channel count ────────────
        if in_channels != 3:
            orig = backbone[0][0]   # Conv2d(3, 32, ...)
            new  = nn.Conv2d(
                in_channels, orig.out_channels,
                kernel_size=orig.kernel_size,
                stride=orig.stride,
                padding=orig.padding,
                bias=False,
            )
            if pretrained_backbone:
                # Tile pretrained weights; rescale to preserve activation magnitude
                w       = orig.weight.data                  # (32, 3, k, k)
                repeats = in_channels // 3
                parts   = [w] * repeats
                if in_channels % 3:
                    parts.append(w[:, : in_channels % 3, :, :])
                new.weight.data = torch.cat(parts, dim=1) / (in_channels / 3)
            backbone[0][0] = new

        # ── Backbone segments ─────────────────────────────────────────────────
        # features[0:4]  → stride-4,  24 ch  (low-level for decoder)
        # features[4:14] → stride-16, 96 ch  (encoder output for ASPP)
        self.low_encoder  = backbone[:4]
        self.high_encoder = backbone[4:14]

        self.aspp    = ASPP(in_ch=96, out_ch=256)
        self.decoder = Decoder(low_level_ch=24, aspp_ch=256, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape[-2:]
        low_feat    = self.low_encoder(x)          # stride-4,  24ch
        high_feat   = self.high_encoder(low_feat)  # stride-16, 96ch
        aspp_out    = self.aspp(high_feat)
        out         = self.decoder(aspp_out, low_feat)
        return F.interpolate(out, size=input_shape, mode="bilinear", align_corners=False)


# ─── Factory ──────────────────────────────────────────────────────────────────

def build_model(cfg: dict) -> nn.Module:
    """
    Build a :class:`DeepLabV3Plus` model from a config dict.

    Parameters
    ----------
    cfg : dict
        Must contain a ``"model"`` key with sub-keys
        ``num_classes``, ``in_channels``, ``pretrained_backbone``.
    """
    m = cfg["model"]
    return DeepLabV3Plus(
        num_classes         = m["num_classes"],
        in_channels         = m["in_channels"],
        pretrained_backbone = m["pretrained_backbone"],
    )
