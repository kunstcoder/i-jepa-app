import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.ijepa_backbone import IJEPABackbone


class LinearDecoder(nn.Module):
    """Simple 1x1 conv + bilinear upsample."""

    def __init__(self, embed_dim: int, num_classes: int, img_size: int = 224):
        super().__init__()
        self.img_size = img_size
        self.conv = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, D, H, W] -> [B, num_classes, img_size, img_size]"""
        x = self.conv(x)
        return F.interpolate(
            x, size=(self.img_size, self.img_size),
            mode="bilinear", align_corners=False,
        )


class ProgressiveDecoder(nn.Module):
    """Conv + upsample decoder that progressively restores spatial resolution."""

    def __init__(self, embed_dim: int, num_classes: int, img_size: int = 224):
        super().__init__()
        self.img_size = img_size

        self.layers = nn.Sequential(
            # stage 1
            nn.Conv2d(embed_dim, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            # stage 2
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            # stage 3
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            # stage 4
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, D, H, W] -> [B, num_classes, img_size, img_size]"""
        x = self.layers(x)
        x = self.final(x)
        return F.interpolate(
            x, size=(self.img_size, self.img_size),
            mode="bilinear", align_corners=False,
        )


class IJEPASegmentor(nn.Module):
    """Semantic segmentation model built on top of a frozen I-JEPA backbone.

    Supports two decoder types:
    - 'linear':      1x1 conv + bilinear upsample (fast, simple)
    - 'progressive': multi-stage conv + upsample (better quality)
    """

    def __init__(
        self,
        backbone: IJEPABackbone,
        num_classes: int,
        decoder_type: str = "progressive",
    ):
        super().__init__()
        self.backbone = backbone

        if decoder_type == "linear":
            self.decoder = LinearDecoder(
                backbone.embed_dim, num_classes, backbone.img_size,
            )
        elif decoder_type == "progressive":
            self.decoder = ProgressiveDecoder(
                backbone.embed_dim, num_classes, backbone.img_size,
            )
        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial = self.backbone.get_spatial_features(x)  # [B, D, H, W]
        return self.decoder(spatial)  # [B, num_classes, H_orig, W_orig]
