import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.ijepa_backbone import IJEPABackbone


class SketchDecoder(nn.Module):
    """Progressive decoder for 1-channel sketch reconstruction."""

    def __init__(self, embed_dim: int, img_size: int = 224):
        super().__init__()
        self.img_size = img_size
        self.layers = nn.Sequential(
            nn.Conv2d(embed_dim, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        x = self.final(x)
        x = F.interpolate(
            x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False,
        )
        return torch.sigmoid(x)


class IJEPASketchReconstructor(nn.Module):
    """I-JEPA-based masked sketch reconstruction model.

    Inputs:
      image: [B, 3, H, W] (normalized)
      mask:  [B, 1, H, W] where 1 indicates region to reconstruct

    Output:
      sketch: [B, 1, H, W] in [0, 1]
    """

    def __init__(self, backbone: IJEPABackbone, mask_fill: float = 0.0):
        super().__init__()
        self.backbone = backbone
        self.mask_fill = mask_fill
        self.decoder = SketchDecoder(embed_dim=backbone.embed_dim, img_size=backbone.img_size)

    def apply_mask(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.shape[1] != 1:
            raise ValueError(f"Expected mask channel=1, got {mask.shape}")
        return image * (1.0 - mask) + self.mask_fill * mask

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked = self.apply_mask(image, mask)
        spatial = self.backbone.get_spatial_features(masked)
        return self.decoder(spatial)
