import torch
import torch.nn as nn

from src.models.vision_transformer import VIT_REGISTRY
from src.utils.checkpoint import load_ijepa_encoder


class IJEPABackbone(nn.Module):
    """Wrapper around I-JEPA's ViT encoder for downstream feature extraction.

    Loads pretrained encoder weights and provides two output modes:
    - forward():              [B, N, D] patch-level features
    - get_spatial_features(): [B, D, H, W] spatial feature map
    """

    def __init__(
        self,
        model_name: str = "vit_huge",
        checkpoint_path: str | None = None,
        img_size: int = 224,
        patch_size: int = 14,
        freeze: bool = True,
    ):
        super().__init__()

        if model_name not in VIT_REGISTRY:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available: {list(VIT_REGISTRY.keys())}"
            )

        self.encoder = VIT_REGISTRY[model_name](
            patch_size=patch_size, img_size=[img_size],
        )
        self.embed_dim = self.encoder.embed_dim
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches_per_side = img_size // patch_size

        if checkpoint_path is not None:
            load_ijepa_encoder(checkpoint_path, self.encoder)

        if freeze:
            self.freeze()

    def freeze(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

    def unfreeze(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract patch-level features. Returns [B, N, D]."""
        return self.encoder(x)

    @torch.no_grad()
    def get_spatial_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial feature map. Returns [B, D, H, W]."""
        features = self.encoder(x)
        B, N, D = features.shape
        H = W = self.num_patches_per_side
        return features.transpose(1, 2).reshape(B, D, H, W)

    def train(self, mode=True):
        # Keep encoder in eval mode if frozen
        super().train(mode)
        if not any(p.requires_grad for p in self.encoder.parameters()):
            self.encoder.eval()
        return self
