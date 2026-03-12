import torch
import torch.nn as nn

from src.models.vision_transformer import VIT_REGISTRY
from src.utils.checkpoint import load_ijepa_encoder


class IJEPABackbone(nn.Module):
    """Wrapper around I-JEPA's ViT encoder for downstream feature extraction."""

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

    def _is_frozen(self) -> bool:
        return not any(p.requires_grad for p in self.encoder.parameters())

    def freeze(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

    def unfreeze(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

    def _forward_impl(self, x: torch.Tensor, return_all_layers: bool = False):
        x = self.encoder.patch_embed(x)
        pos_embed = self.encoder.interpolate_pos_encoding(x, self.encoder.pos_embed)
        x = x + pos_embed

        if return_all_layers:
            layer_tokens = []
            for blk in self.encoder.blocks:
                x = blk(x)
                layer_tokens.append(self.encoder.norm(x))
            return self.encoder.norm(x), layer_tokens

        for blk in self.encoder.blocks:
            x = blk(x)
        return self.encoder.norm(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract final-layer patch tokens. Returns [B, N, D]."""
        if self._is_frozen():
            with torch.no_grad():
                return self._forward_impl(x, return_all_layers=False)
        return self._forward_impl(x, return_all_layers=False)

    def get_last_n_layer_tokens(self, x: torch.Tensor, last_n: int = 4) -> list[torch.Tensor]:
        """Return patch tokens from the last n transformer layers."""
        if last_n < 1:
            raise ValueError(f"last_n must be >=1, got {last_n}")

        if self._is_frozen():
            with torch.no_grad():
                _, layer_tokens = self._forward_impl(x, return_all_layers=True)
        else:
            _, layer_tokens = self._forward_impl(x, return_all_layers=True)

        last_n = min(last_n, len(layer_tokens))
        return layer_tokens[-last_n:]

    def get_spatial_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial feature map from final layer. Returns [B, D, H, W]."""
        features = self.forward(x)
        B, N, D = features.shape
        H = W = self.num_patches_per_side
        return features.transpose(1, 2).reshape(B, D, H, W)

    def train(self, mode=True):
        # Keep encoder in eval mode if frozen
        super().train(mode)
        if self._is_frozen():
            self.encoder.eval()
        return self
