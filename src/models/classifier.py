import torch
import torch.nn as nn

from src.models.ijepa_backbone import IJEPABackbone


class AttentiveProbe(nn.Module):
    """Attention-based pooling followed by an MLP head."""

    def __init__(self, embed_dim: int, num_classes: int, num_heads: int = 1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        query = self.query.expand(B, -1, -1)
        pooled, _ = self.attn(query, x, x)
        pooled = self.norm(pooled.squeeze(1))
        return self.head(pooled)


class MultiLayerPoolingHead(nn.Module):
    """Concat pooled features from last-n layers and classify."""

    def __init__(self, embed_dim: int, num_classes: int, last_n: int = 4, hidden_dim: int | None = None):
        super().__init__()
        self.last_n = last_n
        in_dim = embed_dim * last_n
        hidden_dim = hidden_dim or in_dim
        self.norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, layers: list[torch.Tensor]) -> torch.Tensor:
        pooled = [t.mean(dim=1) for t in layers]
        z = torch.cat(pooled, dim=-1)
        z = self.norm(z)
        return self.mlp(z)


class GlobalLocalFusionHead(nn.Module):
    """Global mean pooling + tiny local conv adapter branch."""

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        grid_size: int,
        local_ratio: float = 0.5,
        hidden_dim: int | None = None,
    ):
        super().__init__()
        local_dim = max(64, int(embed_dim * local_ratio))
        self.grid_size = grid_size
        self.local_adapter = nn.Sequential(
            nn.Conv2d(embed_dim, local_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(local_dim, local_dim, kernel_size=3, padding=1, groups=local_dim),
            nn.GELU(),
        )
        fusion_dim = embed_dim + local_dim
        hidden_dim = hidden_dim or fusion_dim
        self.global_norm = nn.LayerNorm(embed_dim)
        self.local_norm = nn.LayerNorm(local_dim)
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, N, D = tokens.shape
        g = self.global_norm(tokens.mean(dim=1))

        H = W = self.grid_size
        if N != H * W:
            raise ValueError(f"Token count {N} is not compatible with grid {H}x{W}")
        x = tokens.transpose(1, 2).reshape(B, D, H, W)
        x = self.local_adapter(x)
        l = x.mean(dim=(2, 3))
        l = self.local_norm(l)

        z = torch.cat([g, l], dim=-1)
        return self.head(z)


class IJEPAClassifier(nn.Module):
    """Classification model built on top of I-JEPA backbone.

    Supported head modes:
    - linear: mean pooling + linear
    - attentive: attention pooling head
    - multi_layer_mlp: concatenate pooled features from last-n layers
    - global_local_fusion: global mean + tiny local conv adapter
    """

    def __init__(
        self,
        backbone: IJEPABackbone,
        num_classes: int,
        head_mode: str = "linear",
        pool_last_n: int = 4,
        head_hidden_dim: int | None = None,
        local_ratio: float = 0.5,
    ):
        super().__init__()
        self.backbone = backbone
        self.head_mode = head_mode
        self.pool_last_n = pool_last_n

        if head_mode == "linear":
            self.head = nn.Linear(backbone.embed_dim, num_classes)
        elif head_mode == "attentive":
            self.head = AttentiveProbe(backbone.embed_dim, num_classes)
        elif head_mode == "multi_layer_mlp":
            self.head = MultiLayerPoolingHead(
                backbone.embed_dim,
                num_classes,
                last_n=pool_last_n,
                hidden_dim=head_hidden_dim,
            )
        elif head_mode == "global_local_fusion":
            self.head = GlobalLocalFusionHead(
                backbone.embed_dim,
                num_classes,
                grid_size=backbone.num_patches_per_side,
                local_ratio=local_ratio,
                hidden_dim=head_hidden_dim,
            )
        else:
            raise ValueError(f"Unknown head_mode: {head_mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.head_mode == "multi_layer_mlp":
            layers = self.backbone.get_last_n_layer_tokens(x, last_n=self.pool_last_n)
            return self.head(layers)

        features = self.backbone(x)
        if self.head_mode == "linear":
            return self.head(features.mean(dim=1))
        return self.head(features)
