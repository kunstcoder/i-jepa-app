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
        """x: [B, N, D] -> [B, num_classes]"""
        B = x.shape[0]
        query = self.query.expand(B, -1, -1)
        pooled, _ = self.attn(query, x, x)
        pooled = self.norm(pooled.squeeze(1))
        return self.head(pooled)


class IJEPAClassifier(nn.Module):
    """Classification model built on top of a frozen I-JEPA backbone.

    Supports two head modes:
    - 'linear':    global average pooling + single linear layer
    - 'attentive': attention-based pooling + MLP head
    """

    def __init__(
        self,
        backbone: IJEPABackbone,
        num_classes: int,
        head_mode: str = "linear",
    ):
        super().__init__()
        self.backbone = backbone

        if head_mode == "linear":
            self.head = nn.Linear(backbone.embed_dim, num_classes)
        elif head_mode == "attentive":
            self.head = AttentiveProbe(backbone.embed_dim, num_classes)
        else:
            raise ValueError(f"Unknown head_mode: {head_mode}")

        self.head_mode = head_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)  # [B, N, D]
        if self.head_mode == "linear":
            pooled = features.mean(dim=1)  # [B, D]
            return self.head(pooled)
        else:
            return self.head(features)
