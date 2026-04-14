from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn as nn

from wildfire.model_latent_predictor.transformer_01 import LearnedPositionalEncoding

TORCH = cast(Any, torch)


@dataclass(frozen=True)
class StaticHeadTransformerConfig:
    input_dim: int
    output_dim: int
    static_num_dim: int
    static_cat_dim: int
    static_cat_vocab_size: int
    d_model: int = 512
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_history: int = 5
    batch_first: bool = True
    norm_first: bool = False
    static_hidden_dim: int = 256
    static_cat_embed_dim: int = 32

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if self.output_dim <= 0:
            raise ValueError("output_dim must be > 0")
        if self.static_num_dim < 0:
            raise ValueError("static_num_dim must be >= 0")
        if self.static_cat_dim < 0:
            raise ValueError("static_cat_dim must be >= 0")
        if self.static_cat_vocab_size <= 0:
            raise ValueError("static_cat_vocab_size must be > 0")
        if self.static_num_dim == 0 and self.static_cat_dim == 0:
            raise ValueError("at least one static feature is required")
        if self.d_model <= 0:
            raise ValueError("d_model must be > 0")
        if self.nhead <= 0:
            raise ValueError("nhead must be > 0")
        if self.d_model % self.nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be > 0")
        if self.dim_feedforward <= 0:
            raise ValueError("dim_feedforward must be > 0")
        if self.max_history <= 0:
            raise ValueError("max_history must be > 0")
        if self.static_hidden_dim <= 0:
            raise ValueError("static_hidden_dim must be > 0")
        if self.static_cat_embed_dim <= 0:
            raise ValueError("static_cat_embed_dim must be > 0")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dropout must be in [0.0, 1.0)")


class StaticHeadTransformerLatentPredictor(nn.Module):
    def __init__(self, config: StaticHeadTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.input_dim, config.d_model)
        self.positional_encoding = LearnedPositionalEncoding(
            max_history=config.max_history,
            d_model=config.d_model,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=config.batch_first,
            norm_first=config.norm_first,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.num_layers,
        )
        self.static_cat_embeddings = nn.ModuleList(
            [
                nn.Embedding(config.static_cat_vocab_size, config.static_cat_embed_dim)
                for _ in range(config.static_cat_dim)
            ]
        )
        static_input_dim = (config.static_num_dim * 2) + (config.static_cat_dim * config.static_cat_embed_dim)
        self.static_proj = nn.Sequential(
            nn.Linear(static_input_dim, config.static_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.static_hidden_dim, config.d_model),
        )
        self.output_proj = nn.Linear(config.d_model * 2, config.output_dim)

    def forward(self, z_in: Any, g_num: Any, g_num_mask: Any, g_cat: Any) -> Any:
        if z_in.ndim != 3:
            raise ValueError(f"z_in must be rank-3, got shape {tuple(z_in.shape)}")
        if g_num.ndim != 2:
            raise ValueError(f"g_num must be rank-2, got shape {tuple(g_num.shape)}")
        if g_num_mask.ndim != 2:
            raise ValueError(f"g_num_mask must be rank-2, got shape {tuple(g_num_mask.shape)}")
        if g_cat.ndim != 2:
            raise ValueError(f"g_cat must be rank-2, got shape {tuple(g_cat.shape)}")
        if int(g_cat.shape[1]) != self.config.static_cat_dim:
            raise ValueError(
                f"g_cat second dim must be {self.config.static_cat_dim}, got shape {tuple(g_cat.shape)}"
            )

        x = self.input_proj(z_in)
        x = self.positional_encoding(x)
        encoded = self.encoder(x)
        if self.config.batch_first:
            final_state = encoded[:, -1, :]
        else:
            final_state = encoded[-1, :, :]

        cat_embeddings: list[Any] = []
        for idx, embedding in enumerate(self.static_cat_embeddings):
            cat_embeddings.append(embedding(g_cat[:, idx]))
        if cat_embeddings:
            cat_features = TORCH.cat(cat_embeddings, dim=1)
            static_features = TORCH.cat([g_num, g_num_mask, cat_features], dim=1)
        else:
            static_features = TORCH.cat([g_num, g_num_mask], dim=1)
        static_context = self.static_proj(static_features)
        fused = TORCH.cat([final_state, static_context], dim=1)
        return self.output_proj(fused)
