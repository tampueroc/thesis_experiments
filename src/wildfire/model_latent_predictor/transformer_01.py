from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn as nn

TORCH = cast(Any, torch)


@dataclass(frozen=True)
class TransformerConfig:
    input_dim: int
    output_dim: int
    d_model: int = 512
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_history: int = 5
    batch_first: bool = True
    norm_first: bool = False

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if self.output_dim <= 0:
            raise ValueError("output_dim must be > 0")
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
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dropout must be in [0.0, 1.0)")


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_history: int, d_model: int) -> None:
        super().__init__()
        self.max_history = max_history
        self.embedding = nn.Embedding(max_history, d_model)

    def forward(self, x: Any) -> Any:
        if x.ndim != 3:
            raise ValueError(f"x must be rank-3, got shape {tuple(x.shape)}")
        seq_len = int(x.shape[1])
        if seq_len > self.max_history:
            raise ValueError(f"seq_len={seq_len} exceeds max_history={self.max_history}")
        positions = TORCH.arange(seq_len, device=x.device)
        return x + self.embedding(positions).unsqueeze(0)


class TransformerLatentPredictor(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
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
        self.output_proj = nn.Linear(config.d_model, config.output_dim)

    def forward(self, z_in: Any) -> Any:
        if z_in.ndim != 3:
            raise ValueError(f"z_in must be rank-3, got shape {tuple(z_in.shape)}")
        x = self.input_proj(z_in)
        x = self.positional_encoding(x)
        encoded = self.encoder(x)
        if self.config.batch_first:
            final_state = encoded[:, -1, :]
        else:
            final_state = encoded[-1, :, :]
        return self.output_proj(final_state)
