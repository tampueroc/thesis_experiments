from dataclasses import dataclass

import torch.nn as nn


@dataclass(frozen=True)
class LSTMConfig:
    input_dim: int
    hidden_dim: int
    output_dim: int
    num_layers: int = 1
    dropout: float = 0.0
    bidirectional: bool = False
    batch_first: bool = True

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if self.output_dim <= 0:
            raise ValueError("output_dim must be > 0")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be > 0")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dropout must be in [0.0, 1.0)")

class LatentLSTMPredictor(nn.Module):
    def __init__(self, config: LSTMConfig) -> None:
        super().__init__()
        self.config = config
        lstm_dropout = config.dropout if config.num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=lstm_dropout,
            bidirectional=config.bidirectional,
            batch_first=config.batch_first,
        )
        lstm_output_dim = config.hidden_dim * (2 if config.bidirectional else 1)
        self.proj = nn.Linear(lstm_output_dim, config.output_dim)

    def forward(self, z_in):
        if z_in.ndim != 3:
            raise ValueError(f"z_in must be rank-3, got shape {tuple(z_in.shape)}")
        lstm_out, _ = self.lstm(z_in)
        if self.config.batch_first:
            final_state = lstm_out[:, -1, :]
        else:
            final_state = lstm_out[-1, :, :]
        return self.proj(final_state)
