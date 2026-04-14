from __future__ import annotations

from typing import Any, cast

import pytest
import torch

pytest.importorskip("torch.nn", exc_type=ImportError)

from wildfire.model_latent_predictor.transformer_01 import TransformerConfig, TransformerLatentPredictor

TORCH = cast(Any, torch)


def test_transformer_latent_predictor_forward_shape() -> None:
    config = TransformerConfig(
        input_dim=16,
        output_dim=16,
        d_model=32,
        nhead=8,
        num_layers=2,
        dim_feedforward=64,
        dropout=0.1,
        max_history=5,
    )
    model = TransformerLatentPredictor(config)
    z_in = TORCH.randn(4, 5, 16)

    out = model(z_in)

    assert tuple(out.shape) == (4, 16)


def test_transformer_latent_predictor_rejects_history_longer_than_config() -> None:
    config = TransformerConfig(
        input_dim=8,
        output_dim=8,
        d_model=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
        max_history=5,
    )
    model = TransformerLatentPredictor(config)
    z_in = TORCH.randn(2, 6, 8)

    with pytest.raises(ValueError, match="exceeds max_history"):
        model(z_in)
