from __future__ import annotations

from typing import Any, cast

import pytest
import torch

pytest.importorskip("torch.nn", exc_type=ImportError)

from wildfire.model_latent_predictor.transformer_static_01 import (
    StaticTransformerConfig,
    StaticTransformerLatentPredictor,
)

TORCH = cast(Any, torch)


def test_static_transformer_latent_predictor_forward_shape() -> None:
    config = StaticTransformerConfig(
        input_dim=16,
        output_dim=16,
        static_num_dim=3,
        static_cat_dim=1,
        d_model=32,
        nhead=8,
        num_layers=2,
        dim_feedforward=64,
        max_history=5,
    )
    model = StaticTransformerLatentPredictor(config)

    out = model(
        TORCH.randn(4, 5, 16),
        TORCH.randn(4, 3),
        TORCH.ones(4, 3),
        TORCH.zeros(4, 1),
    )

    assert tuple(out.shape) == (4, 16)
