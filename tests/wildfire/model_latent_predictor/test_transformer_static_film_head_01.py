from __future__ import annotations

from typing import Any, cast

import pytest
import torch

pytest.importorskip("torch.nn", exc_type=ImportError)

from wildfire.model_latent_predictor.transformer_static_film_head_01 import (
    StaticFiLMHeadTransformerConfig,
    StaticFiLMHeadTransformerLatentPredictor,
)

TORCH = cast(Any, torch)


def test_static_film_head_transformer_latent_predictor_forward_shape() -> None:
    config = StaticFiLMHeadTransformerConfig(
        input_dim=16,
        output_dim=16,
        static_num_dim=3,
        static_cat_dim=1,
        static_cat_vocab_size=8,
        d_model=32,
        nhead=8,
        num_layers=2,
        dim_feedforward=64,
        max_history=5,
    )
    model = StaticFiLMHeadTransformerLatentPredictor(config)

    out = model(
        TORCH.randn(4, 5, 16),
        TORCH.randn(4, 3),
        TORCH.ones(4, 3),
        TORCH.zeros(4, 1, dtype=TORCH.int64),
    )

    assert tuple(out.shape) == (4, 16)


def test_static_film_head_transformer_rejects_wrong_categorical_width() -> None:
    config = StaticFiLMHeadTransformerConfig(
        input_dim=8,
        output_dim=8,
        static_num_dim=2,
        static_cat_dim=1,
        static_cat_vocab_size=4,
        d_model=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
        max_history=5,
    )
    model = StaticFiLMHeadTransformerLatentPredictor(config)

    with pytest.raises(ValueError, match="g_cat second dim"):
        model(
            TORCH.randn(2, 5, 8),
            TORCH.randn(2, 2),
            TORCH.ones(2, 2),
            TORCH.zeros(2, 2, dtype=TORCH.int64),
        )
