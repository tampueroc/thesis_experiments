from __future__ import annotations

from typing import Any, cast

import pytest
import torch

pytest.importorskip("torch.nn", exc_type=ImportError)

from wildfire.model_latent_decoder.conditional_unet_01 import (
    ConditionalUNetConfig,
    ConditionalUNetDecoder,
)

TORCH = cast(Any, torch)


def test_conditional_unet_decoder_forward_shape() -> None:
    config = ConditionalUNetConfig(
        image_channels=3,
        embedding_dim=16,
        base_channels=8,
        num_pool_layers=4,
        conditioning_channels_per_embedding=4,
        bottleneck_spatial_size=25,
    )
    model = ConditionalUNetDecoder(config)
    prev_image = TORCH.randn(2, 3, 400, 400)
    prev_embedding = TORCH.randn(2, 16)
    target_embedding = TORCH.randn(2, 16)

    out = model(prev_image, prev_embedding, target_embedding)

    assert tuple(out.shape) == (2, 3, 400, 400)


def test_conditional_unet_decoder_rejects_bad_embedding_rank() -> None:
    config = ConditionalUNetConfig(embedding_dim=8)
    model = ConditionalUNetDecoder(config)
    prev_image = TORCH.randn(1, 3, 400, 400)
    prev_embedding = TORCH.randn(1, 1, 8)
    target_embedding = TORCH.randn(1, 8)

    with pytest.raises(ValueError, match="prev_embedding must be rank-2"):
        model(prev_image, prev_embedding, target_embedding)
