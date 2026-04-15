from __future__ import annotations

from typing import Any, cast

import pytest
import torch

pytest.importorskip("torch.nn", exc_type=ImportError)

from wildfire.model_latent_decoder.conditional_unet_binary_01 import (
    BinaryMaskConditionalUNetConfig,
    BinaryMaskConditionalUNetDecoder,
)

TORCH = cast(Any, torch)


def test_binary_conditional_unet_decoder_forward_shape() -> None:
    config = BinaryMaskConditionalUNetConfig(
        embedding_dim=16,
        base_channels=8,
        num_pool_layers=4,
        conditioning_channels_per_embedding=4,
        bottleneck_spatial_size=25,
    )
    model = BinaryMaskConditionalUNetDecoder(config)
    prev_image = TORCH.randn(2, 1, 400, 400)
    prev_embedding = TORCH.randn(2, 16)
    target_embedding = TORCH.randn(2, 16)

    out = model(prev_image, prev_embedding, target_embedding)

    assert tuple(out.shape) == (2, 1, 400, 400)
