from __future__ import annotations

from dataclasses import dataclass

from wildfire.model_latent_decoder.conditional_unet_01 import ConditionalUNetConfig, ConditionalUNetDecoder


@dataclass(frozen=True)
class BinaryMaskConditionalUNetConfig(ConditionalUNetConfig):
    image_channels: int = 1


class BinaryMaskConditionalUNetDecoder(ConditionalUNetDecoder):
    def __init__(self, config: BinaryMaskConditionalUNetConfig) -> None:
        super().__init__(config)
