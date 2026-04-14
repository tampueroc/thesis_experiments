from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn as nn

TORCH = cast(Any, torch)


@dataclass(frozen=True)
class ConditionalUNetConfig:
    image_channels: int = 3
    embedding_dim: int = 384
    base_channels: int = 32
    num_pool_layers: int = 4
    conditioning_channels_per_embedding: int = 32
    bottleneck_spatial_size: int = 25

    def __post_init__(self) -> None:
        if self.image_channels <= 0:
            raise ValueError("image_channels must be > 0")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be > 0")
        if self.base_channels <= 0:
            raise ValueError("base_channels must be > 0")
        if self.num_pool_layers <= 0:
            raise ValueError("num_pool_layers must be > 0")
        if self.conditioning_channels_per_embedding <= 0:
            raise ValueError("conditioning_channels_per_embedding must be > 0")
        if self.bottleneck_spatial_size <= 0:
            raise ValueError("bottleneck_spatial_size must be > 0")


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Any) -> Any:
        return self.block(x)


class ConditionalUNetDecoder(nn.Module):
    def __init__(self, config: ConditionalUNetConfig) -> None:
        super().__init__()
        self.config = config

        encoder_channels = [
            config.base_channels * (2**idx) for idx in range(config.num_pool_layers)
        ]
        self.encoder_blocks = nn.ModuleList()
        in_channels = config.image_channels
        for out_channels in encoder_channels:
            self.encoder_blocks.append(ConvBlock(in_channels, out_channels))
            in_channels = out_channels
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        bottleneck_channels = config.base_channels * (2 ** config.num_pool_layers)
        self.bottleneck = ConvBlock(encoder_channels[-1], bottleneck_channels)
        projection_dim = (
            config.conditioning_channels_per_embedding
            * config.bottleneck_spatial_size
            * config.bottleneck_spatial_size
        )
        self.prev_embedding_proj = nn.Linear(config.embedding_dim, projection_dim)
        self.target_embedding_proj = nn.Linear(config.embedding_dim, projection_dim)
        fused_channels = bottleneck_channels + (2 * config.conditioning_channels_per_embedding)
        self.fused_bottleneck = ConvBlock(fused_channels, bottleneck_channels)

        decoder_in_channels = bottleneck_channels
        skip_channels = list(reversed(encoder_channels))
        self.upconvs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for skip_ch in skip_channels:
            self.upconvs.append(nn.ConvTranspose2d(decoder_in_channels, skip_ch, kernel_size=2, stride=2))
            self.decoder_blocks.append(ConvBlock(skip_ch * 2, skip_ch))
            decoder_in_channels = skip_ch

        self.output_conv = nn.Conv2d(config.base_channels, config.image_channels, kernel_size=1)

    def forward(
        self,
        prev_image: Any,
        prev_embedding: Any,
        target_embedding: Any,
    ) -> Any:
        if prev_image.ndim != 4:
            raise ValueError(f"prev_image must be rank-4, got shape {tuple(prev_image.shape)}")
        if prev_embedding.ndim != 2:
            raise ValueError(f"prev_embedding must be rank-2, got shape {tuple(prev_embedding.shape)}")
        if target_embedding.ndim != 2:
            raise ValueError(f"target_embedding must be rank-2, got shape {tuple(target_embedding.shape)}")

        x = prev_image
        skips: list[Any] = []
        for block in self.encoder_blocks:
            x = block(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        cond_prev = self._project_embedding(self.prev_embedding_proj, prev_embedding)
        cond_target = self._project_embedding(self.target_embedding_proj, target_embedding)
        x = TORCH.cat([x, cond_prev, cond_target], dim=1)
        x = self.fused_bottleneck(x)

        for upconv, decoder_block, skip in zip(self.upconvs, self.decoder_blocks, reversed(skips), strict=True):
            x = upconv(x)
            if x.shape[-2:] != skip.shape[-2:]:
                raise ValueError(
                    f"decoder spatial mismatch: upsampled={tuple(x.shape)} skip={tuple(skip.shape)}"
                )
            x = TORCH.cat([x, skip], dim=1)
            x = decoder_block(x)
        return self.output_conv(x)

    def _project_embedding(self, layer: nn.Linear, embedding: Any) -> Any:
        batch_size = int(embedding.shape[0])
        projected = layer(embedding)
        return projected.view(
            batch_size,
            self.config.conditioning_channels_per_embedding,
            self.config.bottleneck_spatial_size,
            self.config.bottleneck_spatial_size,
        )
