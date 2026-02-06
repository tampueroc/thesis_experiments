from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import pytest
import torch

from wildfire.data.dataset import WildfireSequenceDataset
from wildfire.data.real_data import build_sources, choose_timestamp


@pytest.mark.real_data
def test_dataset_shapes_on_real_data() -> None:
    if os.getenv("RELELA_ONLY_TEST") != "1":
        pytest.skip("set RELELA_ONLY_TEST=1 to run SSH real-data test")

    embeddings_root = Path("/home/tampuero/data/thesis_data/embeddings")
    landscape_dir = Path("/home/tampuero/data/thesis_data/landscape")
    if not embeddings_root.exists() or not landscape_dir.exists():
        pytest.skip("real_data paths not available on this machine")

    timestamp = choose_timestamp(embeddings_root, "")
    model_dir = embeddings_root / timestamp / "fire_frames" / "facebook__dinov2-small"
    if not model_dir.exists():
        pytest.skip(f"model dir missing: {model_dir}")

    z_by_fire, w_by_fire, g_by_fire = build_sources(
        model_dir=model_dir,
        landscape_dir=landscape_dir,
        max_sequences=128,
    )
    dataset = WildfireSequenceDataset(
        embeddings_source=z_by_fire,
        weather_source=w_by_fire,
        static_source=g_by_fire,
        history=5,
    )

    assert len(dataset) > 0
    sample = dataset[0]
    z_in = cast(torch.Tensor, sample["z_in"])
    z_target = cast(torch.Tensor, sample["z_target"])
    w_in = cast(torch.Tensor, sample["w_in"])
    g = cast(torch.Tensor, sample["g"])
    assert tuple(z_in.shape)[0] == 5
    assert len(z_target.shape) == 1
    assert tuple(w_in.shape)[0] == 5
    assert len(g.shape) == 1
