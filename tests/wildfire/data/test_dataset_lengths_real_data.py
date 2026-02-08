from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from wildfire.data.dataset import WildfireAugmentedSequenceDataset, WildfireSequenceDataset
from wildfire.data.real_data import build_sources, choose_timestamp


def _resolve_available_model_dir(base_modality_dir: Path) -> Path | None:
    preferred = [
        "facebook__dinov2-small",
        "facebook__dinov2-large",
    ]
    for slug in preferred:
        candidate = base_modality_dir / slug
        if candidate.exists():
            return candidate
    return None


def _load_real_sources(max_sequences: int = 128) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    if os.getenv("RELELA_ONLY_TEST") != "1":
        pytest.skip("set RELELA_ONLY_TEST=1 to run SSH real-data test")

    embeddings_root = Path("/home/tampuero/data/thesis_data/embeddings")
    landscape_dir = Path("/home/tampuero/data/thesis_data/landscape")
    if not embeddings_root.exists() or not landscape_dir.exists():
        pytest.skip("real_data paths not available on this machine")

    timestamp = choose_timestamp(embeddings_root, "")
    modality_dir = embeddings_root / timestamp / "fire_frames"
    model_dir = _resolve_available_model_dir(modality_dir)
    if model_dir is None:
        pytest.skip(f"no supported model dir found under: {modality_dir}")
    assert model_dir is not None

    return build_sources(
        model_dir=model_dir,
        landscape_dir=landscape_dir,
        max_sequences=max_sequences,
    )


@pytest.mark.real_data
@pytest.mark.parametrize("history", [2, 3, 4, 5, 6], ids=lambda h: f"h{h}")
def test_real_data_fixed_window_lengths_history_2_to_6(history: int) -> None:
    z_by_fire, w_by_fire, g_by_fire = _load_real_sources(max_sequences=128)
    ds = WildfireSequenceDataset(
        embeddings_source=z_by_fire,
        weather_source=w_by_fire,
        static_source=g_by_fire,
        history=history,
        stride=1,
        return_tensors=False,
    )

    expected_len = sum(max(0, int(z.shape[0]) - history) for z in z_by_fire.values())
    assert len(ds) == expected_len


@pytest.mark.real_data
@pytest.mark.parametrize("history_max", [2, 3, 4, 5, 6], ids=lambda h: f"h2_{h}")
def test_real_data_augmented_lengths_history_2_to_6(history_max: int) -> None:
    history_min = 2
    z_by_fire, w_by_fire, g_by_fire = _load_real_sources(max_sequences=128)
    ds = WildfireAugmentedSequenceDataset(
        embeddings_source=z_by_fire,
        weather_source=w_by_fire,
        static_source=g_by_fire,
        history_min=history_min,
        history_max=history_max,
        stride=1,
        return_tensors=False,
    )

    expected_len = 0
    for z in z_by_fire.values():
        timesteps = int(z.shape[0])
        for target_idx in range(history_min, timesteps):
            expected_len += max(0, min(history_max, target_idx) - history_min + 1)

    assert len(ds) == expected_len
