from __future__ import annotations

import os
import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch

from wildfire.data.dataset import WildfireSequenceDataset
from wildfire.data.real_data import build_sources, choose_timestamp, parse_sequence_num


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


@pytest.mark.real_data
def test_static_vectors_align_with_precomputed_sequence_g() -> None:
    if os.getenv("RELELA_ONLY_TEST") != "1":
        pytest.skip("set RELELA_ONLY_TEST=1 to run SSH real-data test")

    embeddings_root = Path("/home/tampuero/data/thesis_data/embeddings")
    landscape_dir = Path("/home/tampuero/data/thesis_data/landscape")
    values_path = landscape_dir / "sequence_static_g_values.npy"
    keys_path = landscape_dir / "sequence_static_g_keys.json"
    if not embeddings_root.exists() or not landscape_dir.exists():
        pytest.skip("real_data paths not available on this machine")
    if not values_path.exists() or not keys_path.exists():
        pytest.skip("precomputed sequence static g files are not available")

    timestamp = choose_timestamp(embeddings_root, "")
    model_dir = embeddings_root / timestamp / "fire_frames" / "facebook__dinov2-large"
    if not model_dir.exists():
        pytest.skip(f"model dir missing: {model_dir}")

    z_by_fire, _, g_by_fire = build_sources(
        model_dir=model_dir,
        landscape_dir=landscape_dir,
        max_sequences=64,
    )
    if not z_by_fire:
        pytest.skip("no sequences loaded from manifest/model dir")

    g_values = np.asarray(np.load(values_path), dtype=np.float32)
    alias_to_row = {str(k): int(v) for k, v in json.loads(keys_path.read_text(encoding="utf-8")).items()}

    checked = 0
    for sequence_id in sorted(g_by_fire.keys()):
        sequence_num = parse_sequence_num(sequence_id)
        aliases = [
            sequence_id,
            f"sequence_{sequence_num:03d}",
            f"sequence_{sequence_num:04d}",
            f"sequence_{sequence_num:05d}",
            str(sequence_num),
        ]
        row_idx = next((alias_to_row[a] for a in aliases if a in alias_to_row), None)
        assert row_idx is not None, f"missing alias mapping for sequence_id={sequence_id}"
        assert 0 <= row_idx < g_values.shape[0], f"row index out of bounds for sequence_id={sequence_id}"
        expected = g_values[row_idx]
        actual = np.asarray(g_by_fire[sequence_id], dtype=np.float32)
        assert actual.shape == expected.shape, f"shape mismatch for sequence_id={sequence_id}"
        assert np.allclose(actual, expected, equal_nan=True), f"g mismatch for sequence_id={sequence_id}"
        checked += 1

    assert checked > 0
