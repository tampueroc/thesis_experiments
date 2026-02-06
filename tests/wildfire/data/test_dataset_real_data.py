from __future__ import annotations

import os
import json
from pathlib import Path

import numpy as np
import pytest

from wildfire.data.dataset import WildfireSequenceDataset
from wildfire.data.real_data import build_sources, choose_timestamp, parse_sequence_num


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


def _encode_expected(g: np.ndarray, missing_value: float = -1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Dataset default schema: index 0 categorical, remaining numeric.
    g_cat_raw = np.asarray([g[0]], dtype=np.float32)
    g_num_raw = np.asarray(g[1:], dtype=np.float32)
    g_num_mask = (g_num_raw != missing_value).astype(np.float32)
    g_num = np.where(g_num_mask == 1.0, g_num_raw, 0.0).astype(np.float32)
    g_cat = np.where(g_cat_raw != missing_value, np.rint(g_cat_raw).astype(np.int64), 0).astype(np.int64)
    return g_num, g_num_mask, g_cat


@pytest.mark.real_data
def test_dataset_shapes_on_real_data() -> None:
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
    z_in = np.asarray(sample["z_in"])
    z_target = np.asarray(sample["z_target"])
    w_in = np.asarray(sample["w_in"])
    g_num = np.asarray(sample["g_num"])
    g_num_mask = np.asarray(sample["g_num_mask"])
    g_cat = np.asarray(sample["g_cat"])
    assert tuple(z_in.shape)[0] == 5
    assert len(z_target.shape) == 1
    assert tuple(w_in.shape)[0] == 5
    assert len(g_num.shape) == 1
    assert len(g_num_mask.shape) == 1
    assert len(g_cat.shape) == 1


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
    modality_dir = embeddings_root / timestamp / "fire_frames"
    model_dir = _resolve_available_model_dir(modality_dir)
    if model_dir is None:
        pytest.skip(f"no supported model dir found under: {modality_dir}")
    assert model_dir is not None

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


@pytest.mark.real_data
def test_dataset_sample_g_matches_manual_static_fetch() -> None:
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
    modality_dir = embeddings_root / timestamp / "fire_frames"
    model_dir = _resolve_available_model_dir(modality_dir)
    if model_dir is None:
        pytest.skip(f"no supported model dir found under: {modality_dir}")
    assert model_dir is not None

    z_by_fire, w_by_fire, g_by_fire = build_sources(
        model_dir=model_dir,
        landscape_dir=landscape_dir,
        max_sequences=32,
    )
    dataset = WildfireSequenceDataset(
        embeddings_source=z_by_fire,
        weather_source=w_by_fire,
        static_source=g_by_fire,
        history=5,
        return_tensors=False,
    )
    if len(dataset) == 0:
        pytest.skip("dataset has no samples for this real-data slice")

    g_values = np.asarray(np.load(values_path), dtype=np.float32)
    alias_to_row = {str(k): int(v) for k, v in json.loads(keys_path.read_text(encoding="utf-8")).items()}

    checked_fire_ids: set[str] = set()
    for idx in range(len(dataset)):
        sample = dataset[idx]
        fire_id = str(sample["fire_id"])
        if fire_id in checked_fire_ids:
            continue
        sequence_num = parse_sequence_num(fire_id)
        aliases = [
            fire_id,
            f"sequence_{sequence_num:03d}",
            f"sequence_{sequence_num:04d}",
            f"sequence_{sequence_num:05d}",
            str(sequence_num),
        ]
        row_idx = next((alias_to_row[a] for a in aliases if a in alias_to_row), None)
        assert row_idx is not None, f"missing alias mapping for sequence_id={fire_id}"
        expected = np.asarray(g_values[row_idx], dtype=np.float32)
        actual_num = np.asarray(sample["g_num"], dtype=np.float32)
        actual_mask = np.asarray(sample["g_num_mask"], dtype=np.float32)
        actual_cat = np.asarray(sample["g_cat"], dtype=np.int64)
        exp_num, exp_mask, exp_cat = _encode_expected(expected)
        assert actual_num.shape == exp_num.shape, f"g_num shape mismatch for sequence_id={fire_id}"
        assert actual_mask.shape == exp_mask.shape, f"g_num_mask shape mismatch for sequence_id={fire_id}"
        assert actual_cat.shape == exp_cat.shape, f"g_cat shape mismatch for sequence_id={fire_id}"
        assert np.allclose(actual_num, exp_num, equal_nan=True), f"dataset g_num mismatch for sequence_id={fire_id}"
        assert np.allclose(actual_mask, exp_mask, equal_nan=True), f"dataset g_num_mask mismatch for sequence_id={fire_id}"
        assert np.array_equal(actual_cat, exp_cat), f"dataset g_cat mismatch for sequence_id={fire_id}"
        checked_fire_ids.add(fire_id)

    assert len(checked_fire_ids) > 0
