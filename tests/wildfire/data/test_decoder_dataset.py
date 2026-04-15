from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from wildfire.data.decoder_dataset import WildfireDecoderDataset


def _write_gray_image(path: Path, value: int) -> None:
    array = np.full((400, 400), value, dtype=np.uint8)
    Image.fromarray(array, mode="L").save(path)


def test_decoder_dataset_single_sequence_shapes(tmp_path: Path) -> None:
    seq_dir = tmp_path / "sequence_001"
    seq_dir.mkdir()
    _write_gray_image(seq_dir / "fire_000.png", 0)
    _write_gray_image(seq_dir / "fire_001.png", 255)
    _write_gray_image(seq_dir / "fire_002.png", 128)

    z = np.arange(3 * 4, dtype=np.float32).reshape(3, 4)
    ds = WildfireDecoderDataset(
        embeddings_source={"sequence_001": z},
        frames_source={"sequence_001": [seq_dir / "fire_000.png", seq_dir / "fire_001.png", seq_dir / "fire_002.png"]},
        return_tensors=False,
    )

    assert len(ds) == 2
    sample = ds[0]
    prev_image = np.asarray(sample["prev_image"], dtype=np.float32)
    target_image = np.asarray(sample["target_image"], dtype=np.float32)
    prev_embedding = np.asarray(sample["prev_embedding"], dtype=np.float32)
    target_embedding = np.asarray(sample["target_embedding"], dtype=np.float32)

    assert tuple(prev_image.shape) == (3, 400, 400)
    assert tuple(target_image.shape) == (3, 400, 400)
    assert tuple(prev_embedding.shape) == (4,)
    assert tuple(target_embedding.shape) == (4,)
    assert float(prev_image.max()) == 0.0
    assert float(target_image.min()) == 1.0
    assert sample["prev_idx"] == 0
    assert sample["target_idx"] == 1


def test_decoder_dataset_normalizes_embeddings(tmp_path: Path) -> None:
    seq_dir = tmp_path / "sequence_001"
    seq_dir.mkdir()
    _write_gray_image(seq_dir / "fire_000.png", 0)
    _write_gray_image(seq_dir / "fire_001.png", 10)

    z = np.array([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32)
    ds = WildfireDecoderDataset(
        embeddings_source={"sequence_001": z},
        frames_source={"sequence_001": [seq_dir / "fire_000.png", seq_dir / "fire_001.png"]},
        normalize_embeddings=True,
        return_tensors=False,
    )

    sample = ds[0]
    prev_embedding = np.asarray(sample["prev_embedding"], dtype=np.float32)
    target_embedding = np.asarray(sample["target_embedding"], dtype=np.float32)
    assert np.allclose(prev_embedding, np.array([0.6, 0.8], dtype=np.float32))
    assert np.allclose(target_embedding, np.array([0.0, 1.0], dtype=np.float32))


def test_decoder_dataset_rejects_misaligned_counts(tmp_path: Path) -> None:
    seq_dir = tmp_path / "sequence_001"
    seq_dir.mkdir()
    _write_gray_image(seq_dir / "fire_000.png", 0)
    _write_gray_image(seq_dir / "fire_001.png", 10)
    z = np.arange(3 * 2, dtype=np.float32).reshape(3, 2)

    with pytest.raises(ValueError, match="frame count"):
        WildfireDecoderDataset(
            embeddings_source={"sequence_001": z},
            frames_source={"sequence_001": [seq_dir / "fire_000.png", seq_dir / "fire_001.png"]},
        )


def test_decoder_dataset_supports_single_channel_masks(tmp_path: Path) -> None:
    seq_dir = tmp_path / "sequence_001"
    seq_dir.mkdir()
    _write_gray_image(seq_dir / "fire_000.png", 0)
    _write_gray_image(seq_dir / "fire_001.png", 255)
    z = np.arange(2 * 2, dtype=np.float32).reshape(2, 2)
    ds = WildfireDecoderDataset(
        embeddings_source={"sequence_001": z},
        frames_source={"sequence_001": [seq_dir / "fire_000.png", seq_dir / "fire_001.png"]},
        image_channels=1,
        return_tensors=False,
    )

    sample = ds[0]
    prev_image = np.asarray(sample["prev_image"], dtype=np.float32)
    target_image = np.asarray(sample["target_image"], dtype=np.float32)
    assert tuple(prev_image.shape) == (1, 400, 400)
    assert tuple(target_image.shape) == (1, 400, 400)
