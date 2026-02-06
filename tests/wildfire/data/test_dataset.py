from __future__ import annotations

from typing import cast

import numpy as np
import pytest
import torch

from wildfire.data.dataset import WildfireSequenceDataset


def test_dataset_single_sample_tensor_shapes() -> None:
    z = np.arange(6 * 4, dtype=np.float32).reshape(6, 4)
    w = np.arange(6 * 3, dtype=np.float32).reshape(6, 3)
    g = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    ds = WildfireSequenceDataset(
        embeddings_source={"fire_a": z},
        weather_source={"fire_a": w},
        static_source={"fire_a": g},
        history=5,
    )

    assert len(ds) == 1
    sample = ds[0]
    z_in = cast(torch.Tensor, sample["z_in"])
    z_target = cast(torch.Tensor, sample["z_target"])
    w_in = cast(torch.Tensor, sample["w_in"])
    g_t = cast(torch.Tensor, sample["g"])
    assert sample["fire_id"] == "fire_a"
    assert tuple(z_in.shape) == (5, 4)
    assert tuple(z_target.shape) == (4,)
    assert tuple(w_in.shape) == (5, 3)
    assert tuple(g_t.shape) == (3,)


def test_dataset_multiple_windows_with_stride() -> None:
    z = np.arange(9 * 2, dtype=np.float32).reshape(9, 2)
    w = np.arange(9 * 5, dtype=np.float32).reshape(9, 5)
    g = np.array([1.0, 2.0], dtype=np.float32)

    ds = WildfireSequenceDataset(
        embeddings_source={"fire_a": z},
        weather_source={"fire_a": w},
        static_source={"fire_a": g},
        history=5,
        stride=2,
    )

    # target indices: 5, 7
    assert len(ds) == 2
    s0 = ds[0]
    s1 = ds[1]
    s0_target = cast(torch.Tensor, s0["z_target"])
    s1_target = cast(torch.Tensor, s1["z_target"])
    assert np.allclose(s0_target.numpy(), z[5])
    assert np.allclose(s1_target.numpy(), z[7])


def test_dataset_return_numpy() -> None:
    z = np.arange(6 * 4, dtype=np.float32).reshape(6, 4)
    w = np.arange(6 * 2, dtype=np.float32).reshape(6, 2)
    g = np.array([0.4, 0.7], dtype=np.float32)

    ds = WildfireSequenceDataset(
        embeddings_source={"fire_a": z},
        weather_source={"fire_a": w},
        static_source={"fire_a": g},
        history=5,
        return_tensors=False,
    )

    sample = ds[0]
    assert isinstance(sample["z_in"], np.ndarray)
    assert isinstance(sample["z_target"], np.ndarray)
    assert isinstance(sample["w_in"], np.ndarray)
    assert isinstance(sample["g"], np.ndarray)


def test_dataset_raises_for_missing_weather() -> None:
    z = np.arange(6 * 4, dtype=np.float32).reshape(6, 4)
    g = np.array([0.1, 0.2], dtype=np.float32)

    with pytest.raises(ValueError, match="missing weather"):
        WildfireSequenceDataset(
            embeddings_source={"fire_a": z},
            weather_source={},
            static_source={"fire_a": g},
            history=5,
        )


def test_dataset_raises_for_timestep_mismatch() -> None:
    z = np.arange(6 * 4, dtype=np.float32).reshape(6, 4)
    w = np.arange(5 * 2, dtype=np.float32).reshape(5, 2)
    g = np.array([0.1, 0.2], dtype=np.float32)

    with pytest.raises(ValueError, match="timesteps"):
        WildfireSequenceDataset(
            embeddings_source={"fire_a": z},
            weather_source={"fire_a": w},
            static_source={"fire_a": g},
            history=5,
        )
