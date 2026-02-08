from __future__ import annotations

import numpy as np
import pytest

from wildfire.data.dataset import WildfireAugmentedSequenceDataset, WildfireSequenceDataset


@pytest.mark.parametrize("history", [2, 3, 4, 5, 6], ids=lambda h: f"h{h}")
def test_fixed_window_dataset_lengths_history_2_to_6(history: int) -> None:
    timesteps = 12
    z = np.arange(timesteps * 4, dtype=np.float32).reshape(timesteps, 4)
    w = np.arange(timesteps * 3, dtype=np.float32).reshape(timesteps, 3)
    g = np.array([1.0, 0.2], dtype=np.float32)

    ds = WildfireSequenceDataset(
        embeddings_source={"fire_a": z},
        weather_source={"fire_a": w},
        static_source={"fire_a": g},
        history=history,
        stride=1,
        return_tensors=False,
    )

    expected_len = timesteps - history
    assert len(ds) == expected_len


@pytest.mark.parametrize("history_max", [2, 3, 4, 5, 6], ids=lambda h: f"h2_{h}")
def test_augmented_dataset_lengths_history_2_to_6(history_max: int) -> None:
    timesteps = 12
    history_min = 2
    z = np.arange(timesteps * 4, dtype=np.float32).reshape(timesteps, 4)
    w = np.arange(timesteps * 3, dtype=np.float32).reshape(timesteps, 3)
    g = np.array([1.0, 0.2], dtype=np.float32)

    ds = WildfireAugmentedSequenceDataset(
        embeddings_source={"fire_a": z},
        weather_source={"fire_a": w},
        static_source={"fire_a": g},
        history_min=history_min,
        history_max=history_max,
        stride=1,
        return_tensors=False,
    )

    expected_len = sum(
        max(0, min(history_max, target_idx) - history_min + 1)
        for target_idx in range(history_min, timesteps)
    )
    assert len(ds) == expected_len


def test_augmented_dataset_pads_to_history_max() -> None:
    z = np.arange(8 * 2, dtype=np.float32).reshape(8, 2)
    w = np.arange(8 * 3, dtype=np.float32).reshape(8, 3)
    g = np.array([1.0, 0.4], dtype=np.float32)
    ds = WildfireAugmentedSequenceDataset(
        embeddings_source={"fire_a": z},
        weather_source={"fire_a": w},
        static_source={"fire_a": g},
        history_min=2,
        history_max=5,
        stride=1,
        return_tensors=False,
    )

    sample = ds[0]
    z_in = np.asarray(sample["z_in"])
    w_in = np.asarray(sample["w_in"])
    history_value = sample["history"]
    assert isinstance(history_value, (int, np.integer))
    assert int(history_value) == 2
    assert z_in.shape == (5, 2)
    assert w_in.shape == (5, 3)
