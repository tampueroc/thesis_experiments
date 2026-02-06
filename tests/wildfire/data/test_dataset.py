from __future__ import annotations

import numpy as np
import pytest

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
    z_in = np.asarray(sample["z_in"])
    z_target = np.asarray(sample["z_target"])
    w_in = np.asarray(sample["w_in"])
    g_num = np.asarray(sample["g_num"])
    g_num_mask = np.asarray(sample["g_num_mask"])
    g_cat = np.asarray(sample["g_cat"])
    assert sample["fire_id"] == "fire_a"
    assert tuple(z_in.shape) == (5, 4)
    assert tuple(z_target.shape) == (4,)
    assert tuple(w_in.shape) == (5, 3)
    assert tuple(g_num.shape) == (2,)
    assert tuple(g_num_mask.shape) == (2,)
    assert tuple(g_cat.shape) == (1,)


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
    s0_target = np.asarray(s0["z_target"])
    s1_target = np.asarray(s1["z_target"])
    assert np.allclose(s0_target, z[5])
    assert np.allclose(s1_target, z[7])


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
    assert isinstance(sample["g_num"], np.ndarray)
    assert isinstance(sample["g_num_mask"], np.ndarray)
    assert isinstance(sample["g_cat"], np.ndarray)


def test_dataset_static_feature_encoding_missing_and_unk() -> None:
    z = np.arange(6 * 4, dtype=np.float32).reshape(6, 4)
    w = np.arange(6 * 2, dtype=np.float32).reshape(6, 2)
    g = np.array([-1.0, 0.7, -1.0, 2.0], dtype=np.float32)
    ds = WildfireSequenceDataset(
        embeddings_source={"fire_a": z},
        weather_source={"fire_a": w},
        static_source={"fire_a": g},
        history=5,
        return_tensors=False,
        static_categorical_indices=(0, 3),
    )
    sample = ds[0]
    g_num = np.asarray(sample["g_num"], dtype=np.float32)
    g_num_mask = np.asarray(sample["g_num_mask"], dtype=np.float32)
    g_cat = np.asarray(sample["g_cat"], dtype=np.int64)
    assert np.allclose(g_num, np.array([0.7, 0.0], dtype=np.float32))
    assert np.allclose(g_num_mask, np.array([1.0, 0.0], dtype=np.float32))
    # cat[0]=-1 -> UNK=0, cat[1]=2 remains raw code 2.
    assert np.array_equal(g_cat, np.array([0, 2], dtype=np.int64))


def test_dataset_keeps_raw_fuel_codes_for_categorical_feature() -> None:
    z = np.arange(6 * 2, dtype=np.float32).reshape(6, 2)
    w = np.arange(6 * 2, dtype=np.float32).reshape(6, 2)

    ds_known = WildfireSequenceDataset(
        embeddings_source={"fire_a": z},
        weather_source={"fire_a": w},
        static_source={"fire_a": np.array([101.0, 0.5], dtype=np.float32)},
        history=5,
        return_tensors=False,
        static_categorical_indices=(0,),
    )
    ds_unknown = WildfireSequenceDataset(
        embeddings_source={"fire_a": z},
        weather_source={"fire_a": w},
        static_source={"fire_a": np.array([9999.0, 0.5], dtype=np.float32)},
        history=5,
        return_tensors=False,
        static_categorical_indices=(0,),
    )

    known_cat = np.asarray(ds_known[0]["g_cat"], dtype=np.int64)
    unknown_cat = np.asarray(ds_unknown[0]["g_cat"], dtype=np.int64)
    assert known_cat.shape == (1,)
    assert unknown_cat.shape == (1,)
    assert int(known_cat[0]) == 101
    assert int(unknown_cat[0]) == 9999


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
