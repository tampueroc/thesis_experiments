from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import torch


ArrayMap = Mapping[str, np.ndarray]
SourceLike = str | Path | ArrayMap


def _torch_from_numpy(array: np.ndarray) -> Any:
    return cast(Any, torch).from_numpy(array)


@dataclass(frozen=True)
class _SampleIndex:
    fire_id: str
    start_idx: int
    target_idx: int


@dataclass(frozen=True)
class _AugmentedSampleIndex:
    fire_id: str
    target_idx: int
    history: int


class WildfireSequenceDataset:
    """Builds temporal training samples from precomputed per-sequence embeddings.

    Each sample has:
    - z_in: (history, E)
    - z_target: (E,)
    - w_in: (history, d_w)
    - g_num: (n_num,) with missing replaced by 0
    - g_num_mask: (n_num,) where 1=valid and 0=missing
    - g_cat: (n_cat,) categorical indices including UNK=0
    - fire_id: str
    """

    def __init__(
        self,
        embeddings_source: SourceLike,
        weather_source: SourceLike,
        static_source: SourceLike,
        history: int = 5,
        stride: int = 1,
        normalize_embeddings: bool = False,
        return_tensors: bool = True,
        static_categorical_indices: tuple[int, ...] = (0,),
        static_missing_value: float = -1.0,
        categorical_unknown_index: int = 0,
    ) -> None:
        if history < 1:
            raise ValueError("history must be >= 1")
        if stride < 1:
            raise ValueError("stride must be >= 1")

        self._history = history
        self._stride = stride
        self._return_tensors = return_tensors
        self._static_categorical_indices = tuple(sorted(set(static_categorical_indices)))
        self._static_missing_value = float(static_missing_value)
        if categorical_unknown_index < 0:
            raise ValueError("categorical_unknown_index must be >= 0")
        self._categorical_unknown_index = int(categorical_unknown_index)

        self._z_by_fire = self._load_per_fire_arrays(embeddings_source, expected_ndim=2)
        if normalize_embeddings:
            self._z_by_fire = self._normalize_per_fire_embeddings(self._z_by_fire)
        self._w_by_fire = self._load_per_fire_arrays(weather_source, expected_ndim=2)
        self._g_by_fire = self._load_per_fire_arrays(static_source, expected_ndim=1)

        missing_weather = sorted(set(self._z_by_fire) - set(self._w_by_fire))
        if missing_weather:
            raise ValueError(f"missing weather for fire_ids: {missing_weather}")

        missing_static = sorted(set(self._z_by_fire) - set(self._g_by_fire))
        if missing_static:
            raise ValueError(f"missing static vectors for fire_ids: {missing_static}")

        self._validate_shapes()
        self._sample_index = self._build_sample_index()

    @property
    def history(self) -> int:
        return self._history

    def __len__(self) -> int:
        return len(self._sample_index)

    def __getitem__(self, idx: object) -> dict[str, object]:
        if not isinstance(idx, int):
            raise TypeError(f"index must be int, got {type(idx).__name__}")
        item = self._sample_index[idx]
        z = self._z_by_fire[item.fire_id]
        w = self._w_by_fire[item.fire_id]
        g = self._g_by_fire[item.fire_id]

        z_in = z[item.start_idx : item.target_idx]
        z_target = z[item.target_idx]
        w_in = w[item.start_idx : item.target_idx]
        g_num, g_num_mask, g_cat = self._encode_static_features(g)

        if self._return_tensors:
            return {
                "z_in": _torch_from_numpy(z_in),
                "z_target": _torch_from_numpy(z_target),
                "w_in": _torch_from_numpy(w_in),
                "g_num": _torch_from_numpy(g_num),
                "g_num_mask": _torch_from_numpy(g_num_mask),
                "g_cat": _torch_from_numpy(g_cat),
                "fire_id": item.fire_id,
            }

        return {
            "z_in": z_in,
            "z_target": z_target,
            "w_in": w_in,
            "g_num": g_num,
            "g_num_mask": g_num_mask,
            "g_cat": g_cat,
            "fire_id": item.fire_id,
        }

    def _build_sample_index(self) -> list[_SampleIndex]:
        index: list[_SampleIndex] = []
        for fire_id in sorted(self._z_by_fire.keys()):
            timesteps = self._z_by_fire[fire_id].shape[0]
            min_target = self._history
            if timesteps <= min_target:
                continue
            for target_idx in range(min_target, timesteps, self._stride):
                start_idx = target_idx - self._history
                index.append(
                    _SampleIndex(
                        fire_id=fire_id,
                        start_idx=start_idx,
                        target_idx=target_idx,
                    )
                )
        return index

    def _validate_shapes(self) -> None:
        for fire_id, z in self._z_by_fire.items():
            w = self._w_by_fire[fire_id]
            g = self._g_by_fire[fire_id]
            if z.shape[0] != w.shape[0]:
                raise ValueError(
                    f"{fire_id}: z timesteps ({z.shape[0]}) != w timesteps ({w.shape[0]})"
                )
            if g.ndim != 1:
                raise ValueError(f"{fire_id}: g must be 1D, got shape {g.shape}")
            if g.shape[0] == 0:
                raise ValueError(f"{fire_id}: g must have at least one element")
            for idx in self._static_categorical_indices:
                if idx < 0 or idx >= g.shape[0]:
                    raise ValueError(
                        f"{fire_id}: static categorical index {idx} out of bounds for g dim {g.shape[0]}"
                    )

    def _encode_static_features(self, g: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cat_indices = np.asarray(self._static_categorical_indices, dtype=np.int64)
        num_indices = np.asarray(
            [i for i in range(g.shape[0]) if i not in self._static_categorical_indices],
            dtype=np.int64,
        )

        g_num_raw = g[num_indices] if num_indices.size else np.zeros((0,), dtype=np.float32)
        g_num_mask = (g_num_raw != self._static_missing_value).astype(np.float32)
        g_num = np.where(g_num_mask == 1.0, g_num_raw, 0.0).astype(np.float32)

        if cat_indices.size:
            g_cat_raw = g[cat_indices]
            cat_valid = g_cat_raw != self._static_missing_value
            g_cat = np.full(cat_indices.shape[0], np.int64(self._categorical_unknown_index), dtype=np.int64)
            for i, _cat_idx in enumerate(cat_indices.tolist()):
                if not bool(cat_valid[i]):
                    continue
                raw_value = int(np.rint(g_cat_raw[i]))
                g_cat[i] = np.int64(raw_value)
        else:
            g_cat = np.zeros((0,), dtype=np.int64)
        return g_num, g_num_mask, g_cat

    @staticmethod
    def _load_per_fire_arrays(source: SourceLike, expected_ndim: int) -> dict[str, np.ndarray]:
        if isinstance(source, Mapping):
            loaded: dict[str, np.ndarray] = {
                str(k): np.asarray(v, dtype=np.float32) for k, v in source.items()
            }
            WildfireSequenceDataset._validate_ndim(loaded, expected_ndim=expected_ndim)
            return loaded

        source_path = Path(source)
        if source_path.is_file():
            raise ValueError(
                f"Expected a directory for source={source_path}, got file. "
                "Use a directory containing <fire_id>.npy files."
            )
        if not source_path.exists() or not source_path.is_dir():
            raise FileNotFoundError(f"source directory not found: {source_path}")

        loaded: dict[str, np.ndarray] = {}
        for npy_path in sorted(source_path.glob("*.npy")):
            fire_id = npy_path.stem
            loaded[fire_id] = np.asarray(np.load(npy_path), dtype=np.float32)
        if not loaded:
            raise ValueError(f"no .npy files found in {source_path}")

        WildfireSequenceDataset._validate_ndim(loaded, expected_ndim=expected_ndim)
        return loaded

    @staticmethod
    def _validate_ndim(items: dict[str, np.ndarray], expected_ndim: int) -> None:
        for fire_id, array in items.items():
            if array.ndim != expected_ndim:
                raise ValueError(
                    f"{fire_id}: expected {expected_ndim}D array, got shape {array.shape}"
                )

    @staticmethod
    def _normalize_per_fire_embeddings(items: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        normalized: dict[str, np.ndarray] = {}
        for fire_id, array in items.items():
            norms = np.linalg.norm(array, axis=1, keepdims=True)
            safe_norms = np.clip(norms, a_min=1e-8, a_max=None).astype(np.float32, copy=False)
            normalized[fire_id] = (array / safe_norms).astype(np.float32, copy=False)
        return normalized


class WildfireAugmentedSequenceDataset:
    """Builds temporal samples by enumerating multiple history lengths per target.

    For each fire_id and each target timestep t, this dataset emits one sample per
    history h in [history_min, history_max] where t-h >= 0.
    """

    def __init__(
        self,
        embeddings_source: SourceLike,
        weather_source: SourceLike,
        static_source: SourceLike,
        history_min: int = 2,
        history_max: int = 5,
        stride: int = 1,
        normalize_embeddings: bool = False,
        return_tensors: bool = True,
        static_categorical_indices: tuple[int, ...] = (0,),
        static_missing_value: float = -1.0,
        categorical_unknown_index: int = 0,
    ) -> None:
        if history_min < 1:
            raise ValueError("history_min must be >= 1")
        if history_max < history_min:
            raise ValueError("history_max must be >= history_min")
        if stride < 1:
            raise ValueError("stride must be >= 1")
        if categorical_unknown_index < 0:
            raise ValueError("categorical_unknown_index must be >= 0")

        self._history_min = history_min
        self._history_max = history_max
        self._stride = stride
        self._return_tensors = return_tensors
        self._static_categorical_indices = tuple(sorted(set(static_categorical_indices)))
        self._static_missing_value = float(static_missing_value)
        self._categorical_unknown_index = int(categorical_unknown_index)

        self._z_by_fire = WildfireSequenceDataset._load_per_fire_arrays(embeddings_source, expected_ndim=2)
        if normalize_embeddings:
            self._z_by_fire = WildfireSequenceDataset._normalize_per_fire_embeddings(self._z_by_fire)
        self._w_by_fire = WildfireSequenceDataset._load_per_fire_arrays(weather_source, expected_ndim=2)
        self._g_by_fire = WildfireSequenceDataset._load_per_fire_arrays(static_source, expected_ndim=1)

        missing_weather = sorted(set(self._z_by_fire) - set(self._w_by_fire))
        if missing_weather:
            raise ValueError(f"missing weather for fire_ids: {missing_weather}")
        missing_static = sorted(set(self._z_by_fire) - set(self._g_by_fire))
        if missing_static:
            raise ValueError(f"missing static for fire_ids: {missing_static}")

        self._validate_shapes()
        self._sample_index = self._build_sample_index()

    def __len__(self) -> int:
        return len(self._sample_index)

    @property
    def history_max(self) -> int:
        return self._history_max

    def __getitem__(self, idx: object) -> dict[str, object]:
        if not isinstance(idx, int):
            raise TypeError(f"index must be int, got {type(idx).__name__}")
        item = self._sample_index[idx]
        z = self._z_by_fire[item.fire_id]
        w = self._w_by_fire[item.fire_id]
        g = self._g_by_fire[item.fire_id]

        start_idx = item.target_idx - item.history
        z_in = z[start_idx : item.target_idx]
        z_target = z[item.target_idx]
        w_in = w[start_idx : item.target_idx]
        if item.history < self._history_max:
            pad_len = self._history_max - item.history
            z_pad = np.repeat(z_in[0:1], pad_len, axis=0)
            w_pad = np.repeat(w_in[0:1], pad_len, axis=0)
            z_in = np.concatenate([z_pad, z_in], axis=0)
            w_in = np.concatenate([w_pad, w_in], axis=0)
        g_num, g_num_mask, g_cat = self._encode_static_features(g)

        if self._return_tensors:
            return {
                "z_in": _torch_from_numpy(z_in),
                "z_target": _torch_from_numpy(z_target),
                "w_in": _torch_from_numpy(w_in),
                "g_num": _torch_from_numpy(g_num),
                "g_num_mask": _torch_from_numpy(g_num_mask),
                "g_cat": _torch_from_numpy(g_cat),
                "history": int(item.history),
                "fire_id": item.fire_id,
            }

        return {
            "z_in": z_in,
            "z_target": z_target,
            "w_in": w_in,
            "g_num": g_num,
            "g_num_mask": g_num_mask,
            "g_cat": g_cat,
            "history": int(item.history),
            "fire_id": item.fire_id,
        }

    def _build_sample_index(self) -> list[_AugmentedSampleIndex]:
        index: list[_AugmentedSampleIndex] = []
        for fire_id in sorted(self._z_by_fire.keys()):
            timesteps = self._z_by_fire[fire_id].shape[0]
            if timesteps <= self._history_min:
                continue
            for target_idx in range(self._history_min, timesteps, self._stride):
                max_history_for_target = min(self._history_max, target_idx)
                for history in range(self._history_min, max_history_for_target + 1):
                    index.append(
                        _AugmentedSampleIndex(
                            fire_id=fire_id,
                            target_idx=target_idx,
                            history=history,
                        )
                    )
        return index

    def _validate_shapes(self) -> None:
        for fire_id, z in self._z_by_fire.items():
            w = self._w_by_fire[fire_id]
            g = self._g_by_fire[fire_id]
            if z.shape[0] != w.shape[0]:
                raise ValueError(
                    f"{fire_id}: z timesteps ({z.shape[0]}) != w timesteps ({w.shape[0]})"
                )
            if g.ndim != 1:
                raise ValueError(f"{fire_id}: g must be 1D, got shape {g.shape}")
            if g.shape[0] == 0:
                raise ValueError(f"{fire_id}: g must have at least one element")
            for idx in self._static_categorical_indices:
                if idx < 0 or idx >= g.shape[0]:
                    raise ValueError(
                        f"{fire_id}: static categorical index {idx} out of bounds for g dim {g.shape[0]}"
                    )

    def _encode_static_features(self, g: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cat_indices = np.asarray(self._static_categorical_indices, dtype=np.int64)
        num_indices = np.asarray(
            [i for i in range(g.shape[0]) if i not in self._static_categorical_indices],
            dtype=np.int64,
        )

        g_num_raw = g[num_indices] if num_indices.size else np.zeros((0,), dtype=np.float32)
        g_num_mask = (g_num_raw != self._static_missing_value).astype(np.float32)
        g_num = np.where(g_num_mask == 1.0, g_num_raw, 0.0).astype(np.float32)

        if cat_indices.size:
            g_cat_raw = g[cat_indices]
            cat_valid = g_cat_raw != self._static_missing_value
            g_cat = np.full(cat_indices.shape[0], np.int64(self._categorical_unknown_index), dtype=np.int64)
            for i, _cat_idx in enumerate(cat_indices.tolist()):
                if not bool(cat_valid[i]):
                    continue
                raw_value = int(np.rint(g_cat_raw[i]))
                g_cat[i] = np.int64(raw_value)
        else:
            g_cat = np.zeros((0,), dtype=np.int64)
        return g_num, g_num_mask, g_cat
