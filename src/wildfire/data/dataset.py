from __future__ import annotations

import csv
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
        return_tensors: bool = True,
        static_categorical_indices: tuple[int, ...] = (0,),
        fuel_feature_index: int | None = 0,
        fuel_lookup_csv: str | Path | None = None,
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
        self._fuel_feature_index = fuel_feature_index
        self._static_missing_value = float(static_missing_value)
        if categorical_unknown_index < 0:
            raise ValueError("categorical_unknown_index must be >= 0")
        self._categorical_unknown_index = int(categorical_unknown_index)
        self._fuel_code_to_index = self._load_fuel_lookup(fuel_lookup_csv)

        self._z_by_fire = self._load_per_fire_arrays(embeddings_source, expected_ndim=2)
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
            for i, cat_idx in enumerate(cat_indices.tolist()):
                if not bool(cat_valid[i]):
                    continue
                raw_value = int(np.rint(g_cat_raw[i]))
                if self._fuel_feature_index is not None and cat_idx == self._fuel_feature_index:
                    g_cat[i] = np.int64(
                        self._fuel_code_to_index.get(raw_value, self._categorical_unknown_index)
                    )
                else:
                    g_cat[i] = np.int64(raw_value + 1)
        else:
            g_cat = np.zeros((0,), dtype=np.int64)
        return g_num, g_num_mask, g_cat

    @staticmethod
    def _default_fuel_lookup_path() -> Path:
        return Path(__file__).resolve().parents[1] / "resources" / "spain_lookup_table.csv"

    def _load_fuel_lookup(self, fuel_lookup_csv: str | Path | None) -> dict[int, int]:
        lookup_path = Path(fuel_lookup_csv) if fuel_lookup_csv is not None else self._default_fuel_lookup_path()
        if not lookup_path.exists():
            return {}

        codes: set[int] = set()
        with lookup_path.open(encoding="utf-8") as fh:
            reader = csv.reader(fh)
            for i, row in enumerate(reader):
                if not row:
                    continue
                if i == 0 and row[0].strip().lower() == "grid_value":
                    continue
                try:
                    code = int(row[0].strip())
                except (TypeError, ValueError):
                    continue
                codes.add(code)

        # Keep compatibility with historical usage where 141 is treated as non-fuel.
        codes.add(141)
        sorted_codes = sorted(codes)
        return {code: idx for idx, code in enumerate(sorted_codes, start=1)}

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
