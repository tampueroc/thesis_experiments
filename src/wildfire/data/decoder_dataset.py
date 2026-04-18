from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import torch
from PIL import Image

from wildfire.data.real_data import parse_sequence_num


ArrayMap = Mapping[str, np.ndarray]
FramePathMap = Mapping[str, list[Path]]
LandscapeMap = Mapping[str, np.ndarray]
NODATA_VALUE = -1.0
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
FRAME_INDEX_PATTERN = re.compile(r"(\d+)(?!.*\d)")


def _torch_from_numpy(array: np.ndarray) -> Any:
    return cast(Any, torch).from_numpy(array)


def load_mask_tensor(
    image_path: Path,
    *,
    image_channels: int,
    binarize_mask: bool = False,
    mask_threshold: float = 0.5,
) -> np.ndarray:
    if image_channels <= 0:
        raise ValueError("image_channels must be > 0")
    if mask_threshold < 0.0 or mask_threshold > 1.0:
        raise ValueError("mask_threshold must be in [0.0, 1.0]")
    with Image.open(image_path) as image:
        gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    if binarize_mask:
        gray = (gray >= mask_threshold).astype(np.float32, copy=False)
    if image_channels == 1:
        return gray[None, :, :].astype(np.float32, copy=False)
    chw = np.repeat(gray[None, :, :], image_channels, axis=0)
    return chw.astype(np.float32, copy=False)


def load_mask_rgb_tensor(image_path: Path) -> np.ndarray:
    return load_mask_tensor(image_path, image_channels=3)


def extract_frame_index(path: Path) -> int:
    match = FRAME_INDEX_PATTERN.search(path.stem)
    if match is None:
        raise ValueError(f"Could not parse frame index from filename: {path.name}")
    return int(match.group(1))


def collect_indexed_images(sequence_dir: Path) -> dict[int, Path]:
    indexed: dict[int, Path] = {}
    for file_path in sorted(sequence_dir.iterdir()):
        if not file_path.is_file() or file_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        frame_index = extract_frame_index(file_path)
        if frame_index in indexed:
            raise ValueError(f"duplicate frame index {frame_index} in {sequence_dir}")
        indexed[frame_index] = file_path
    return indexed


def build_decoder_sources(
    model_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, list[Path]]]:
    manifest_path = model_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sequences = manifest.get("sequences", [])
    if not sequences:
        raise RuntimeError(f"no sequences in manifest: {manifest_path}")

    input_dir_raw = manifest.get("input_dir")
    if not isinstance(input_dir_raw, str) or not input_dir_raw:
        raise ValueError(f"manifest missing input_dir: {manifest_path}")
    input_dir = Path(input_dir_raw)

    z_by_fire: dict[str, np.ndarray] = {}
    frames_by_fire: dict[str, list[Path]] = {}
    for seq in sequences:
        sequence_id = str(seq["sequence_id"])
        sequence_path = Path(str(seq["sequence_path"]))
        z = np.asarray(np.load(sequence_path), dtype=np.float32)
        if z.ndim != 2:
            raise ValueError(f"{sequence_id}: expected 2D embedding, got {z.shape}")

        sequence_dir = input_dir / sequence_id
        if not sequence_dir.exists():
            raise FileNotFoundError(f"sequence image directory not found: {sequence_dir}")
        indexed = collect_indexed_images(sequence_dir)
        ordered_frames = [indexed[idx] for idx in sorted(indexed)]
        if len(ordered_frames) < z.shape[0]:
            raise ValueError(
                f"{sequence_id}: image frames ({len(ordered_frames)}) < embedding steps ({z.shape[0]})"
            )
        if len(ordered_frames) != z.shape[0]:
            ordered_frames = ordered_frames[: z.shape[0]]

        z_by_fire[sequence_id] = z
        frames_by_fire[sequence_id] = ordered_frames

    return z_by_fire, frames_by_fire


def _resize_chw_nearest(chw: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    out_h, out_w = [int(v) for v in out_hw]
    if chw.ndim != 3:
        raise ValueError(f"expected CHW array, got {chw.shape}")
    if chw.shape[1:] == (out_h, out_w):
        return chw.astype(np.float32, copy=False)
    resized_channels: list[np.ndarray] = []
    for channel in chw:
        image = Image.fromarray(np.asarray(channel, dtype=np.float32), mode="F")
        resized = image.resize((out_w, out_h), resample=Image.Resampling.NEAREST)
        resized_channels.append(np.asarray(resized, dtype=np.float32))
    return np.stack(resized_channels, axis=0).astype(np.float32, copy=False)


def build_landscape_patch_sources(
    landscape_chw_dir: Path,
    indices_dir: Path,
    frames_by_fire: FramePathMap,
) -> dict[str, np.ndarray]:
    landscape_chw_path = landscape_chw_dir / "landscape_channels_chw.npy"
    indices_path = indices_dir / "indices.json"
    if not landscape_chw_path.exists():
        raise FileNotFoundError(f"landscape CHW array not found: {landscape_chw_path}")
    if not indices_path.exists():
        raise FileNotFoundError(f"indices.json not found: {indices_path}")

    landscape_chw = np.asarray(np.load(landscape_chw_path), dtype=np.float32)
    if landscape_chw.ndim != 3:
        raise ValueError(f"expected CHW landscape array, got {landscape_chw.shape}")
    indices = json.loads(indices_path.read_text(encoding="utf-8"))

    out: dict[str, np.ndarray] = {}
    for fire_id, frame_paths in frames_by_fire.items():
        if not frame_paths:
            raise ValueError(f"{fire_id}: empty frame path list")
        bbox = indices.get(str(parse_sequence_num(fire_id)))
        if not isinstance(bbox, list | tuple) or len(bbox) != 4:
            raise ValueError(f"{fire_id}: invalid bbox in indices.json: {bbox}")
        row_start, row_end, col_start, col_end = [int(v) for v in bbox]
        patch = landscape_chw[:, row_start:row_end, col_start:col_end]
        if patch.ndim != 3 or patch.shape[1] == 0 or patch.shape[2] == 0:
            raise ValueError(f"{fire_id}: invalid landscape patch shape {patch.shape}")
        with Image.open(frame_paths[0]) as image:
            width, height = image.convert("L").size
        out[fire_id] = _resize_chw_nearest(patch, (height, width))
    return out


def select_landscape_channels(
    landscape_by_fire: LandscapeMap,
    channel_indices: tuple[int, ...],
) -> dict[str, np.ndarray]:
    if not channel_indices:
        raise ValueError("channel_indices must be non-empty")
    out: dict[str, np.ndarray] = {}
    for fire_id, arr in landscape_by_fire.items():
        chw = np.asarray(arr, dtype=np.float32)
        if chw.ndim != 3:
            raise ValueError(f"{fire_id}: expected CHW landscape array, got {chw.shape}")
        max_idx = chw.shape[0] - 1
        for idx in channel_indices:
            if idx < 0 or idx > max_idx:
                raise ValueError(f"{fire_id}: channel index {idx} out of bounds for {chw.shape[0]} channels")
        out[str(fire_id)] = chw[np.asarray(channel_indices, dtype=np.int64)]
    return out


def compute_landscape_channel_stats(
    landscape_by_fire: LandscapeMap,
    nodata_value: float = NODATA_VALUE,
) -> tuple[np.ndarray, np.ndarray]:
    means: list[float] = []
    stds: list[float] = []
    sample = next(iter(landscape_by_fire.values()))
    channel_count = int(np.asarray(sample).shape[0])
    for channel_idx in range(channel_count):
        valid_values: list[np.ndarray] = []
        for arr in landscape_by_fire.values():
            plane = np.asarray(arr, dtype=np.float32)[channel_idx]
            valid = plane[plane != nodata_value]
            if valid.size > 0:
                valid_values.append(valid)
        if not valid_values:
            means.append(0.0)
            stds.append(1.0)
            continue
        merged = np.concatenate(valid_values, axis=0)
        mean = float(merged.mean())
        std = float(merged.std())
        means.append(mean)
        stds.append(std if std > 1e-8 else 1.0)
    return np.asarray(means, dtype=np.float32), np.asarray(stds, dtype=np.float32)


def normalize_landscape_sources(
    landscape_by_fire: LandscapeMap,
    mean: np.ndarray,
    std: np.ndarray,
    nodata_value: float = NODATA_VALUE,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for fire_id, arr in landscape_by_fire.items():
        chw = np.asarray(arr, dtype=np.float32).copy()
        if chw.ndim != 3:
            raise ValueError(f"{fire_id}: expected CHW landscape array, got {chw.shape}")
        if chw.shape[0] != mean.shape[0] or chw.shape[0] != std.shape[0]:
            raise ValueError(
                f"{fire_id}: channel mismatch for normalization, got {chw.shape[0]} vs mean/std {mean.shape[0]}"
            )
        for channel_idx in range(chw.shape[0]):
            plane = chw[channel_idx]
            valid_mask = plane != nodata_value
            if np.any(valid_mask):
                plane[valid_mask] = (plane[valid_mask] - mean[channel_idx]) / std[channel_idx]
            plane[~valid_mask] = 0.0
            chw[channel_idx] = plane
        out[str(fire_id)] = chw.astype(np.float32, copy=False)
    return out


@dataclass(frozen=True)
class _DecoderSampleIndex:
    fire_id: str
    prev_idx: int
    target_idx: int


class WildfireDecoderDataset:
    """Builds decoder samples aligned from frame pairs and latent embeddings."""

    def __init__(
        self,
        embeddings_source: ArrayMap,
        frames_source: FramePathMap,
        image_channels: int = 3,
        binarize_masks: bool = False,
        mask_threshold: float = 0.5,
        normalize_embeddings: bool = False,
        min_target_idx: int = 1,
        return_tensors: bool = True,
    ) -> None:
        if min_target_idx < 1:
            raise ValueError("min_target_idx must be >= 1")
        self._return_tensors = return_tensors
        self._image_channels = int(image_channels)
        self._binarize_masks = bool(binarize_masks)
        self._mask_threshold = float(mask_threshold)
        self._min_target_idx = int(min_target_idx)
        self._z_by_fire = {
            str(fire_id): np.asarray(values, dtype=np.float32)
            for fire_id, values in embeddings_source.items()
        }
        if normalize_embeddings:
            self._z_by_fire = self._normalize_per_fire_embeddings(self._z_by_fire)
        self._frames_by_fire = {
            str(fire_id): [Path(path) for path in paths]
            for fire_id, paths in frames_source.items()
        }
        self._validate_sources()
        self._sample_index = self._build_sample_index()

    def __len__(self) -> int:
        return len(self._sample_index)

    def __getitem__(self, idx: object) -> dict[str, object]:
        if not isinstance(idx, int):
            raise TypeError(f"index must be int, got {type(idx).__name__}")
        item = self._sample_index[idx]
        z = self._z_by_fire[item.fire_id]
        frames = self._frames_by_fire[item.fire_id]
        prev_image = load_mask_tensor(
            frames[item.prev_idx],
            image_channels=self._image_channels,
            binarize_mask=self._binarize_masks,
            mask_threshold=self._mask_threshold,
        )
        target_image = load_mask_tensor(
            frames[item.target_idx],
            image_channels=self._image_channels,
            binarize_mask=self._binarize_masks,
            mask_threshold=self._mask_threshold,
        )
        prev_embedding = z[item.prev_idx]
        target_embedding = z[item.target_idx]

        sample: dict[str, object] = {
            "prev_image": prev_image,
            "prev_embedding": prev_embedding,
            "target_embedding": target_embedding,
            "target_image": target_image,
            "fire_id": item.fire_id,
            "prev_idx": int(item.prev_idx),
            "target_idx": int(item.target_idx),
        }
        if not self._return_tensors:
            return sample
        return {
            "prev_image": _torch_from_numpy(prev_image),
            "prev_embedding": _torch_from_numpy(prev_embedding),
            "target_embedding": _torch_from_numpy(target_embedding),
            "target_image": _torch_from_numpy(target_image),
            "fire_id": item.fire_id,
            "prev_idx": int(item.prev_idx),
            "target_idx": int(item.target_idx),
        }

    def _validate_sources(self) -> None:
        missing_frames = sorted(set(self._z_by_fire) - set(self._frames_by_fire))
        if missing_frames:
            raise ValueError(f"missing frames for fire_ids: {missing_frames}")
        for fire_id, z in self._z_by_fire.items():
            if z.ndim != 2:
                raise ValueError(f"{fire_id}: expected 2D embedding array, got {z.shape}")
            frames = self._frames_by_fire[fire_id]
            if len(frames) != z.shape[0]:
                raise ValueError(
                    f"{fire_id}: frame count ({len(frames)}) != embedding steps ({z.shape[0]})"
                )
            if z.shape[0] < 2:
                raise ValueError(f"{fire_id}: need at least 2 timesteps, got {z.shape[0]}")

    def _build_sample_index(self) -> list[_DecoderSampleIndex]:
        index: list[_DecoderSampleIndex] = []
        for fire_id in sorted(self._z_by_fire):
            steps = int(self._z_by_fire[fire_id].shape[0])
            for target_idx in range(self._min_target_idx, steps):
                index.append(
                    _DecoderSampleIndex(
                        fire_id=fire_id,
                        prev_idx=target_idx - 1,
                        target_idx=target_idx,
                    )
                )
        return index

    @staticmethod
    def _normalize_per_fire_embeddings(items: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        normalized: dict[str, np.ndarray] = {}
        for fire_id, array in items.items():
            norms = np.linalg.norm(array, axis=1, keepdims=True)
            safe_norms = np.clip(norms, a_min=1e-8, a_max=None).astype(np.float32, copy=False)
            normalized[fire_id] = (array / safe_norms).astype(np.float32, copy=False)
        return normalized


class WildfireBlendedDecoderDataset:
    """Builds decoder samples from real previous embeddings and predicted target embeddings.

    Contract:
    - prev_image: real frame at t
    - prev_embedding: real embedding at t
    - target_embedding: predicted embedding at t+1
    - target_image: real frame at t+1
    """

    def __init__(
        self,
        prev_embeddings_source: ArrayMap,
        target_embeddings_source: ArrayMap,
        frames_source: FramePathMap,
        image_channels: int = 3,
        binarize_masks: bool = False,
        mask_threshold: float = 0.5,
        normalize_embeddings: bool = False,
        min_target_idx: int = 1,
        return_tensors: bool = True,
    ) -> None:
        if min_target_idx < 1:
            raise ValueError("min_target_idx must be >= 1")
        self._return_tensors = return_tensors
        self._image_channels = int(image_channels)
        self._binarize_masks = bool(binarize_masks)
        self._mask_threshold = float(mask_threshold)
        self._min_target_idx = int(min_target_idx)
        self._prev_z_by_fire = {
            str(fire_id): np.asarray(values, dtype=np.float32)
            for fire_id, values in prev_embeddings_source.items()
        }
        self._target_z_by_fire = {
            str(fire_id): np.asarray(values, dtype=np.float32)
            for fire_id, values in target_embeddings_source.items()
        }
        if normalize_embeddings:
            self._prev_z_by_fire = WildfireDecoderDataset._normalize_per_fire_embeddings(self._prev_z_by_fire)
            self._target_z_by_fire = WildfireDecoderDataset._normalize_per_fire_embeddings(self._target_z_by_fire)
        self._frames_by_fire = {
            str(fire_id): [Path(path) for path in paths]
            for fire_id, paths in frames_source.items()
        }
        self._validate_sources()
        self._sample_index = self._build_sample_index()

    def __len__(self) -> int:
        return len(self._sample_index)

    def __getitem__(self, idx: object) -> dict[str, object]:
        if not isinstance(idx, int):
            raise TypeError(f"index must be int, got {type(idx).__name__}")
        item = self._sample_index[idx]
        prev_z = self._prev_z_by_fire[item.fire_id]
        target_z = self._target_z_by_fire[item.fire_id]
        frames = self._frames_by_fire[item.fire_id]
        prev_image = load_mask_tensor(
            frames[item.prev_idx],
            image_channels=self._image_channels,
            binarize_mask=self._binarize_masks,
            mask_threshold=self._mask_threshold,
        )
        target_image = load_mask_tensor(
            frames[item.target_idx],
            image_channels=self._image_channels,
            binarize_mask=self._binarize_masks,
            mask_threshold=self._mask_threshold,
        )
        prev_embedding = prev_z[item.prev_idx]
        target_embedding = target_z[item.target_idx]

        sample: dict[str, object] = {
            "prev_image": prev_image,
            "prev_embedding": prev_embedding,
            "target_embedding": target_embedding,
            "target_image": target_image,
            "fire_id": item.fire_id,
            "prev_idx": int(item.prev_idx),
            "target_idx": int(item.target_idx),
        }
        if not self._return_tensors:
            return sample
        return {
            "prev_image": _torch_from_numpy(prev_image),
            "prev_embedding": _torch_from_numpy(prev_embedding),
            "target_embedding": _torch_from_numpy(target_embedding),
            "target_image": _torch_from_numpy(target_image),
            "fire_id": item.fire_id,
            "prev_idx": int(item.prev_idx),
            "target_idx": int(item.target_idx),
        }

    def _validate_sources(self) -> None:
        missing_target = sorted(set(self._prev_z_by_fire) - set(self._target_z_by_fire))
        if missing_target:
            raise ValueError(f"missing target embeddings for fire_ids: {missing_target}")
        missing_frames = sorted(set(self._prev_z_by_fire) - set(self._frames_by_fire))
        if missing_frames:
            raise ValueError(f"missing frames for fire_ids: {missing_frames}")
        for fire_id, prev_z in self._prev_z_by_fire.items():
            target_z = self._target_z_by_fire[fire_id]
            frames = self._frames_by_fire[fire_id]
            if prev_z.ndim != 2:
                raise ValueError(f"{fire_id}: expected 2D prev embedding array, got {prev_z.shape}")
            if target_z.ndim != 2:
                raise ValueError(f"{fire_id}: expected 2D target embedding array, got {target_z.shape}")
            if prev_z.shape != target_z.shape:
                raise ValueError(
                    f"{fire_id}: prev embedding shape {prev_z.shape} != target embedding shape {target_z.shape}"
                )
            if len(frames) != prev_z.shape[0]:
                raise ValueError(
                    f"{fire_id}: frame count ({len(frames)}) != embedding steps ({prev_z.shape[0]})"
                )
            if prev_z.shape[0] < 2:
                raise ValueError(f"{fire_id}: need at least 2 timesteps, got {prev_z.shape[0]}")

    def _build_sample_index(self) -> list[_DecoderSampleIndex]:
        index: list[_DecoderSampleIndex] = []
        for fire_id in sorted(self._prev_z_by_fire):
            steps = int(self._prev_z_by_fire[fire_id].shape[0])
            for target_idx in range(self._min_target_idx, steps):
                index.append(
                    _DecoderSampleIndex(
                        fire_id=fire_id,
                        prev_idx=target_idx - 1,
                        target_idx=target_idx,
                    )
                )
        return index


class WildfireLandscapeDecoderDataset:
    """Builds decoder samples with per-fire spatial landscape conditioning."""

    def __init__(
        self,
        embeddings_source: ArrayMap,
        frames_source: FramePathMap,
        landscape_source: LandscapeMap,
        image_channels: int = 1,
        binarize_masks: bool = False,
        mask_threshold: float = 0.5,
        normalize_embeddings: bool = False,
        min_target_idx: int = 1,
        return_tensors: bool = True,
    ) -> None:
        if min_target_idx < 1:
            raise ValueError("min_target_idx must be >= 1")
        self._return_tensors = return_tensors
        self._image_channels = int(image_channels)
        self._binarize_masks = bool(binarize_masks)
        self._mask_threshold = float(mask_threshold)
        self._min_target_idx = int(min_target_idx)
        self._z_by_fire = {
            str(fire_id): np.asarray(values, dtype=np.float32)
            for fire_id, values in embeddings_source.items()
        }
        if normalize_embeddings:
            self._z_by_fire = WildfireDecoderDataset._normalize_per_fire_embeddings(self._z_by_fire)
        self._frames_by_fire = {
            str(fire_id): [Path(path) for path in paths]
            for fire_id, paths in frames_source.items()
        }
        self._landscape_by_fire = {
            str(fire_id): np.asarray(values, dtype=np.float32)
            for fire_id, values in landscape_source.items()
        }
        self._validate_sources()
        self._sample_index = self._build_sample_index()

    def __len__(self) -> int:
        return len(self._sample_index)

    def __getitem__(self, idx: object) -> dict[str, object]:
        if not isinstance(idx, int):
            raise TypeError(f"index must be int, got {type(idx).__name__}")
        item = self._sample_index[idx]
        z = self._z_by_fire[item.fire_id]
        frames = self._frames_by_fire[item.fire_id]
        prev_image = load_mask_tensor(
            frames[item.prev_idx],
            image_channels=self._image_channels,
            binarize_mask=self._binarize_masks,
            mask_threshold=self._mask_threshold,
        )
        target_image = load_mask_tensor(
            frames[item.target_idx],
            image_channels=self._image_channels,
            binarize_mask=self._binarize_masks,
            mask_threshold=self._mask_threshold,
        )
        landscape = self._landscape_by_fire[item.fire_id]
        prev_embedding = z[item.prev_idx]
        target_embedding = z[item.target_idx]

        sample: dict[str, object] = {
            "prev_image": prev_image,
            "landscape": landscape,
            "prev_embedding": prev_embedding,
            "target_embedding": target_embedding,
            "target_image": target_image,
            "fire_id": item.fire_id,
            "prev_idx": int(item.prev_idx),
            "target_idx": int(item.target_idx),
        }
        if not self._return_tensors:
            return sample
        return {
            "prev_image": _torch_from_numpy(prev_image),
            "landscape": _torch_from_numpy(landscape),
            "prev_embedding": _torch_from_numpy(prev_embedding),
            "target_embedding": _torch_from_numpy(target_embedding),
            "target_image": _torch_from_numpy(target_image),
            "fire_id": item.fire_id,
            "prev_idx": int(item.prev_idx),
            "target_idx": int(item.target_idx),
        }

    def _validate_sources(self) -> None:
        missing_frames = sorted(set(self._z_by_fire) - set(self._frames_by_fire))
        if missing_frames:
            raise ValueError(f"missing frames for fire_ids: {missing_frames}")
        missing_landscape = sorted(set(self._z_by_fire) - set(self._landscape_by_fire))
        if missing_landscape:
            raise ValueError(f"missing landscape for fire_ids: {missing_landscape}")
        for fire_id, z in self._z_by_fire.items():
            if z.ndim != 2:
                raise ValueError(f"{fire_id}: expected 2D embedding array, got {z.shape}")
            frames = self._frames_by_fire[fire_id]
            if len(frames) != z.shape[0]:
                raise ValueError(
                    f"{fire_id}: frame count ({len(frames)}) != embedding steps ({z.shape[0]})"
                )
            if z.shape[0] < 2:
                raise ValueError(f"{fire_id}: need at least 2 timesteps, got {z.shape[0]}")
            landscape = self._landscape_by_fire[fire_id]
            if landscape.ndim != 3:
                raise ValueError(f"{fire_id}: expected CHW landscape array, got {landscape.shape}")
            with Image.open(frames[0]) as image:
                width, height = image.convert("L").size
            if landscape.shape[1:] != (height, width):
                raise ValueError(
                    f"{fire_id}: landscape shape {landscape.shape[1:]} != frame size {(height, width)}"
                )

    def _build_sample_index(self) -> list[_DecoderSampleIndex]:
        index: list[_DecoderSampleIndex] = []
        for fire_id in sorted(self._z_by_fire):
            steps = int(self._z_by_fire[fire_id].shape[0])
            for target_idx in range(self._min_target_idx, steps):
                index.append(
                    _DecoderSampleIndex(
                        fire_id=fire_id,
                        prev_idx=target_idx - 1,
                        target_idx=target_idx,
                    )
                )
        return index
