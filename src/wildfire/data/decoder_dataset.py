from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import torch
from PIL import Image


ArrayMap = Mapping[str, np.ndarray]
FramePathMap = Mapping[str, list[Path]]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
FRAME_INDEX_PATTERN = re.compile(r"(\d+)(?!.*\d)")


def _torch_from_numpy(array: np.ndarray) -> Any:
    return cast(Any, torch).from_numpy(array)


def load_mask_rgb_tensor(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as image:
        gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    chw = np.repeat(gray[None, :, :], 3, axis=0)
    return chw.astype(np.float32, copy=False)


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
        normalize_embeddings: bool = False,
        return_tensors: bool = True,
    ) -> None:
        self._return_tensors = return_tensors
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
        prev_image = load_mask_rgb_tensor(frames[item.prev_idx])
        target_image = load_mask_rgb_tensor(frames[item.target_idx])
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
            for target_idx in range(1, steps):
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
