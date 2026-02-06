from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
FRAME_INDEX_PATTERN = re.compile(r"(\d+)(?!.*\d)")
MODALITIES = ("fire_frames", "isochrones")


@dataclass(frozen=True)
class SequenceFiles:
    fire_id: str
    fire_frames: dict[int, Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute DINO embeddings for wildfire fire masks. "
            "For each sequence, saves one file with shape (T, E)."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help=(
            "Input path. "
            "If --run-both is set, this should be the root containing fire_frames/ and isochrones/. "
            "Otherwise this should be a single modality directory with per-sequence folders."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output root directory. Embeddings are written under output_dir/embeddings/...",
    )
    parser.add_argument(
        "--input-type",
        choices=["fire_frames", "isochrones"],
        default="",
        help="Single modality folder name to encode (required unless --run-both is set).",
    )
    parser.add_argument(
        "--run-both",
        action="store_true",
        help="Process both fire_frames and isochrones in one run.",
    )
    parser.add_argument(
        "--model-name",
        default="facebook/dinov2-large",
        help="Hugging Face model id for DINO/DINOv2 encoder.",
    )
    parser.add_argument(
        "--expected-timesteps",
        type=int,
        default=6,
        help="Expected number of timesteps per fire. Defaults to 6.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=400,
        help="Input image size metadata value.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any fire is missing expected timesteps [0..T-1].",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Torch device selection.",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Computation dtype for model forward pass.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of frames per model batch.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--run-timestamp",
        default="",
        help=(
            "Optional run timestamp folder name. "
            "If omitted, auto-generates UTC ISO-like format YYYY-MM-DDTHH-MM-SSZ."
        ),
    )
    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="Sanity mode: skip model loading/inference and write mock embeddings.",
    )
    parser.add_argument(
        "--mock-embedding-dim",
        type=int,
        default=1024,
        help="Embedding dimension used with --skip-embedding.",
    )
    return parser.parse_args()


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
            raise ValueError(
                f"Duplicate frame index {frame_index} in sequence {sequence_dir.name}"
            )
        indexed[frame_index] = file_path
    return indexed


def collect_sequences(input_dir: Path) -> list[SequenceFiles]:
    if not input_dir.exists():
        raise FileNotFoundError(f"input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"not a directory: {input_dir}")

    sequences: list[SequenceFiles] = []
    for sequence_dir in sorted(input_dir.iterdir()):
        if not sequence_dir.is_dir():
            continue

        fire_frames = collect_indexed_images(sequence_dir)
        if not fire_frames:
            continue

        sequences.append(
            SequenceFiles(
                fire_id=sequence_dir.name,
                fire_frames=fire_frames,
            )
        )
    return sequences


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


def resolve_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = mapping[dtype_name]
    if device.type == "cpu" and dtype != torch.float32:
        return torch.float32
    return dtype


def load_mask_as_rgb(image_path: Path) -> Image.Image:
    with Image.open(image_path) as image:
        gray = image.convert("L")
        rgb = Image.merge("RGB", (gray, gray, gray))
    return rgb


def select_fire_indices(
    fire_frames: dict[int, Path], expected_timesteps: int, strict: bool
) -> list[int]:
    expected = list(range(expected_timesteps))
    available = set(fire_frames.keys())
    selected = [idx for idx in expected if idx in available]
    if strict and len(selected) != expected_timesteps:
        missing = [idx for idx in expected if idx not in available]
        raise ValueError(f"Missing timesteps: {missing}")
    return selected


def encode_batch(
    batch_images: list[Image.Image],
    processor: Any,
    model: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> np.ndarray:
    inputs = processor(images=batch_images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        if device.type == "cpu":
            outputs = model(**inputs)
        else:
            with torch.autocast(device_type=device.type, dtype=dtype):
                outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
    return embeddings


def process_fire(
    fire: SequenceFiles,
    model_output_dir: Path,
    processor: Any | None,
    model: Any | None,
    device: torch.device,
    dtype: torch.dtype,
    expected_timesteps: int,
    strict: bool,
    batch_size: int,
    overwrite: bool,
    skip_embedding: bool,
    mock_embedding_dim: int,
) -> dict[str, int | str]:
    fire_indices = select_fire_indices(
        fire_frames=fire.fire_frames, expected_timesteps=expected_timesteps, strict=strict
    )
    if not fire_indices:
        raise ValueError("No usable timesteps found.")

    ordered_paths = [fire.fire_frames[idx] for idx in fire_indices]
    model_output_dir.mkdir(parents=True, exist_ok=True)

    if skip_embedding:
        stacked_embeddings = np.zeros(
            (len(ordered_paths), mock_embedding_dim), dtype=np.float32
        )
    else:
        if processor is None or model is None:
            raise RuntimeError("processor/model are required when skip_embedding is False")
        all_embeddings: list[np.ndarray] = []
        for start in range(0, len(ordered_paths), batch_size):
            batch_paths = ordered_paths[start : start + batch_size]
            batch_images = [load_mask_as_rgb(path) for path in batch_paths]
            batch_embeddings = encode_batch(
                batch_images=batch_images,
                processor=processor,
                model=model,
                device=device,
                dtype=dtype,
            )
            all_embeddings.append(batch_embeddings)
        stacked_embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32, copy=False)
    per_fire_path = model_output_dir / f"{fire.fire_id}.npy"
    if not per_fire_path.exists() or overwrite:
        np.save(per_fire_path, stacked_embeddings)

    return {
        "sequence_id": fire.fire_id,
        "num_timesteps_saved": len(fire_indices),
        "embedding_dim": int(stacked_embeddings.shape[1]),
        "sequence_path": str(per_fire_path),
    }


def model_slug(model_name: str) -> str:
    return model_name.replace("/", "__")


def make_run_timestamp(user_value: str) -> str:
    if user_value:
        return user_value
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def encoder_name(model_name: str) -> str:
    if model_name == "facebook/dinov2-large":
        return "dino_v2_large"
    return model_slug(model_name)


def upsert_model_meta(meta: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    models = meta.get("models", [])
    if not isinstance(models, list):
        models = []
    updated = False
    for index, model_entry in enumerate(models):
        if (
            isinstance(model_entry, dict)
            and model_entry.get("model_slug") == entry["model_slug"]
        ):
            models[index] = entry
            updated = True
            break
    if not updated:
        models.append(entry)
    meta["models"] = models
    return meta


def main() -> None:
    args = parse_args()
    if args.run_both:
        modality_inputs = [(modality, args.input_dir / modality) for modality in MODALITIES]
    else:
        if not args.input_type:
            raise ValueError("--input-type is required unless --run-both is set")
        modality_inputs = [(args.input_type, args.input_dir)]

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    run_timestamp = make_run_timestamp(args.run_timestamp)
    processor: Any | None = None
    model: Any | None = None
    if args.skip_embedding:
        print(f"[info] skip-embedding mode enabled (mock_dim={args.mock_embedding_dim})")
    else:
        print(f"[info] loading model {args.model_name} on device={device} dtype={dtype}")
        processor = AutoImageProcessor.from_pretrained(args.model_name)
        model = AutoModel.from_pretrained(args.model_name).to(device)
        model.eval()
    timestamp_dir = args.output_dir / "embeddings" / run_timestamp

    modality_summaries: dict[str, list[dict[str, int | str]]] = {}
    for input_type, modality_input_dir in modality_inputs:
        fires = collect_sequences(modality_input_dir)
        if not fires:
            raise RuntimeError(
                f"No sequence folders with image files were found for {input_type}: {modality_input_dir}"
            )

        output_base = (
            args.output_dir
            / "embeddings"
            / run_timestamp
            / input_type
            / model_slug(args.model_name)
        )

        summaries: list[dict[str, int | str]] = []
        for fire in fires:
            summary = process_fire(
                fire=fire,
                model_output_dir=output_base,
                processor=processor,
                model=model,
                device=device,
                dtype=dtype,
                expected_timesteps=args.expected_timesteps,
                strict=args.strict,
                batch_size=args.batch_size,
                overwrite=args.overwrite,
                skip_embedding=args.skip_embedding,
                mock_embedding_dim=args.mock_embedding_dim,
            )
            summaries.append(summary)
            print(
                f"[ok] {input_type}/{summary['sequence_id']}: "
                f"saved {summary['num_timesteps_saved']} timestep embeddings "
                f"(E={summary['embedding_dim']})"
            )
        modality_summaries[input_type] = summaries

        manifest_path = output_base / "manifest.json"
        manifest = {
            "input_type": input_type,
            "input_dir": str(modality_input_dir),
            "sequences": [
                {
                    "sequence_id": summary["sequence_id"],
                    "sequence_path": summary["sequence_path"],
                    "num_frames": summary["num_timesteps_saved"],
                }
                for summary in summaries
            ],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[done] wrote manifest: {manifest_path}")

    embedding_dim = (
        args.mock_embedding_dim
        if args.skip_embedding or not modality_summaries
        else int(next(iter(modality_summaries.values()))[0]["embedding_dim"])
    )
    timestamp_dir.mkdir(parents=True, exist_ok=True)
    meta_path = timestamp_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(meta, dict):
            meta = {}
    else:
        meta = {}

    meta["run_timestamp"] = run_timestamp
    meta["build_date"] = run_timestamp[:10]

    model_entry = {
        "encoder": encoder_name(args.model_name),
        "model_name": args.model_name,
        "model_slug": model_slug(args.model_name),
        "input_size": args.input_size,
        "num_frames": args.expected_timesteps,
        "embedding_dim": embedding_dim,
        "modalities": list(modality_summaries.keys()),
        "device": str(device),
        "skip_embedding": args.skip_embedding,
    }
    meta = upsert_model_meta(meta, model_entry)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[done] wrote meta: {meta_path}")


if __name__ == "__main__":
    main()
