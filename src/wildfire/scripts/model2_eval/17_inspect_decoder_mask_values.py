from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from wildfire.data.decoder_dataset import load_mask_tensor
from wildfire.data.real_data import choose_timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect decoder mask value ranges on the real dataset."
    )
    parser.add_argument(
        "--embeddings-root",
        type=Path,
        default=Path("/home/tampuero/data/thesis_data/embeddings"),
        help="Root folder containing timestamped embeddings outputs.",
    )
    parser.add_argument(
        "--timestamp",
        default="",
        help="Specific embedding timestamp folder. If omitted, uses latest lexicographic timestamp.",
    )
    parser.add_argument(
        "--input-type",
        choices=["fire_frames", "isochrones"],
        default="fire_frames",
        help="Embedding modality to inspect.",
    )
    parser.add_argument(
        "--embeddings-model-slug",
        default="facebook__dinov2-small",
        help="Embedding model slug under modality folder.",
    )
    parser.add_argument(
        "--image-channels",
        type=int,
        default=1,
        help="How many channels to request from the decoder dataset loader.",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=8,
        help="Maximum number of sequences to inspect.",
    )
    parser.add_argument(
        "--max-frames-per-sequence",
        type=int,
        default=6,
        help="Maximum number of frames to inspect per sequence.",
    )
    parser.add_argument(
        "--show-unique-limit",
        type=int,
        default=16,
        help="If the number of unique values is at most this limit, print them explicitly.",
    )
    return parser.parse_args()


def summarize_mask(mask: np.ndarray, unique_limit: int) -> dict[str, Any]:
    unique_values = np.unique(mask)
    rounded_unique = np.unique(np.round(unique_values, decimals=6))
    summary: dict[str, Any] = {
        "dtype": str(mask.dtype),
        "min": float(mask.min()),
        "max": float(mask.max()),
        "num_unique_values": int(unique_values.size),
        "is_binary_01": bool(np.all(np.isin(rounded_unique, np.array([0.0, 1.0], dtype=np.float32)))),
    }
    if int(unique_values.size) <= unique_limit:
        summary["unique_values"] = [float(v) for v in unique_values.tolist()]
    return summary


def main() -> int:
    args = parse_args()
    timestamp = choose_timestamp(args.embeddings_root, args.timestamp)
    model_dir = args.embeddings_root / timestamp / args.input_type / args.embeddings_model_slug
    manifest_path = model_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sequences = manifest.get("sequences", [])
    if not sequences:
        raise RuntimeError(f"no sequences in manifest: {manifest_path}")
    input_dir_raw = manifest.get("input_dir")
    if not isinstance(input_dir_raw, str) or not input_dir_raw:
        raise ValueError(f"manifest missing input_dir: {manifest_path}")
    input_dir = Path(input_dir_raw)

    selected_sequences = sequences[: args.max_sequences] if args.max_sequences > 0 else sequences
    aggregate_unique_values: set[float] = set()
    aggregate_binary = True

    print(f"[info] timestamp={timestamp}")
    print(f"[info] model_dir={model_dir}")
    print(f"[info] input_dir={input_dir}")
    print(f"[info] image_channels={args.image_channels}")

    for seq in selected_sequences:
        sequence_id = str(seq["sequence_id"])
        sequence_dir = input_dir / sequence_id
        if not sequence_dir.exists():
            print(f"[warn] missing sequence_dir={sequence_dir}")
            continue
        frame_paths = sorted(p for p in sequence_dir.iterdir() if p.is_file())[: args.max_frames_per_sequence]
        if not frame_paths:
            print(f"[warn] no frames found for {sequence_id}")
            continue
        print(f"\n[sequence] {sequence_id}")
        for frame_path in frame_paths:
            mask = load_mask_tensor(frame_path, image_channels=args.image_channels)
            summary = summarize_mask(mask, unique_limit=args.show_unique_limit)
            aggregate_binary = aggregate_binary and bool(summary["is_binary_01"])
            if "unique_values" in summary:
                aggregate_unique_values.update(float(v) for v in summary["unique_values"])
            print(
                json.dumps(
                    {
                        "frame": frame_path.name,
                        **summary,
                    },
                    sort_keys=True,
                )
            )

    if aggregate_unique_values:
        sorted_unique_values = sorted(aggregate_unique_values)
    else:
        sorted_unique_values = []
    print("\n[summary]")
    print(json.dumps({"binary_01_for_all_inspected_masks": aggregate_binary}, sort_keys=True))
    if len(sorted_unique_values) <= args.show_unique_limit:
        print(json.dumps({"aggregate_unique_values": sorted_unique_values}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
