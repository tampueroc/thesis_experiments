from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import torch
from wildfire.data.dataset import WildfireSequenceDataset
from wildfire.data.real_data import build_sources, choose_timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test WildfireSequenceDataset with real thesis_data files."
    )
    parser.add_argument(
        "--embeddings-root",
        type=Path,
        default=Path("/home/tampuero/data/thesis_data/embeddings"),
        help="Root folder containing timestamped embeddings outputs.",
    )
    parser.add_argument(
        "--landscape-dir",
        type=Path,
        default=Path("/home/tampuero/data/thesis_data/landscape"),
        help="Landscape folder containing WeatherHistory.csv, Weathers/, and indices.json.",
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
        help="Embedding modality to test.",
    )
    parser.add_argument(
        "--model-slug",
        default="facebook__dinov2-small",
        help="Embedding model slug under modality folder.",
    )
    parser.add_argument(
        "--history",
        type=int,
        default=5,
        help="Dataset history window.",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=256,
        help="Limit number of sequences loaded for smoke test.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = choose_timestamp(args.embeddings_root, args.timestamp)
    model_dir = args.embeddings_root / timestamp / args.input_type / args.model_slug
    if not model_dir.exists():
        raise FileNotFoundError(f"embedding model directory not found: {model_dir}")

    z_by_fire, w_by_fire, g_by_fire = build_sources(
        model_dir=model_dir,
        landscape_dir=args.landscape_dir,
        max_sequences=args.max_sequences,
    )
    dataset = WildfireSequenceDataset(
        embeddings_source=z_by_fire,
        weather_source=w_by_fire,
        static_source=g_by_fire,
        history=args.history,
    )
    if len(dataset) == 0:
        raise RuntimeError("dataset has zero samples after history windowing")

    sample = dataset[0]
    z_in = cast(torch.Tensor, sample["z_in"])
    z_target = cast(torch.Tensor, sample["z_target"])
    w_in = cast(torch.Tensor, sample["w_in"])
    g = cast(torch.Tensor, sample["g"])
    print(f"[ok] timestamp={timestamp}")
    print(f"[ok] model_dir={model_dir}")
    print(f"[ok] sequences_loaded={len(z_by_fire)}")
    print(f"[ok] num_samples={len(dataset)}")
    print(f"[ok] fire_id={sample['fire_id']}")
    print(f"[ok] z_in shape={tuple(z_in.shape)}")
    print(f"[ok] z_target shape={tuple(z_target.shape)}")
    print(f"[ok] w_in shape={tuple(w_in.shape)}")
    print(f"[ok] g shape={tuple(g.shape)}")


if __name__ == "__main__":
    main()
