from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


NODATA_VALUE = -1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render pixel-level terrain elevation map from landscape_channels_chw.npy."
    )
    parser.add_argument("--landscape-chw-path", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("landscape_static_analss"),
        help="Base output directory for timestamped run folder.",
    )
    parser.add_argument("--run-timestamp", type=str, default="")
    parser.add_argument(
        "--elevation-channel-index",
        type=int,
        default=4,
        help="0-based channel index for elevation (default 4 == a5).",
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="terrain",
        help="Matplotlib colormap for elevation rendering.",
    )
    return parser.parse_args()


def resolve_run_dir(base: Path, requested: str) -> Path:
    ts = requested or datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    out = base / ts
    out.mkdir(parents=True, exist_ok=True)
    return out


def main() -> None:
    args = parse_args()
    if not args.landscape_chw_path.exists():
        raise FileNotFoundError(f"landscape CHW path not found: {args.landscape_chw_path}")

    chw = np.asarray(np.load(args.landscape_chw_path), dtype=np.float32)
    if chw.ndim != 3:
        raise ValueError(f"expected CHW array, got shape={chw.shape}")
    if not (0 <= args.elevation_channel_index < chw.shape[0]):
        raise ValueError(
            f"elevation channel index={args.elevation_channel_index} out of bounds for channels={chw.shape[0]}"
        )

    run_dir = resolve_run_dir(args.output_dir, args.run_timestamp.strip())
    out_map_path = run_dir / "elevation_pixel_map.npy"
    out_png_path = run_dir / "elevation_pixel_heatmap.png"
    out_meta_path = run_dir / "elevation_pixel_heatmap_meta.json"

    elevation = chw[args.elevation_channel_index]
    mask = elevation == NODATA_VALUE
    elevation_for_plot = np.where(mask, np.nan, elevation)
    np.save(out_map_path, elevation_for_plot.astype(np.float32))

    sns.set_theme(style="white", context="talk")
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        elevation_for_plot,
        mask=np.isnan(elevation_for_plot),
        cmap=args.colormap,
        cbar_kws={"label": "Elevation"},
        xticklabels=False,
        yticklabels=False,
    )
    plt.title("Pixel-Level Terrain Elevation (from GeoTIFF a5)")
    plt.tight_layout()
    plt.savefig(out_png_path, dpi=240)
    plt.close()

    valid = elevation_for_plot[~np.isnan(elevation_for_plot)]
    meta = {
        "source": str(args.landscape_chw_path),
        "elevation_channel_index": args.elevation_channel_index,
        "nodata_value": NODATA_VALUE,
        "shape": [int(elevation.shape[0]), int(elevation.shape[1])],
        "n_pixels": int(elevation.size),
        "n_nodata": int(mask.sum()),
        "valid_ratio": float((~mask).sum() / elevation.size),
        "min_valid": float(valid.min()) if valid.size else None,
        "max_valid": float(valid.max()) if valid.size else None,
        "mean_valid": float(valid.mean()) if valid.size else None,
        "colormap": args.colormap,
    }
    out_meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[ok] run_dir={run_dir}")
    print(f"[ok] elevation_map={out_map_path}")
    print(f"[ok] heatmap_png={out_png_path}")
    print(f"[ok] meta={out_meta_path}")


if __name__ == "__main__":
    main()
