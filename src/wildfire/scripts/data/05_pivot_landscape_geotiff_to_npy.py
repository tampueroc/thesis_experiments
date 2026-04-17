from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import tifffile


RAW_NODATA_VALUE = -9999.0
NODATA_VALUE = -1.0

STATIC_CHANNEL_MEANINGS: list[dict[str, str]] = [
    {"channel": "a1", "feature": "fuels", "meaning": "fuel model code", "expected_type": "categorical"},
    {"channel": "a2", "feature": "arqueo", "meaning": "archaeo/land-use class", "expected_type": "categorical"},
    {"channel": "a3", "feature": "cbd", "meaning": "canopy bulk density", "expected_type": "continuous"},
    {"channel": "a4", "feature": "cbh", "meaning": "canopy base height", "expected_type": "continuous"},
    {"channel": "a5", "feature": "elevation", "meaning": "elevation", "expected_type": "continuous"},
    {"channel": "a6", "feature": "flora", "meaning": "flora/vegetation class", "expected_type": "categorical"},
    {"channel": "a7", "feature": "paleo", "meaning": "paleo/soil-geology class", "expected_type": "categorical"},
    {"channel": "a8", "feature": "unknown_a8", "meaning": "loaded but not identified in snippet", "expected_type": "unknown"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pivot Input_Geotiff.tif into NPY artifacts: channel-first cube, "
            "pixel feature matrix, per-channel files, and metadata."
        )
    )
    parser.add_argument("--landscape-dir", type=Path, required=True, help="Path containing Input_Geotiff.tif.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("landscape_npy"),
        help="Base output directory for timestamped runs.",
    )
    parser.add_argument(
        "--run-timestamp",
        type=str,
        default="",
        help="Optional run folder name; default is current UTC ISO-like timestamp.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float64"],
        default="float32",
        help="Output dtype for NPY artifacts.",
    )
    return parser.parse_args()


def resolve_run_dir(base_output_dir: Path, run_timestamp: str) -> Path:
    ts = run_timestamp or datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    run_dir = base_output_dir / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def to_channel_first(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 3:
        raise ValueError(f"expected 3D geotiff tensor, got shape={arr.shape}")
    if arr.shape[0] <= 32:
        return arr
    if arr.shape[-1] <= 32:
        return arr.transpose(2, 0, 1)
    raise ValueError(f"cannot infer channel axis from shape={arr.shape}")


def normalize_nodata(arr: np.ndarray) -> np.ndarray:
    return np.where(arr == RAW_NODATA_VALUE, NODATA_VALUE, arr)


def build_meta(channels_chw: np.ndarray, dtype_name: str) -> dict[str, object]:
    channels, height, width = channels_chw.shape
    channel_meta: list[dict[str, object]] = []
    for i in range(channels):
        info = STATIC_CHANNEL_MEANINGS[i] if i < len(STATIC_CHANNEL_MEANINGS) else {
            "channel": f"a{i + 1}",
            "feature": f"a{i + 1}",
            "meaning": "",
            "expected_type": "unknown",
        }
        vals = channels_chw[i].reshape(-1)
        nodata_count = int((vals == NODATA_VALUE).sum())
        valid = vals[vals != NODATA_VALUE]
        channel_meta.append(
            {
                **info,
                "n_total": int(vals.size),
                "n_nodata": nodata_count,
                "nodata_ratio": float(nodata_count / vals.size),
                "n_valid": int(valid.size),
                "n_unique_valid": int(np.unique(valid).size) if valid.size else 0,
                "min_valid": float(valid.min()) if valid.size else None,
                "max_valid": float(valid.max()) if valid.size else None,
            }
        )
    return {
        "source": "Input_Geotiff.tif",
        "raw_nodata_value": RAW_NODATA_VALUE,
        "nodata_value": NODATA_VALUE,
        "dtype": dtype_name,
        "shape_chw": [int(channels), int(height), int(width)],
        "shape_hwc": [int(height), int(width), int(channels)],
        "shape_pixels_by_channels": [int(height * width), int(channels)],
        "channels": channel_meta,
        "build_date": datetime.now(UTC).date().isoformat(),
    }


def main() -> None:
    args = parse_args()
    geotiff_path = args.landscape_dir / "Input_Geotiff.tif"
    if not geotiff_path.exists():
        raise FileNotFoundError(f"Input_Geotiff.tif not found: {geotiff_path}")

    run_dir = resolve_run_dir(args.output_dir, args.run_timestamp.strip())
    channels_dir = run_dir / "channels"
    channels_dir.mkdir(parents=True, exist_ok=True)

    raw = np.asarray(tifffile.imread(geotiff_path))
    channels_chw = to_channel_first(raw)
    channels_chw = normalize_nodata(channels_chw).astype(args.dtype, copy=False)

    channels_hwc = np.moveaxis(channels_chw, 0, -1)
    pixels_by_channels = channels_hwc.reshape(-1, channels_hwc.shape[-1])

    np.save(run_dir / "landscape_channels_chw.npy", channels_chw)
    np.save(run_dir / "landscape_channels_hwc.npy", channels_hwc)
    np.save(run_dir / "landscape_pixels_by_channels.npy", pixels_by_channels)
    np.save(run_dir / "landscape_nodata_mask_chw.npy", channels_chw == NODATA_VALUE)

    for i in range(channels_chw.shape[0]):
        info = STATIC_CHANNEL_MEANINGS[i] if i < len(STATIC_CHANNEL_MEANINGS) else {"feature": f"a{i + 1}"}
        np.save(channels_dir / f"a{i + 1}_{info['feature']}.npy", channels_chw[i])

    meta = build_meta(channels_chw, dtype_name=args.dtype)
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (run_dir / "static_channel_meanings.json").write_text(
        json.dumps(STATIC_CHANNEL_MEANINGS, indent=2),
        encoding="utf-8",
    )

    print(f"[ok] geotiff={geotiff_path}")
    print(f"[ok] run_dir={run_dir}")
    print(f"[ok] saved={run_dir / 'landscape_channels_chw.npy'}")
    print(f"[ok] saved={run_dir / 'landscape_pixels_by_channels.npy'}")
    print(f"[ok] saved={run_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
