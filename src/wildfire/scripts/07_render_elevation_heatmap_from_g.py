from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render full elevation heatmap from per-sequence static g vectors "
            "and indices.json."
        )
    )
    parser.add_argument("--landscape-dir", type=Path, required=True, help="Directory containing indices.json.")
    parser.add_argument(
        "--g-values-path",
        type=Path,
        required=True,
        help="Path to sequence_static_g_values.npy.",
    )
    parser.add_argument(
        "--g-keys-path",
        type=Path,
        required=True,
        help="Path to sequence_static_g_keys.json (alias->row index).",
    )
    parser.add_argument(
        "--g-meta-path",
        type=Path,
        default=None,
        help="Optional sequence_static_g_meta.json to resolve feature index by name.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("landscape_static_analss"),
        help="Base output directory for timestamped run folder.",
    )
    parser.add_argument(
        "--run-timestamp",
        type=str,
        default="",
        help="Optional run folder timestamp; default uses current UTC.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1173,
        help="Landscape raster height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1406,
        help="Landscape raster width.",
    )
    parser.add_argument(
        "--elevation-feature-name",
        type=str,
        default="elevation_mean",
        help="Feature name in g metadata used for elevation.",
    )
    parser.add_argument(
        "--default-elevation-index",
        type=int,
        default=4,
        help="Fallback elevation feature index if meta file is missing.",
    )
    return parser.parse_args()


def resolve_run_dir(base_dir: Path, requested: str) -> Path:
    ts = requested or datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    out = base_dir / ts
    out.mkdir(parents=True, exist_ok=True)
    return out


def resolve_elevation_index(meta_path: Path | None, feature_name: str, fallback: int) -> int:
    if meta_path is None or not meta_path.exists():
        return fallback
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    names = meta.get("g_feature_names", [])
    if feature_name in names:
        return int(names.index(feature_name))
    return fallback


def sequence_aliases(sequence_num: int) -> list[str]:
    return [
        str(sequence_num),
        f"sequence_{sequence_num:03d}",
        f"sequence_{sequence_num:04d}",
        f"sequence_{sequence_num:05d}",
    ]


def lookup_row_idx(alias_to_row: dict[str, int], sequence_num: int) -> int | None:
    for alias in sequence_aliases(sequence_num):
        if alias in alias_to_row:
            return int(alias_to_row[alias])
    return None


def main() -> None:
    args = parse_args()
    indices_path = args.landscape_dir / "indices.json"
    if not indices_path.exists():
        raise FileNotFoundError(f"indices.json not found: {indices_path}")
    if not args.g_values_path.exists():
        raise FileNotFoundError(f"g values not found: {args.g_values_path}")
    if not args.g_keys_path.exists():
        raise FileNotFoundError(f"g keys not found: {args.g_keys_path}")

    run_dir = resolve_run_dir(args.output_dir, args.run_timestamp.strip())
    out_map_path = run_dir / "elevation_mean_map.npy"
    out_png_path = run_dir / "elevation_mean_heatmap.png"
    out_meta_path = run_dir / "elevation_heatmap_meta.json"

    indices = json.loads(indices_path.read_text(encoding="utf-8"))
    alias_to_row = {str(k): int(v) for k, v in json.loads(args.g_keys_path.read_text(encoding="utf-8")).items()}
    g_values = np.asarray(np.load(args.g_values_path), dtype=np.float32)
    elevation_idx = resolve_elevation_index(args.g_meta_path, args.elevation_feature_name, args.default_elevation_index)
    if elevation_idx < 0 or elevation_idx >= g_values.shape[1]:
        raise ValueError(f"invalid elevation index={elevation_idx} for g shape={g_values.shape}")

    sum_map = np.zeros((args.height, args.width), dtype=np.float64)
    count_map = np.zeros((args.height, args.width), dtype=np.int32)

    assigned = 0
    missing = 0
    for key in sorted(indices.keys(), key=lambda x: int(x)):
        seq_num = int(key)
        row_idx = lookup_row_idx(alias_to_row, seq_num)
        if row_idx is None or row_idx >= g_values.shape[0]:
            missing += 1
            continue
        row_start, row_end, col_start, col_end = [int(v) for v in indices[key]]
        elevation = float(g_values[row_idx, elevation_idx])
        sum_map[row_start:row_end, col_start:col_end] += elevation
        count_map[row_start:row_end, col_start:col_end] += 1
        assigned += 1

    heatmap = np.full((args.height, args.width), np.nan, dtype=np.float32)
    covered = count_map > 0
    heatmap[covered] = (sum_map[covered] / count_map[covered]).astype(np.float32)
    np.save(out_map_path, heatmap)

    sns.set_theme(style="white", context="talk")
    plt.figure(figsize=(12, 8))
    mask = np.isnan(heatmap)
    sns.heatmap(
        heatmap,
        mask=mask,
        cmap="terrain",
        cbar_kws={"label": "Elevation mean"},
        xticklabels=False,
        yticklabels=False,
    )
    plt.title("Landscape Elevation Map Reconstructed from Sequence g Vectors")
    plt.tight_layout()
    plt.savefig(out_png_path, dpi=220)
    plt.close()

    out_meta = {
        "indices_path": str(indices_path),
        "g_values_path": str(args.g_values_path),
        "g_keys_path": str(args.g_keys_path),
        "elevation_feature_index": elevation_idx,
        "elevation_feature_name": args.elevation_feature_name,
        "height": args.height,
        "width": args.width,
        "num_sequences_indexed": len(indices),
        "num_sequences_assigned": assigned,
        "num_sequences_missing_g": missing,
        "num_pixels_covered": int(covered.sum()),
        "num_pixels_total": int(heatmap.size),
        "coverage_ratio": float(covered.sum() / heatmap.size),
    }
    out_meta_path.write_text(json.dumps(out_meta, indent=2), encoding="utf-8")

    print(f"[ok] run_dir={run_dir}")
    print(f"[ok] elevation_map={out_map_path}")
    print(f"[ok] heatmap_png={out_png_path}")
    print(f"[ok] meta={out_meta_path}")
    print(f"[ok] sequences_assigned={assigned}")
    print(f"[ok] sequences_missing={missing}")


if __name__ == "__main__":
    main()
