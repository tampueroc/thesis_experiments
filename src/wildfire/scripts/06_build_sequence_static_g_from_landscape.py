from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wildfire.data.real_data import NODATA_VALUE, static_vector_from_landscape_patch


FEATURE_NAMES = [
    "fuels_mode",
    "arqueo_presence_ratio",
    "cbd_mean",
    "cbh_mean",
    "elevation_mean",
    "flora_presence_ratio",
    "paleo_presence_ratio",
    "unknown_a8_presence_ratio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build per-sequence static g vectors from indices.json and "
            "pivoted landscape_channels_chw.npy (no-data value: -1)."
        )
    )
    parser.add_argument(
        "--landscape-dir",
        type=Path,
        required=True,
        help="Directory containing indices.json.",
    )
    parser.add_argument(
        "--landscape-chw-path",
        type=Path,
        required=True,
        help="Path to landscape_channels_chw.npy from pivot script.",
    )
    parser.add_argument(
        "--output-values-path",
        type=Path,
        default=None,
        help="Output .npy path for sequence static g matrix. Default: <landscape-dir>/sequence_static_g_values.npy",
    )
    parser.add_argument(
        "--output-keys-path",
        type=Path,
        default=None,
        help="Output json path for alias->row mapping. Default: <landscape-dir>/sequence_static_g_keys.json",
    )
    parser.add_argument(
        "--output-meta-path",
        type=Path,
        default=None,
        help="Output metadata json path. Default: <landscape-dir>/sequence_static_g_meta.json",
    )
    return parser.parse_args()


def sequence_aliases(sequence_num: int) -> list[str]:
    return [
        str(sequence_num),
        f"sequence_{sequence_num:03d}",
        f"sequence_{sequence_num:04d}",
        f"sequence_{sequence_num:05d}",
    ]


def main() -> None:
    args = parse_args()
    output_values_path = args.output_values_path or (args.landscape_dir / "sequence_static_g_values.npy")
    output_keys_path = args.output_keys_path or (args.landscape_dir / "sequence_static_g_keys.json")
    output_meta_path = args.output_meta_path or (args.landscape_dir / "sequence_static_g_meta.json")

    indices_path = args.landscape_dir / "indices.json"
    if not indices_path.exists():
        raise FileNotFoundError(f"indices.json not found: {indices_path}")
    if not args.landscape_chw_path.exists():
        raise FileNotFoundError(f"landscape CHW array not found: {args.landscape_chw_path}")

    indices = json.loads(indices_path.read_text(encoding="utf-8"))
    landscape_chw = np.asarray(np.load(args.landscape_chw_path), dtype=np.float32)
    if landscape_chw.ndim != 3:
        raise ValueError(f"expected CHW array with ndim=3, got shape={landscape_chw.shape}")

    vectors: list[np.ndarray] = []
    alias_to_row: dict[str, int] = {}
    sequence_index: list[dict[str, object]] = []
    for row_idx, key in enumerate(sorted(indices.keys(), key=lambda x: int(x))):
        sequence_num = int(key)
        bbox = indices[key]
        g = static_vector_from_landscape_patch(landscape_chw, bbox)
        vectors.append(g)
        for alias in sequence_aliases(sequence_num):
            alias_to_row[alias] = row_idx
        sequence_index.append(
            {
                "sequence_num": sequence_num,
                "row_idx": row_idx,
                "aliases": sequence_aliases(sequence_num),
                "bbox": [int(v) for v in bbox],
            }
        )

    g_matrix = np.stack(vectors).astype(np.float32) if vectors else np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)

    output_values_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_values_path, g_matrix)
    output_keys_path.write_text(json.dumps(alias_to_row, indent=2), encoding="utf-8")
    meta = {
        "source_landscape_chw_path": str(args.landscape_chw_path),
        "source_indices_path": str(indices_path),
        "nodata_value": NODATA_VALUE,
        "num_sequences": int(g_matrix.shape[0]),
        "g_dim": len(FEATURE_NAMES),
        "g_feature_names": FEATURE_NAMES,
        "sequences": sequence_index,
    }
    output_meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[ok] output_values={output_values_path}")
    print(f"[ok] output_keys={output_keys_path}")
    print(f"[ok] output_meta={output_meta_path}")
    print(f"[ok] sequences={len(sequence_index)}")
    print(f"[ok] g_dim={len(FEATURE_NAMES)}")


if __name__ == "__main__":
    main()
