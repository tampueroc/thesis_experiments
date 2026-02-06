from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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

FEATURE_MEANINGS: dict[str, dict[str, str]] = {
    "row_start": {"meaning": "bbox row start", "expected_type": "continuous"},
    "row_end": {"meaning": "bbox row end", "expected_type": "continuous"},
    "col_start": {"meaning": "bbox col start", "expected_type": "continuous"},
    "col_end": {"meaning": "bbox col end", "expected_type": "continuous"},
    "height": {"meaning": "bbox height", "expected_type": "continuous"},
    "width": {"meaning": "bbox width", "expected_type": "continuous"},
    "center_row": {"meaning": "bbox center row", "expected_type": "continuous"},
    "center_col": {"meaning": "bbox center col", "expected_type": "continuous"},
    "touches_top": {"meaning": "bbox touches top boundary", "expected_type": "categorical"},
    "touches_bottom": {"meaning": "bbox touches bottom boundary", "expected_type": "categorical"},
    "touches_left": {"meaning": "bbox touches left boundary", "expected_type": "categorical"},
    "touches_right": {"meaning": "bbox touches right boundary", "expected_type": "categorical"},
    "fuels": {"meaning": "fuel model code", "expected_type": "categorical"},
    "arqueo": {"meaning": "archaeo/land-use class", "expected_type": "categorical"},
    "cbd": {"meaning": "canopy bulk density", "expected_type": "continuous"},
    "cbh": {"meaning": "canopy base height", "expected_type": "continuous"},
    "elevation": {"meaning": "elevation", "expected_type": "continuous"},
    "flora": {"meaning": "flora/vegetation class", "expected_type": "categorical"},
    "paleo": {"meaning": "paleo/soil-geology class", "expected_type": "categorical"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze static landscape features used to derive g and generate "
            "artifacts (summary tables + seaborn plots)."
        )
    )
    parser.add_argument(
        "--landscape-dir",
        type=Path,
        required=True,
        help="Path containing indices.json (for example /.../thesis_data/landscape).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("landscape_static_analss"),
        help="Output directory for generated artifacts.",
    )
    parser.add_argument(
        "--top-categorical-values",
        type=int,
        default=20,
        help="Max number of top values to keep per detected categorical feature.",
    )
    return parser.parse_args()


def build_feature_frame(indices_path: Path) -> pd.DataFrame:
    raw = json.loads(indices_path.read_text(encoding="utf-8"))
    records: list[dict[str, float | int]] = []
    for fire_id_str, bbox in raw.items():
        row_start, row_end, col_start, col_end = [int(v) for v in bbox]
        height = row_end - row_start
        width = col_end - col_start
        center_row = row_start + height / 2.0
        center_col = col_start + width / 2.0
        records.append(
            {
                "fire_id": int(fire_id_str),
                "row_start": row_start,
                "row_end": row_end,
                "col_start": col_start,
                "col_end": col_end,
                "height": height,
                "width": width,
                "center_row": center_row,
                "center_col": center_col,
                "touches_top": int(row_start == 0),
                "touches_bottom": int(row_end >= 1173),
                "touches_left": int(col_start == 0),
                "touches_right": int(col_end >= 1406),
            }
        )
    return pd.DataFrame.from_records(records).sort_values("fire_id").reset_index(drop=True)


def detect_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    n = len(df)
    for col in df.columns:
        if col == "fire_id":
            continue
        s = df[col]
        metadata = FEATURE_MEANINGS.get(col, {})
        expected_type = metadata.get("expected_type", "")
        unique_n = int(s.nunique(dropna=False))
        is_integer_like = bool((s.dropna() % 1 == 0).all())
        unique_ratio = unique_n / n if n else 0.0
        is_binary = unique_n <= 2
        likely_categorical = is_binary or (is_integer_like and unique_n <= 20 and unique_ratio <= 0.05)
        if expected_type == "categorical":
            likely_categorical = True
        if expected_type == "continuous":
            likely_categorical = False
        rows.append(
            {
                "feature": col,
                "meaning": metadata.get("meaning", ""),
                "expected_type": expected_type,
                "n": n,
                "dtype": str(s.dtype),
                "min": float(s.min()),
                "max": float(s.max()),
                "mean": float(s.mean()),
                "std": float(s.std()),
                "n_unique": unique_n,
                "unique_ratio": unique_ratio,
                "is_integer_like": is_integer_like,
                "likely_categorical": likely_categorical,
            }
        )
    return pd.DataFrame(rows).sort_values("feature").reset_index(drop=True)


def save_range_plot(df: pd.DataFrame, output_path: Path) -> None:
    order = (
        df.groupby("feature", as_index=False)["value"]
        .mean()
        .sort_values("value", ascending=False)["feature"]
        .astype(str)
        .tolist()
    )
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="value", y="feature", order=order, color="#7aa6c2")
    plt.title("Static Feature Ranges")
    plt.xlabel("Value")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_distribution_plot(df: pd.DataFrame, output_path: Path) -> None:
    g = sns.FacetGrid(df, col="feature", col_wrap=4, sharex=False, sharey=False, height=2.4)
    g.map_dataframe(sns.histplot, x="value", bins=40, kde=False, color="#2a9d8f")
    g.set_titles("{col_name}")
    g.tight_layout()
    g.savefig(output_path, dpi=180)
    plt.close("all")


def save_categorical_counts(
    df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_path: Path,
    top_k: int,
) -> None:
    cats = summary_df[summary_df["likely_categorical"]]["feature"].tolist()
    rows: list[dict[str, object]] = []
    for feature in cats:
        vc = df[feature].value_counts(dropna=False).head(top_k)
        for value, count in vc.items():
            rows.append(
                {
                    "feature": feature,
                    "value": value,
                    "count": int(count),
                }
            )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    indices_path = args.landscape_dir / "indices.json"
    if not indices_path.exists():
        raise FileNotFoundError(f"indices.json not found: {indices_path}")

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_feature_frame(indices_path)
    summary_df = detect_categorical_columns(df)
    long_df = df.drop(columns=["fire_id"]).melt(var_name="feature", value_name="value")

    summary_path = out_dir / "g_feature_summary.csv"
    features_path = out_dir / "g_feature_table.csv"
    meanings_path = out_dir / "static_channel_meanings.csv"
    categories_path = out_dir / "g_categorical_candidates.csv"
    cat_counts_path = out_dir / "g_categorical_value_counts.csv"
    ranges_plot_path = out_dir / "g_feature_ranges.png"
    dist_plot_path = out_dir / "g_feature_distributions.png"

    df.to_csv(features_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    pd.DataFrame(STATIC_CHANNEL_MEANINGS).to_csv(meanings_path, index=False)
    summary_df[summary_df["likely_categorical"]].to_csv(categories_path, index=False)
    save_categorical_counts(df, summary_df, cat_counts_path, top_k=args.top_categorical_values)
    save_range_plot(long_df, ranges_plot_path)
    save_distribution_plot(long_df, dist_plot_path)

    print(f"[ok] rows={len(df)}")
    print(f"[ok] summary={summary_path}")
    print(f"[ok] meanings={meanings_path}")
    print(f"[ok] categories={categories_path}")
    print(f"[ok] range_plot={ranges_plot_path}")
    print(f"[ok] dist_plot={dist_plot_path}")


if __name__ == "__main__":
    main()
