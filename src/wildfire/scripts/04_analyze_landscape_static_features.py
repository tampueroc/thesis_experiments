from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tifffile


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

RAW_NODATA_VALUE = -9999.0
NODATA_VALUE = -1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze static landscape channels from Input_Geotiff.tif.",
    )
    parser.add_argument(
        "--landscape-dir",
        type=Path,
        required=True,
        help="Path containing Input_Geotiff.tif (for example /.../thesis_data/landscape).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("landscape_static_analss"),
        help="Output directory for generated artifacts.",
    )
    parser.add_argument(
        "--sample-per-channel",
        type=int,
        default=120_000,
        help="Max sampled pixels per channel for histogram plotting.",
    )
    parser.add_argument(
        "--categorical-unique-threshold",
        type=int,
        default=64,
        help="Mark channel as likely categorical when unique non-NaN values are <= threshold.",
    )
    parser.add_argument(
        "--top-categorical-values",
        type=int,
        default=30,
        help="Top K values saved for likely categorical channels.",
    )
    return parser.parse_args()


def _to_channel_first(arr: np.ndarray) -> tuple[np.ndarray, int]:
    if arr.ndim != 3:
        raise ValueError(f"expected 3D GeoTIFF data, got shape={arr.shape}")
    shape = arr.shape
    if shape[0] <= 32:
        return arr, shape[0]
    if shape[-1] <= 32:
        return arr.transpose(2, 0, 1), shape[-1]
    raise ValueError(f"cannot infer channel axis from shape={shape}")


def _normalize_nodata(arr: np.ndarray) -> np.ndarray:
    return np.where(arr == RAW_NODATA_VALUE, NODATA_VALUE, arr)


def _valid_values(series: pd.Series) -> pd.Series:
    return series[pd.notna(series) & (series != NODATA_VALUE)]


def build_channel_data_frame(
    channel_first: np.ndarray,
    categorical_unique_threshold: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx, channel_values in enumerate(channel_first, start=1):
        feature_meta = next((m for m in STATIC_CHANNEL_MEANINGS if m["channel"] == f"a{idx}"), None)
        flat = channel_values.reshape(-1)
        series = pd.Series(flat)
        n_total = int(len(series))
        n_nodata = int((series == NODATA_VALUE).sum())
        finite = _valid_values(series)
        n = int(len(finite))
        unique_n = int(finite.nunique(dropna=True))
        is_integer_like = bool(((finite % 1) == 0).all()) if n else False
        likely_categorical = unique_n <= categorical_unique_threshold and is_integer_like
        rows.append(
            {
                "channel": f"a{idx}",
                "feature": feature_meta["feature"] if feature_meta else f"a{idx}",
                "meaning": feature_meta["meaning"] if feature_meta else "",
                "expected_type": feature_meta["expected_type"] if feature_meta else "",
                "dtype": str(series.dtype),
                "n_total": n_total,
                "n_nodata": n_nodata,
                "nodata_ratio": float(n_nodata / n_total) if n_total else float("nan"),
                "n_pixels": n,
                "n_unique": unique_n,
                "min": float(finite.min()) if n else float("nan"),
                "max": float(finite.max()) if n else float("nan"),
                "mean": float(finite.mean()) if n else float("nan"),
                "std": float(finite.std()) if n else float("nan"),
                "is_integer_like": is_integer_like,
                "likely_categorical": likely_categorical,
            }
        )
    return pd.DataFrame(rows).sort_values("channel").reset_index(drop=True)


def build_long_distribution_frame(
    channel_first: np.ndarray,
    summary_df: pd.DataFrame,
    sample_per_channel: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for idx, channel_values in enumerate(channel_first, start=1):
        channel = f"a{idx}"
        feature_row = summary_df[summary_df["channel"] == channel]
        if feature_row.empty:
            continue
        feature_name = str(feature_row.iloc[0]["feature"])
        series = pd.Series(channel_values.reshape(-1))
        series = _valid_values(series)
        if len(series) > sample_per_channel:
            series = series.sample(n=sample_per_channel, random_state=7)
        rows.append(pd.DataFrame({"channel": channel, "feature": feature_name, "value": series.values}))
    if not rows:
        return pd.DataFrame(columns=["channel", "feature", "value"])
    return pd.concat(rows, ignore_index=True)


def save_distribution_plot(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        return
    g = sns.FacetGrid(df, col="feature", col_wrap=4, sharex=False, sharey=False, height=2.6)
    g.map_dataframe(sns.histplot, x="value", bins=80, kde=False, color="#2a9d8f")
    g.set_titles("{col_name}")
    g.tight_layout()
    g.savefig(output_path, dpi=180)
    plt.close("all")


def save_range_plot(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        return
    order = (
        df.groupby("feature", as_index=False)["value"]
        .mean()
        .sort_values("value", ascending=False)["feature"]
        .astype(str)
        .tolist()
    )
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="value", y="feature", order=order, color="#7aa6c2")
    plt.title("Landscape Channel Ranges")
    plt.xlabel("Value")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_categorical_counts(
    channel_first: np.ndarray,
    summary_df: pd.DataFrame,
    output_path: Path,
    top_k: int,
) -> None:
    rows: list[dict[str, object]] = []
    categories = summary_df[summary_df["likely_categorical"]]["channel"].tolist()
    for channel in categories:
        channel_idx = int(channel[1:]) - 1
        values = pd.Series(channel_first[channel_idx].reshape(-1))
        values = _valid_values(values)
        vc = values.value_counts(dropna=False).head(top_k)
        feature_name = str(summary_df[summary_df["channel"] == channel].iloc[0]["feature"])
        for value, count in vc.items():
            rows.append(
                {
                    "channel": channel,
                    "feature": feature_name,
                    "value": value,
                    "count": int(count),
                }
            )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def save_nodata_distribution(summary_df: pd.DataFrame, output_path: Path) -> None:
    summary_df[["channel", "feature", "n_total", "n_nodata", "nodata_ratio"]].to_csv(output_path, index=False)


def save_nodata_plot(summary_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = summary_df.sort_values("nodata_ratio", ascending=False)
    plt.figure(figsize=(8, 4))
    sns.barplot(data=plot_df, x="feature", y="nodata_ratio", color="#e76f51")
    plt.title("No-Data Ratio by Channel")
    plt.xlabel("Feature")
    plt.ylabel("No-data ratio")
    plt.ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    geotiff_path = args.landscape_dir / "Input_Geotiff.tif"
    if not geotiff_path.exists():
        raise FileNotFoundError(f"Input_Geotiff.tif not found: {geotiff_path}")

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = np.asarray(tifffile.imread(geotiff_path))
    raw = _normalize_nodata(raw)
    channel_first, n_channels = _to_channel_first(raw)

    summary_df = build_channel_data_frame(
        channel_first=channel_first,
        categorical_unique_threshold=max(1, args.categorical_unique_threshold),
    )
    long_df = build_long_distribution_frame(
        channel_first=channel_first,
        summary_df=summary_df,
        sample_per_channel=max(1, args.sample_per_channel),
    )

    summary_path = out_dir / "g_channel_summary.csv"
    meanings_path = out_dir / "static_channel_meanings.csv"
    cat_counts_path = out_dir / "g_channel_categorical_value_counts.csv"
    nodata_dist_path = out_dir / "g_channel_nodata_distribution.csv"
    nodata_plot_path = out_dir / "g_channel_nodata_rates.png"
    ranges_plot_path = out_dir / "g_channel_ranges.png"
    dist_plot_path = out_dir / "g_channel_distributions.png"

    summary_df.to_csv(summary_path, index=False)
    pd.DataFrame(STATIC_CHANNEL_MEANINGS).to_csv(meanings_path, index=False)
    save_categorical_counts(
        channel_first=channel_first,
        summary_df=summary_df,
        output_path=cat_counts_path,
        top_k=max(1, args.top_categorical_values),
    )
    save_nodata_distribution(summary_df, nodata_dist_path)
    save_nodata_plot(summary_df, nodata_plot_path)
    save_range_plot(long_df, ranges_plot_path)
    save_distribution_plot(long_df, dist_plot_path)

    print(f"[ok] geotiff={geotiff_path}")
    print(f"[ok] channels={n_channels}")
    print(f"[ok] summary={summary_path}")
    print(f"[ok] meanings={meanings_path}")
    print(f"[ok] categorical_counts={cat_counts_path}")
    print(f"[ok] nodata_dist={nodata_dist_path}")
    print(f"[ok] nodata_plot={nodata_plot_path}")
    print(f"[ok] range_plot={ranges_plot_path}")
    print(f"[ok] dist_plot={dist_plot_path}")


if __name__ == "__main__":
    main()
