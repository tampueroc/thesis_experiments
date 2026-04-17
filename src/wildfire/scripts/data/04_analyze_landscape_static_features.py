from __future__ import annotations

import argparse
from datetime import UTC, datetime
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
        description="Analyze Input_Geotiff.tif by channel and emit publication-ready seaborn plots.",
    )
    parser.add_argument("--landscape-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("landscape_static_analss"))
    parser.add_argument("--sample-per-channel", type=int, default=120_000)
    parser.add_argument("--categorical-unique-threshold", type=int, default=64)
    parser.add_argument("--top-categorical-values", type=int, default=20)
    parser.add_argument(
        "--run-timestamp",
        type=str,
        default="",
        help="Optional timestamp folder name (default: current UTC as ISO-like).",
    )
    return parser.parse_args()


def _to_channel_first(arr: np.ndarray) -> tuple[np.ndarray, int]:
    if arr.ndim != 3:
        raise ValueError(f"expected 3D GeoTIFF data, got shape={arr.shape}")
    if arr.shape[0] <= 32:
        return arr, arr.shape[0]
    if arr.shape[-1] <= 32:
        return arr.transpose(2, 0, 1), arr.shape[-1]
    raise ValueError(f"cannot infer channel axis from shape={arr.shape}")


def _normalize_nodata(arr: np.ndarray) -> np.ndarray:
    return np.where(arr == RAW_NODATA_VALUE, NODATA_VALUE, arr)


def _valid_values(series: pd.Series) -> pd.Series:
    return series[pd.notna(series) & (series != NODATA_VALUE)]


def _resolve_run_dir(base_dir: Path, run_timestamp: str) -> Path:
    ts = run_timestamp or datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    out = base_dir / ts
    out.mkdir(parents=True, exist_ok=True)
    return out


def _channel_meta(channel_idx: int) -> dict[str, str]:
    channel = f"a{channel_idx}"
    return next((m for m in STATIC_CHANNEL_MEANINGS if m["channel"] == channel), {"channel": channel, "feature": channel, "meaning": "", "expected_type": "unknown"})


def build_summary(channel_first: np.ndarray, categorical_unique_threshold: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx, channel_values in enumerate(channel_first, start=1):
        meta = _channel_meta(idx)
        series = pd.Series(channel_values.reshape(-1))
        valid = _valid_values(series)
        n_total = int(len(series))
        n_nodata = int((series == NODATA_VALUE).sum())
        n_valid = int(len(valid))
        n_unique_valid = int(valid.nunique(dropna=True))
        is_integer_like = bool(((valid % 1) == 0).all()) if n_valid else False
        likely_categorical = n_unique_valid <= categorical_unique_threshold and is_integer_like
        n_unique_all = n_unique_valid + int(n_nodata > 0)
        rows.append(
            {
                "channel": meta["channel"],
                "feature": meta["feature"],
                "meaning": meta["meaning"],
                "expected_type": meta["expected_type"],
                "n_total": n_total,
                "n_nodata": n_nodata,
                "nodata_ratio": float(n_nodata / n_total) if n_total else float("nan"),
                "n_valid": n_valid,
                "valid_ratio": float(n_valid / n_total) if n_total else float("nan"),
                "n_unique_valid": n_unique_valid,
                "n_unique_all": n_unique_all,
                "min": float(valid.min()) if n_valid else float("nan"),
                "max": float(valid.max()) if n_valid else float("nan"),
                "mean": float(valid.mean()) if n_valid else float("nan"),
                "std": float(valid.std()) if n_valid else float("nan"),
                "is_integer_like": is_integer_like,
                "likely_categorical": likely_categorical,
            }
        )
    summary = pd.DataFrame(rows).sort_values("channel").reset_index(drop=True)
    summary["is_binary_presence"] = (
        (summary["n_unique_valid"] == 1)
        & (summary["n_unique_all"] == 2)
        & (summary["min"] == 1.0)
        & (summary["max"] == 1.0)
    )
    summary["is_multiclass_categorical"] = summary["likely_categorical"] & ~summary["is_binary_presence"] & (summary["n_unique_valid"] > 1)
    summary["is_continuous"] = ~summary["likely_categorical"]
    return summary


def build_long_values(
    channel_first: np.ndarray,
    summary_df: pd.DataFrame,
    only_column: str,
    sample_per_channel: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    selected = summary_df[summary_df[only_column]]["channel"].tolist()
    for channel in selected:
        idx = int(channel[1:]) - 1
        feature = str(summary_df[summary_df["channel"] == channel]["feature"].iloc[0])
        values = _valid_values(pd.Series(channel_first[idx].reshape(-1)))
        if len(values) > sample_per_channel:
            values = values.sample(n=sample_per_channel, random_state=7)
        rows.append(pd.DataFrame({"channel": channel, "feature": feature, "value": values.values}))
    if not rows:
        return pd.DataFrame(columns=["channel", "feature", "value"])
    return pd.concat(rows, ignore_index=True)


def build_categorical_counts(
    channel_first: np.ndarray,
    summary_df: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    channels = summary_df[summary_df["is_multiclass_categorical"]]["channel"].tolist()
    for channel in channels:
        idx = int(channel[1:]) - 1
        feature = str(summary_df[summary_df["channel"] == channel]["feature"].iloc[0])
        vc = _valid_values(pd.Series(channel_first[idx].reshape(-1))).value_counts().head(top_k)
        total = int(vc.sum()) if len(vc) else 1
        for value, count in vc.items():
            rows.append(
                {
                    "channel": channel,
                    "feature": feature,
                    "value": str(value),
                    "count": int(count),
                    "ratio_in_topk": float(count / total),
                }
            )
    return pd.DataFrame(rows, columns=["channel", "feature", "value", "count", "ratio_in_topk"])


def save_continuous_distribution_plot(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        return
    g = sns.FacetGrid(df, col="feature", col_wrap=3, sharex=False, sharey=False, height=3.0)
    g.map_dataframe(sns.histplot, x="value", bins=60, stat="density", color="#2a9d8f", alpha=0.75)
    g.map_dataframe(sns.kdeplot, x="value", color="#1d3557", linewidth=1.6)
    g.set_axis_labels("Value", "Density")
    g.set_titles("{col_name}")
    g.tight_layout()
    g.savefig(output_path, dpi=200)
    plt.close("all")


def save_continuous_range_plot(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        return
    median_df = df.groupby("feature", as_index=False).agg(value=("value", "median"))
    order = median_df.sort_values(by="value", ascending=False)["feature"].tolist()
    plt.figure(figsize=(11, 5.5))
    sns.boxenplot(data=df, x="value", y="feature", order=order, color="#457b9d")
    plt.title("Continuous Channel Value Ranges (No -1 No-Data)")
    plt.xlabel("Value")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_multiclass_plot(counts_df: pd.DataFrame, output_path: Path) -> None:
    if counts_df.empty:
        return
    g = sns.FacetGrid(counts_df, col="feature", col_wrap=2, sharex=False, sharey=False, height=3.2)
    g.map_dataframe(sns.barplot, x="value", y="ratio_in_topk", color="#264653")
    g.set_axis_labels("Class Value", "Share Within Top-K")
    g.set_titles("{col_name}")
    for ax in g.axes.flatten():
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylim(0, 1)
    g.tight_layout()
    g.savefig(output_path, dpi=200)
    plt.close("all")


def save_binary_presence_plot(summary_df: pd.DataFrame, output_path: Path) -> None:
    binary_df = summary_df[summary_df["is_binary_presence"]].copy()
    if binary_df.empty:
        return
    plot_df = pd.concat(
        [
            binary_df[["feature", "valid_ratio"]].rename(columns={"valid_ratio": "ratio"}).assign(state="present(1)"),
            binary_df[["feature", "nodata_ratio"]].rename(columns={"nodata_ratio": "ratio"}).assign(state="no_data(-1)"),
        ],
        ignore_index=True,
    )
    plt.figure(figsize=(10, 4.2))
    sns.barplot(data=plot_df, x="feature", y="ratio", hue="state", palette=["#2a9d8f", "#e76f51"])
    plt.title("Binary Presence Channels: Present vs No-Data")
    plt.xlabel("Feature")
    plt.ylabel("Ratio")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_nodata_plot(summary_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(9, 4.2))
    plot_df = summary_df.sort_values("nodata_ratio", ascending=False)
    sns.barplot(data=plot_df, x="feature", y="nodata_ratio", color="#e76f51")
    plt.title("No-Data (-1) Ratio by Channel")
    plt.xlabel("Feature")
    plt.ylabel("No-data ratio")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    geotiff_path = args.landscape_dir / "Input_Geotiff.tif"
    if not geotiff_path.exists():
        raise FileNotFoundError(f"Input_Geotiff.tif not found: {geotiff_path}")

    sns.set_theme(style="whitegrid", context="talk")
    run_dir = _resolve_run_dir(args.output_dir, args.run_timestamp.strip())

    channel_first, n_channels = _to_channel_first(_normalize_nodata(np.asarray(tifffile.imread(geotiff_path))))
    summary_df = build_summary(channel_first, categorical_unique_threshold=max(1, args.categorical_unique_threshold))

    continuous_df = build_long_values(
        channel_first=channel_first,
        summary_df=summary_df,
        only_column="is_continuous",
        sample_per_channel=max(1, args.sample_per_channel),
    )
    multiclass_counts_df = build_categorical_counts(
        channel_first=channel_first,
        summary_df=summary_df,
        top_k=max(1, args.top_categorical_values),
    )

    summary_path = run_dir / "g_channel_summary.csv"
    meanings_path = run_dir / "static_channel_meanings.csv"
    multiclass_counts_path = run_dir / "g_channel_multiclass_value_counts.csv"
    cont_dist_path = run_dir / "g_channel_continuous_distributions.png"
    cont_range_path = run_dir / "g_channel_continuous_ranges.png"
    multiclass_plot_path = run_dir / "g_channel_multiclass_top_values.png"
    binary_plot_path = run_dir / "g_channel_binary_presence.png"
    nodata_plot_path = run_dir / "g_channel_nodata_rates.png"

    summary_df.to_csv(summary_path, index=False)
    pd.DataFrame(STATIC_CHANNEL_MEANINGS).to_csv(meanings_path, index=False)
    multiclass_counts_df.to_csv(multiclass_counts_path, index=False)
    save_continuous_distribution_plot(continuous_df, cont_dist_path)
    save_continuous_range_plot(continuous_df, cont_range_path)
    save_multiclass_plot(multiclass_counts_df, multiclass_plot_path)
    save_binary_presence_plot(summary_df, binary_plot_path)
    save_nodata_plot(summary_df, nodata_plot_path)

    print(f"[ok] geotiff={geotiff_path}")
    print(f"[ok] channels={n_channels}")
    print(f"[ok] run_dir={run_dir}")
    print(f"[ok] summary={summary_path}")
    print(f"[ok] meanings={meanings_path}")
    print(f"[ok] multiclass_counts={multiclass_counts_path}")
    print(f"[ok] continuous_dist={cont_dist_path}")
    print(f"[ok] continuous_range={cont_range_path}")
    print(f"[ok] multiclass_plot={multiclass_plot_path}")
    print(f"[ok] binary_plot={binary_plot_path}")
    print(f"[ok] nodata_plot={nodata_plot_path}")


if __name__ == "__main__":
    main()
