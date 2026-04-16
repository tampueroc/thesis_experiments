from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_SUMMARY_KEYS = [
    "val/z_mse",
    "val/z_cosine",
    "rollout/val/z_mse@10",
    "rollout/val/z_cosine@10",
    "val/loss",
    "val/bce",
    "val/dice_loss",
    "val/soft_dice",
    "val/soft_iou",
    "val/hard_dice",
    "val/hard_iou",
    "train/loss",
    "train/hard_dice",
    "train/hard_iou",
    "monitor_value_best",
]
DEFAULT_BASE_KEYS = ["run_id", "name", "state", "created_at"]
DEFAULT_CONFIG_KEYS = [
    "component",
    "family",
    "variant",
    "timestamp",
    "source_timestamp",
    "embeddings_model_slug",
    "source_embeddings_model_slug",
    "d_model",
    "nhead",
    "dim_feedforward",
    "norm_first",
    "batch_size",
    "epochs",
    "learning_rate",
    "lr",
    "dataset",
    "dataset/normalize_static_num",
    "normalize_embeddings",
    "hard_threshold",
    "mask_threshold",
    "prediction_mode",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Weights & Biases runs into a pandas DataFrame for analysis."
    )
    parser.add_argument(
        "--project",
        default="tampueroc-university-of-chile/latent_wildfire",
        help="W&B project in the form entity/project-name.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path. Supported suffixes: .csv, .json, .jsonl, .parquet, .pkl.",
    )
    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="Keep raw summary/config dict columns in the exported DataFrame.",
    )
    parser.add_argument(
        "--summary-key",
        action="append",
        dest="summary_keys",
        default=None,
        help=(
            "Summary metric key to export. Repeat to add more keys. "
            "Defaults to the core validation z-metrics."
        ),
    )
    parser.add_argument(
        "--config-key",
        action="append",
        dest="config_keys",
        default=None,
        help=(
            "Config key to export. Repeat to add more keys. "
            "Defaults to the core architecture, training, and dataset fields."
        ),
    )
    return parser.parse_args()


def fetch_runs(project: str) -> pd.DataFrame:
    try:
        import wandb  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "wandb is required to fetch runs. Install it in your environment before running this script."
        ) from exc

    api = wandb.Api()
    runs = api.runs(project)

    records: list[dict[str, Any]] = []
    for run in runs:
        summary = run.summary._json_dict
        config = {key: value for key, value in run.config.items() if not key.startswith("_")}
        records.append(
            {
                "run_id": run.id,
                "name": run.name,
                "state": run.state,
                "created_at": run.created_at,
                "summary": summary,
                "config": config,
            }
        )

    return pd.DataFrame.from_records(records)


def flatten_runs_dataframe(
    runs_df: pd.DataFrame,
    *,
    include_raw: bool,
    summary_keys: list[str],
    config_keys: list[str],
) -> pd.DataFrame:
    if runs_df.empty:
        return runs_df.copy()

    summary_records = [
        {key: summary.get(key) for key in summary_keys}
        for summary in runs_df["summary"]
    ]
    config_records = [
        {key: config.get(key) for key in config_keys}
        for config in runs_df["config"]
    ]
    summary_df = pd.DataFrame.from_records(summary_records).add_prefix("summary.")
    config_df = pd.DataFrame.from_records(config_records).add_prefix("config.")
    if include_raw:
        base_df = runs_df.copy()
    else:
        base_df = runs_df.loc[:, [column for column in DEFAULT_BASE_KEYS if column in runs_df.columns]]
    return pd.concat([base_df, summary_df, config_df], axis=1)


def write_dataframe(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()

    if suffix == ".csv":
        df.to_csv(output_path, index=False)
        return
    if suffix == ".json":
        payload = df.to_json(orient="records", indent=2)
        if payload is None:
            raise RuntimeError("pandas returned no JSON payload for .json export")
        output_path.write_text(payload, encoding="utf-8")
        return
    if suffix == ".jsonl":
        payload = df.to_json(orient="records", lines=True)
        if payload is None:
            raise RuntimeError("pandas returned no JSON payload for .jsonl export")
        output_path.write_text(payload, encoding="utf-8")
        return
    if suffix == ".parquet":
        df.to_parquet(output_path, index=False)
        return
    if suffix == ".pkl":
        df.to_pickle(output_path)
        return

    supported = ", ".join([".csv", ".json", ".jsonl", ".parquet", ".pkl"])
    raise ValueError(f"unsupported output suffix {suffix!r}; expected one of: {supported}")


def main() -> int:
    args = parse_args()
    runs_df = fetch_runs(args.project)
    analysis_df = flatten_runs_dataframe(
        runs_df,
        include_raw=args.include_raw,
        summary_keys=args.summary_keys or DEFAULT_SUMMARY_KEYS,
        config_keys=args.config_keys or DEFAULT_CONFIG_KEYS,
    )

    print(f"Fetched {len(analysis_df)} runs from {args.project}.")
    print(json.dumps({"columns": analysis_df.columns.tolist()}, indent=2))

    if args.output is not None:
        write_dataframe(analysis_df, args.output)
        print(f"Wrote analysis DataFrame to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
