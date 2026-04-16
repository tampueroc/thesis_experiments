from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from wildfire.data.real_data import build_sources
from wildfire.model_latent_predictor.model_01 import LSTMConfig, LatentLSTMPredictor
from wildfire.model_latent_predictor.transformer_01 import TransformerConfig, TransformerLatentPredictor
from wildfire.model_latent_predictor.transformer_static_01 import (
    StaticTransformerConfig,
    StaticTransformerLatentPredictor,
)
from wildfire.model_latent_predictor.transformer_static_film_head_01 import (
    StaticFiLMHeadTransformerConfig,
    StaticFiLMHeadTransformerLatentPredictor,
)
from wildfire.model_latent_predictor.transformer_static_head_01 import (
    StaticHeadTransformerConfig,
    StaticHeadTransformerLatentPredictor,
)

TORCH = cast(Any, torch)
LOGGER = logging.getLogger("wildfire.export.latent_predictor_embeddings")
STATIC_CATEGORICAL_INDICES = (0,)
STATIC_MISSING_VALUE = -1.0
CATEGORICAL_UNKNOWN_INDEX = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export predicted latent embeddings from a trained Model 1 run into a "
            "manifest-backed per-sequence dataset for downstream decoder experiments."
        )
    )
    parser.add_argument("--run-id", default="", help="Predictor run_id under artifacts/wildfire/latent_predictor/*/.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Explicit predictor run directory containing metrics.json and best_model.pt.")
    parser.add_argument(
        "--landscape-dir",
        type=Path,
        default=Path("/home/tampuero/data/thesis_data/landscape"),
        help="Landscape folder containing WeatherHistory.csv, Weathers/, and indices.json.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "trainval", "holdout", "all"],
        default="trainval",
        help="Which split to export. Defaults to train+val to keep holdout untouched during development.",
    )
    parser.add_argument(
        "--prediction-mode",
        choices=["teacher_forced", "rollout"],
        default="teacher_forced",
        help="Prediction mode. Teacher-forced is the safer default for initial decoder blending.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device string. Default: auto (cuda, then mps, else cpu).",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=-1,
        help="Optional override for source sequence limit. -1 means use the training run setting.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional explicit export directory. Default: <run_dir>/predicted_embeddings/<prediction_mode>_<split>/",
    )
    return parser.parse_args()


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "export.log"
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)


def resolve_device(device_arg: str) -> Any:
    if device_arg != "auto":
        return TORCH.device(device_arg)
    if TORCH.cuda.is_available():
        return TORCH.device("cuda")
    if hasattr(TORCH.backends, "mps") and TORCH.backends.mps.is_available():
        return TORCH.device("mps")
    return TORCH.device("cpu")


def resolve_run_dir(run_id: str, explicit_run_dir: Path | None) -> Path:
    if explicit_run_dir is not None:
        return explicit_run_dir
    if not run_id:
        raise ValueError("either --run-id or --run-dir is required")
    roots = list((Path("artifacts/wildfire") / "latent_predictor").glob(f"*/{run_id}"))
    if not roots:
        raise FileNotFoundError(f"run_id not found under artifacts/wildfire/latent_predictor: {run_id}")
    if len(roots) > 1:
        raise RuntimeError(f"run_id resolved to multiple directories: {roots}")
    return roots[0]


def load_run_summary(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "metrics.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"run summary not found: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError(f"invalid metrics.json format: {summary_path}")
    return summary


def split_fire_ids(
    fire_ids: list[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[set[str], set[str], set[str]]:
    if min(train_ratio, val_ratio) <= 0.0:
        raise ValueError("train/val ratios must both be > 0")
    ratio_sum = train_ratio + val_ratio
    if ratio_sum > 1.0 + 1e-6:
        raise ValueError(f"train+val must be <= 1.0, got {ratio_sum:.6f}")

    ordered_ids = sorted(fire_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(ordered_ids)

    n_total = len(ordered_ids)
    if n_total < 2:
        raise ValueError(f"need at least 2 sequences to build train/val splits, got {n_total}")

    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    if n_train == 0:
        n_train = 1
    if n_val == 0:
        n_val = 1

    if n_train + n_val >= n_total:
        n_val = max(1, n_total - n_train - 1)
        if n_val <= 0:
            n_val = 1
            n_train = max(1, n_total - n_val - 1)
        if n_train + n_val >= n_total:
            raise ValueError("not enough sequences to keep holdout set separate")

    train_ids = set(ordered_ids[:n_train])
    val_ids = set(ordered_ids[n_train : n_train + n_val])
    holdout_ids = set(ordered_ids[n_train + n_val :])
    return train_ids, val_ids, holdout_ids


def select_ids(
    split: str,
    train_ids: set[str],
    val_ids: set[str],
    holdout_ids: set[str],
) -> set[str]:
    if split == "train":
        return train_ids
    if split == "val":
        return val_ids
    if split == "trainval":
        return train_ids | val_ids
    if split == "holdout":
        return holdout_ids
    return train_ids | val_ids | holdout_ids


def encode_static_features(g: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cat_indices = np.asarray(STATIC_CATEGORICAL_INDICES, dtype=np.int64)
    num_indices = np.asarray([i for i in range(g.shape[0]) if i not in STATIC_CATEGORICAL_INDICES], dtype=np.int64)

    g_num_raw = g[num_indices] if num_indices.size else np.zeros((0,), dtype=np.float32)
    g_num_mask = (g_num_raw != STATIC_MISSING_VALUE).astype(np.float32)
    g_num = np.where(g_num_mask == 1.0, g_num_raw, 0.0).astype(np.float32)

    if cat_indices.size:
        g_cat_raw = g[cat_indices]
        cat_valid = g_cat_raw != STATIC_MISSING_VALUE
        g_cat = np.full(cat_indices.shape[0], np.int64(CATEGORICAL_UNKNOWN_INDEX), dtype=np.int64)
        for idx, is_valid in enumerate(cat_valid.tolist()):
            if not bool(is_valid):
                continue
            g_cat[idx] = np.int64(int(np.rint(g_cat_raw[idx])))
    else:
        g_cat = np.zeros((0,), dtype=np.int64)
    return g_num, g_num_mask, g_cat


def build_model(summary: dict[str, Any], device: Any) -> tuple[Any, bool, bool]:
    variant = str(summary.get("variant", ""))
    config_data = summary.get("config")
    if not isinstance(config_data, dict):
        raise ValueError("run summary missing config block")

    is_residual = "residual" in variant
    uses_static = "static" in variant

    if variant == "model_01" and str(summary.get("family", "")) == "lstm":
        model = LatentLSTMPredictor(LSTMConfig(**config_data))
    elif variant in {"model_01", "model_01_residual"}:
        model = TransformerLatentPredictor(TransformerConfig(**config_data))
    elif variant == "model_01_static":
        model = StaticTransformerLatentPredictor(StaticTransformerConfig(**config_data))
    elif variant == "model_01_static_head":
        model = StaticHeadTransformerLatentPredictor(StaticHeadTransformerConfig(**config_data))
    elif variant in {"model_01_static_film_head", "model_01_static_film_head_residual"}:
        model = StaticFiLMHeadTransformerLatentPredictor(StaticFiLMHeadTransformerConfig(**config_data))
    else:
        raise ValueError(f"unsupported predictor variant for export: {variant}")
    return model.to(device), is_residual, uses_static


def predict_next_embedding(
    model: Any,
    *,
    window: np.ndarray,
    g: np.ndarray,
    uses_static: bool,
    is_residual: bool,
    normalize_embeddings: bool,
    device: Any,
) -> np.ndarray:
    window_t = TORCH.tensor(window, dtype=TORCH.float32, device=device).unsqueeze(0)
    if uses_static:
        g_num, g_num_mask, g_cat = encode_static_features(g)
        g_num_t = TORCH.tensor(g_num, dtype=TORCH.float32, device=device).unsqueeze(0)
        g_num_mask_t = TORCH.tensor(g_num_mask, dtype=TORCH.float32, device=device).unsqueeze(0)
        g_cat_t = TORCH.tensor(g_cat, dtype=TORCH.int64, device=device).unsqueeze(0)
        raw_pred = model(window_t, g_num_t, g_num_mask_t, g_cat_t)
    else:
        raw_pred = model(window_t)

    if is_residual:
        raw_pred = window_t[:, -1, :] + raw_pred
    if normalize_embeddings:
        raw_pred = raw_pred / TORCH.norm(raw_pred, dim=1, keepdim=True).clamp_min(1e-8)
    return np.asarray(raw_pred.squeeze(0).detach().cpu(), dtype=np.float32)


def export_sequence_predictions(
    *,
    model: Any,
    z: np.ndarray,
    g: np.ndarray,
    history: int,
    uses_static: bool,
    is_residual: bool,
    normalize_embeddings: bool,
    prediction_mode: str,
    device: Any,
) -> np.ndarray:
    if z.ndim != 2:
        raise ValueError(f"expected 2D sequence embeddings, got {z.shape}")
    if z.shape[0] < history + 1:
        raise ValueError(f"sequence too short for history={history}: {z.shape}")

    exported = np.asarray(z, dtype=np.float32).copy()
    for target_idx in range(history, z.shape[0]):
        if prediction_mode == "teacher_forced":
            window = z[target_idx - history : target_idx]
        else:
            window = exported[target_idx - history : target_idx]
        exported[target_idx] = predict_next_embedding(
            model,
            window=window,
            g=g,
            uses_static=uses_static,
            is_residual=is_residual,
            normalize_embeddings=normalize_embeddings,
            device=device,
        )
    return exported


def main() -> None:
    args = parse_args()
    run_dir = resolve_run_dir(args.run_id, args.run_dir)
    summary = load_run_summary(run_dir)

    default_output_dir = run_dir / "predicted_embeddings" / f"{args.prediction_mode}_{args.split}"
    output_dir = args.output_dir if args.output_dir is not None else default_output_dir
    setup_logging(output_dir)

    checkpoint_path = run_dir / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    model_dir = Path(str(summary.get("model_dir", "")))
    if not model_dir.exists():
        raise FileNotFoundError(f"source model_dir not found: {model_dir}")
    manifest_path = model_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    input_dir = str(manifest.get("input_dir", ""))
    if not input_dir:
        raise ValueError(f"manifest missing input_dir: {manifest_path}")

    timestamp = str(summary.get("timestamp", ""))
    input_type = str(summary.get("config", {}).get("input_type", summary.get("input_type", "fire_frames")))
    embeddings_model_slug = str(summary.get("config", {}).get("embeddings_model_slug", ""))
    seed = int(summary.get("config", {}).get("seed", 7))
    train_ratio = float(summary.get("config", {}).get("train_ratio", 0.7))
    val_ratio = float(summary.get("config", {}).get("val_ratio", 0.15))
    history = int(summary.get("history", summary.get("config", {}).get("max_history", 5)))
    normalize_embeddings = bool(summary.get("config", {}).get("dataset/normalize_embeddings", summary.get("dataset/normalize_embeddings", False)))
    if "dataset/normalize_embeddings" not in summary.get("config", {}) and "dataset/normalize_embeddings" not in summary:
        normalize_embeddings = bool(summary.get("config", {}).get("normalize_embeddings", False))

    source_max_sequences = int(summary.get("config", {}).get("max_sequences", 0))
    effective_max_sequences = source_max_sequences if args.max_sequences < 0 else args.max_sequences

    LOGGER.info("run_dir=%s", run_dir)
    LOGGER.info("checkpoint=%s", checkpoint_path)
    LOGGER.info("source_model_dir=%s", model_dir)
    LOGGER.info("prediction_mode=%s split=%s history=%d normalize_embeddings=%s", args.prediction_mode, args.split, history, normalize_embeddings)

    z_by_fire, w_by_fire, g_by_fire = build_sources(
        model_dir=model_dir,
        landscape_dir=args.landscape_dir,
        max_sequences=effective_max_sequences,
    )
    if normalize_embeddings:
        from wildfire.data.dataset import WildfireSequenceDataset

        z_by_fire = WildfireSequenceDataset._normalize_per_fire_embeddings(z_by_fire)

    train_ids, val_ids, holdout_ids = split_fire_ids(
        fire_ids=list(z_by_fire.keys()),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )
    selected_ids = select_ids(args.split, train_ids, val_ids, holdout_ids)
    if not selected_ids:
        raise RuntimeError(f"selected split has no sequences: {args.split}")

    device = resolve_device(args.device)
    model, is_residual, uses_static = build_model(summary, device)
    state = TORCH.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    exported_sequences: list[dict[str, Any]] = []
    with TORCH.no_grad():
        for fire_id in sorted(selected_ids):
            z = z_by_fire[fire_id]
            g = g_by_fire[fire_id]
            exported = export_sequence_predictions(
                model=model,
                z=z,
                g=g,
                history=history,
                uses_static=uses_static,
                is_residual=is_residual,
                normalize_embeddings=normalize_embeddings,
                prediction_mode=args.prediction_mode,
                device=device,
            )
            seq_path = (output_dir / f"{fire_id}.npy").resolve()
            np.save(seq_path, exported.astype(np.float32, copy=False))
            exported_sequences.append(
                {
                    "sequence_id": fire_id,
                    "sequence_path": str(seq_path),
                    "num_frames": int(exported.shape[0]),
                    "num_predicted_frames": max(0, int(exported.shape[0]) - history),
                }
            )

    export_manifest = {
        "input_type": input_type,
        "input_dir": input_dir,
        "source_model_dir": str(model_dir.resolve()),
        "source_run_id": str(summary.get("run_id", "")),
        "source_checkpoint": str(checkpoint_path.resolve()),
        "source_variant": str(summary.get("variant", "")),
        "source_family": str(summary.get("family", "")),
        "source_timestamp": timestamp,
        "source_embeddings_model_slug": embeddings_model_slug,
        "prediction_mode": args.prediction_mode,
        "split": args.split,
        "history": history,
        "normalize_embeddings": normalize_embeddings,
        "sequences": exported_sequences,
    }
    (output_dir / "manifest.json").write_text(json.dumps(export_manifest, indent=2), encoding="utf-8")

    split_summary = {
        "train_ids": sorted(train_ids),
        "val_ids": sorted(val_ids),
        "holdout_ids": sorted(holdout_ids),
        "selected_ids": sorted(selected_ids),
    }
    (output_dir / "splits.json").write_text(json.dumps(split_summary, indent=2), encoding="utf-8")

    LOGGER.info("exported %d sequences to %s", len(exported_sequences), output_dir)


if __name__ == "__main__":
    main()
