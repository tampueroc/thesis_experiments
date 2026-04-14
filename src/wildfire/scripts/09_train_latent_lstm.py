from __future__ import annotations

import argparse
import json
import logging
import random
import tomllib
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import DataLoader
from wildfire.data.dataset import WildfireAugmentedSequenceDataset, WildfireSequenceDataset
from wildfire.data.real_data import build_sources, choose_timestamp
from wildfire.logging.wandb_handler import WandbHandler
from wildfire.model_latent_predictor.model_01 import LSTMConfig, LatentLSTMPredictor

TORCH = cast(Any, torch)
LOGGER = logging.getLogger("wildfire.train.model_01")


def load_config(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")
    with config_path.open("rb") as fh:
        raw = tomllib.load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a TOML table at top level: {config_path}")
    return raw


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional TOML config file. CLI flags override config values.",
    )
    pre_args, _ = pre_parser.parse_known_args()
    cfg = load_config(pre_args.config)
    early_cfg_raw = cfg.get("early_stopping", {})
    early_cfg = early_cfg_raw if isinstance(early_cfg_raw, dict) else {}

    def cfg_value(key: str, default: Any) -> Any:
        return cfg.get(key, default)

    def early_value(key: str, default: Any) -> Any:
        return early_cfg.get(key, cfg_value(f"early_stop_{key}", default))

    parser = argparse.ArgumentParser(
        parents=[pre_parser],
        description=(
            "Train LatentLSTMPredictor on sequence embeddings with fire_id-level "
            "train/val/test split."
        )
    )
    parser.add_argument(
        "--embeddings-root",
        type=Path,
        default=Path(cfg_value("embeddings_root", "/home/tampuero/data/thesis_data/embeddings")),
        help="Root folder containing timestamped embeddings outputs.",
    )
    parser.add_argument(
        "--landscape-dir",
        type=Path,
        default=Path(cfg_value("landscape_dir", "/home/tampuero/data/thesis_data/landscape")),
        help="Landscape folder containing WeatherHistory.csv, Weathers/, and indices.json.",
    )
    parser.add_argument(
        "--timestamp",
        default=cfg_value("timestamp", ""),
        help="Specific embedding timestamp folder. If omitted, uses latest lexicographic timestamp.",
    )
    parser.add_argument(
        "--input-type",
        choices=["fire_frames", "isochrones"],
        default=cfg_value("input_type", "fire_frames"),
        help="Embedding modality to train on.",
    )
    parser.add_argument(
        "--embeddings-model-slug",
        default=cfg_value("embeddings_model_slug", cfg_value("model_slug", "facebook__dinov2-small")),
        help="Embedding model slug under modality folder.",
    )
    parser.add_argument(
        "--model-slug",
        dest="embeddings_model_slug",
        default=argparse.SUPPRESS,
        help="Deprecated alias for --embeddings-model-slug.",
    )
    parser.add_argument(
        "--component",
        default=cfg_value("component", "latent_predictor"),
        help="Top-level model component (for example latent_predictor, latent_decoder).",
    )
    parser.add_argument(
        "--family",
        default=cfg_value("family", "lstm"),
        help="Architecture family (for example lstm, transformer, mlp).",
    )
    parser.add_argument(
        "--variant",
        default=cfg_value("variant", "model_01"),
        help="Model variant identifier.",
    )
    parser.add_argument(
        "--dataset-windowing",
        choices=["fixed", "variable", "sliding"],
        default=cfg_value("dataset_windowing", "fixed"),
        help="Dataset windowing mode.",
    )
    parser.add_argument(
        "--dataset-history",
        type=int,
        default=int(cfg_value("dataset_history", cfg_value("history", 5))),
        help="History window length for fixed mode.",
    )
    parser.add_argument(
        "--dataset-history-min",
        type=int,
        default=int(cfg_value("dataset_history_min", 2)),
        help="Minimum history length for variable/sliding mode.",
    )
    parser.add_argument(
        "--dataset-history-max",
        type=int,
        default=int(cfg_value("dataset_history_max", 10)),
        help="Maximum history length for variable/sliding mode.",
    )
    parser.add_argument(
        "--dataset-sampling",
        choices=["random_end_index", "enumerate_history_lengths"],
        default=cfg_value("dataset_sampling", "random_end_index"),
        help="Sampling variant name for dataset mode metadata.",
    )
    parser.add_argument(
        "--dataset-stride",
        type=int,
        default=int(cfg_value("dataset_stride", 1)),
        help="Dataset stride for sample index construction.",
    )
    parser.set_defaults(normalize_embeddings=bool(cfg_value("normalize_embeddings", False)))
    parser.add_argument(
        "--normalize-embeddings",
        dest="normalize_embeddings",
        action="store_true",
        help="L2-normalize each timestep embedding after loading dataset arrays.",
    )
    parser.add_argument(
        "--no-normalize-embeddings",
        dest="normalize_embeddings",
        action="store_false",
        help="Disable embedding normalization even if enabled in config.",
    )
    parser.add_argument(
        "--history", type=int, default=int(cfg_value("history", 5)), help="History window length."
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=int(cfg_value("max_sequences", 0)),
        help="Maximum number of manifest sequences to load. Use 0 for all sequences.",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=float(cfg_value("train_ratio", 0.7)), help="Train split ratio."
    )
    parser.add_argument(
        "--val-ratio", type=float, default=float(cfg_value("val_ratio", 0.15)), help="Validation split ratio."
    )
    parser.add_argument(
        "--seed", type=int, default=int(cfg_value("seed", 7)), help="Random seed for split and training."
    )
    parser.add_argument("--batch-size", type=int, default=int(cfg_value("batch_size", 32)), help="Batch size.")
    parser.add_argument("--epochs", type=int, default=int(cfg_value("epochs", 20)), help="Number of training epochs.")
    parser.add_argument(
        "--learning-rate", type=float, default=float(cfg_value("learning_rate", 1e-3)), help="Adam learning rate."
    )
    parser.add_argument("--hidden-dim", type=int, default=int(cfg_value("hidden_dim", 256)), help="LSTM hidden size.")
    parser.add_argument("--num-layers", type=int, default=int(cfg_value("num_layers", 2)), help="LSTM layer count.")
    parser.add_argument(
        "--dropout", type=float, default=float(cfg_value("dropout", 0.1)), help="LSTM dropout for stacked LSTM."
    )
    parser.add_argument(
        "--bidirectional",
        action="store_true",
        default=bool(cfg_value("bidirectional", False)),
        help="Use bidirectional LSTM.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=int(cfg_value("num_workers", 0)),
        help="Dataloader worker count.",
    )
    parser.add_argument(
        "--device",
        default=cfg_value("device", "auto"),
        choices=["auto", "cpu", "cuda", "mps"],
        help="Training device.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(cfg_value("output_dir", "artifacts/wildfire")),
        help="Directory to store best checkpoint and metrics.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=bool(cfg_value("wandb", False)),
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "--wandb-project",
        default=cfg_value("wandb_project", "latent_wildfire"),
        help="W&B project name.",
    )
    parser.add_argument(
        "--wandb-entity",
        default=cfg_value("wandb_entity", ""),
        help="W&B entity/team (optional).",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=cfg_value("wandb_run_name", ""),
        help="W&B run name. If empty, generated from timestamp and model slug.",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default=cfg_value("wandb_mode", "online"),
        help="W&B mode for run syncing.",
    )
    parser.add_argument(
        "--wandb-tags",
        default=cfg_value("wandb_tags", ""),
        help="Comma-separated W&B tags.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        default=bool(cfg_value("no_progress", False)),
        help="Disable tqdm progress bars.",
    )
    parser.add_argument(
        "--early-stop-metric",
        default=str(early_value("metric", "val/z_mse")),
        help="Metric key to monitor for checkpointing/early stopping.",
    )
    parser.add_argument(
        "--early-stop-mode",
        choices=["min", "max"],
        default=str(early_value("mode", "min")),
        help="Whether lower ('min') or higher ('max') monitored metric is better.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=int(early_value("patience", 12)),
        help="Number of non-improving epochs before stopping.",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=float(early_value("min_delta", 1e-4)),
        help="Minimum absolute improvement to count as better.",
    )
    parser.add_argument(
        "--early-stop-enabled",
        action="store_true",
        default=bool(early_value("enabled", False)),
        help="Enable early stopping.",
    )
    parser.add_argument(
        "--no-early-stop",
        action="store_false",
        dest="early_stop_enabled",
        help="Disable early stopping.",
    )
    return parser.parse_args()


def resolve_device(device_name: str) -> Any:
    if device_name == "auto":
        if TORCH.cuda.is_available():
            return TORCH.device("cuda")
        if hasattr(TORCH.backends, "mps") and TORCH.backends.mps.is_available():
            return TORCH.device("mps")
        return TORCH.device("cpu")
    return TORCH.device(device_name)


def make_run_id(
    component: str,
    family: str,
    variant: str,
    timestamp: str,
    seed: int,
) -> str:
    launch_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{component}-{family}-{variant}-{timestamp}-s{seed}-{launch_time}"


def setup_logging(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)


def maybe_tqdm(iterable: Any, *, enabled: bool, desc: str) -> Any:
    if not enabled:
        return iterable
    try:
        from tqdm import tqdm
    except Exception:
        LOGGER.warning("tqdm not available; continuing without progress bars")
        return iterable
    return tqdm(iterable, desc=desc, leave=True)


def resolve_history_settings(args: argparse.Namespace) -> tuple[int, int, int]:
    if args.dataset_windowing == "fixed":
        if args.dataset_history < 1:
            raise ValueError("dataset_history must be >= 1")
        return args.dataset_history, args.dataset_history, args.dataset_history

    if args.dataset_history_min < 1:
        raise ValueError("dataset_history_min must be >= 1")
    if args.dataset_history_max < args.dataset_history_min:
        raise ValueError("dataset_history_max must be >= dataset_history_min")
    return args.dataset_history_max, args.dataset_history_min, args.dataset_history_max


def dataset_variant_name(windowing: str, history_min: int, history_max: int) -> str:
    if windowing == "fixed":
        return f"fixed_h{history_max}"
    return f"variable_h{history_min}-{history_max}"


def apply_variable_history_inplace(
    z_in: Any,
    w_in: Any | None,
    *,
    enabled: bool,
    history_min: int,
    history_max: int,
) -> None:
    if not enabled:
        return
    if history_max <= history_min:
        return
    batch_size = int(z_in.shape[0])
    for i in range(batch_size):
        hist = random.randint(history_min, history_max)
        prefix_len = history_max - hist
        if prefix_len <= 0:
            continue
        z_anchor = z_in[i, prefix_len : prefix_len + 1, :]
        z_in[i, :prefix_len, :] = z_anchor
        if w_in is not None:
            w_anchor = w_in[i, prefix_len : prefix_len + 1, :]
            w_in[i, :prefix_len, :] = w_anchor


def is_better(current: float, best: float | None, mode: str, min_delta: float) -> bool:
    if not np.isfinite(current):
        return False
    if best is None:
        return True
    if mode == "min":
        return current < (best - min_delta)
    return current > (best + min_delta)


class EarlyStopper:
    def __init__(self, mode: str, patience: int, min_delta: float) -> None:
        if mode not in {"min", "max"}:
            raise ValueError("early-stop mode must be 'min' or 'max'")
        if patience < 1:
            raise ValueError("early-stop patience must be >= 1")
        if min_delta < 0.0:
            raise ValueError("early-stop min_delta must be >= 0")
        self._mode = mode
        self._patience = patience
        self._min_delta = min_delta
        self.best: float | None = None
        self.bad_epochs = 0

    def update(self, value: float) -> tuple[bool, bool]:
        improved = is_better(value, self.best, self._mode, self._min_delta)
        if improved:
            self.best = value
            self.bad_epochs = 0
            return True, False
        self.bad_epochs += 1
        return False, self.bad_epochs >= self._patience


def required_rollout_horizon(metric_name: str) -> int | None:
    if not metric_name.startswith("rollout/val/"):
        return None
    if "@" not in metric_name:
        raise ValueError(f"rollout metric must include @<horizon>: {metric_name}")
    try:
        return int(metric_name.rsplit("@", maxsplit=1)[1])
    except ValueError as exc:
        raise ValueError(f"invalid rollout horizon in metric: {metric_name}") from exc


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
    rng = random.Random(seed)
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


def select_sources(
    z_by_fire: dict[str, np.ndarray],
    w_by_fire: dict[str, np.ndarray],
    g_by_fire: dict[str, np.ndarray],
    keys: set[str],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    return (
        {k: z_by_fire[k] for k in keys},
        {k: w_by_fire[k] for k in keys},
        {k: g_by_fire[k] for k in keys},
    )


def run_epoch(
    model: LatentLSTMPredictor,
    loader: Any,
    loss_fn: Any,
    device: Any,
    optimizer: Any | None,
    show_progress: bool,
    desc: str,
    variable_windowing: bool,
    history_min: int,
    history_max: int,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)

    total_z_mse = 0.0
    total_z_delta_mse = 0.0
    total_z_cosine = 0.0
    total_items = 0
    grad_norm_sum = 0.0
    grad_norm_steps = 0
    for batch in maybe_tqdm(loader, enabled=show_progress, desc=desc):
        z_in = batch["z_in"].to(device)
        w_in = batch["w_in"].to(device) if "w_in" in batch else None
        z_target = batch["z_target"].to(device)
        apply_variable_history_inplace(
            z_in,
            w_in,
            enabled=variable_windowing,
            history_min=history_min,
            history_max=history_max,
        )

        if training:
            if optimizer is None:
                raise RuntimeError("optimizer is required during training")
            optimizer.zero_grad(set_to_none=True)

        preds = model(z_in)
        z_mse = loss_fn(preds, z_target)
        z_prev = z_in[:, -1, :]
        z_delta_pred = preds - z_prev
        z_delta_true = z_target - z_prev
        z_delta_mse = loss_fn(z_delta_pred, z_delta_true)
        cosine = TORCH.nn.functional.cosine_similarity(preds, z_target, dim=1)
        z_cosine = TORCH.mean(cosine)
        loss = z_mse

        if training:
            if optimizer is None:
                raise RuntimeError("optimizer is required during training")
            loss.backward()
            grad_norm_sq = 0.0
            for param in model.parameters():
                grad = param.grad
                if grad is None:
                    continue
                grad_norm_sq += float(grad.detach().pow(2).sum().item())
            grad_norm_sum += grad_norm_sq**0.5
            grad_norm_steps += 1
            optimizer.step()

        batch_items = z_in.shape[0]
        total_z_mse += float(z_mse.item()) * batch_items
        total_z_delta_mse += float(z_delta_mse.item()) * batch_items
        total_z_cosine += float(z_cosine.item()) * batch_items
        total_items += int(batch_items)

    if total_items == 0:
        raise RuntimeError("dataloader yielded zero items")
    metrics = {
        "z_mse": total_z_mse / total_items,
        "z_delta_mse": total_z_delta_mse / total_items,
        "z_cosine": total_z_cosine / total_items,
    }
    if training:
        metrics["grad_norm"] = grad_norm_sum / max(1, grad_norm_steps)
    return metrics


def rollout_metrics_at_horizon(
    model: LatentLSTMPredictor,
    z_by_fire: dict[str, np.ndarray],
    history: int,
    horizon: int,
    device: Any,
    normalize_embeddings: bool = False,
) -> dict[str, float]:
    model.eval()
    total_mse = 0.0
    total_norm = 0.0
    total_cosine = 0.0
    total_count = 0
    with TORCH.no_grad():
        for z in z_by_fire.values():
            if z.ndim != 2:
                continue
            timesteps = int(z.shape[0])
            if timesteps < history + horizon:
                continue
            for target_t in range(history, timesteps - horizon + 1):
                window = z[target_t - history : target_t]
                window_t = TORCH.tensor(window, dtype=TORCH.float32, device=device).unsqueeze(0)
                pred = None
                for _step in range(horizon):
                    pred = model(window_t)
                    if normalize_embeddings:
                        pred = pred / TORCH.norm(pred, dim=1, keepdim=True).clamp_min(1e-8)
                    window_t = TORCH.cat([window_t[:, 1:, :], pred.unsqueeze(1)], dim=1)
                if pred is None:
                    continue
                gt = TORCH.tensor(z[target_t + horizon - 1], dtype=TORCH.float32, device=device).unsqueeze(0)
                mse = TORCH.mean((pred - gt) ** 2)
                norm = TORCH.norm(pred, dim=1).mean()
                cosine = TORCH.nn.functional.cosine_similarity(pred, gt, dim=1).mean()
                total_mse += float(mse.item())
                total_norm += float(norm.item())
                total_cosine += float(cosine.item())
                total_count += 1
    if total_count == 0:
        return {
            f"rollout/val/z_mse@{horizon}": float("nan"),
            f"rollout/val/z_norm_mean@{horizon}": float("nan"),
            f"rollout/val/z_cosine@{horizon}": float("nan"),
        }
    return {
        f"rollout/val/z_mse@{horizon}": total_mse / total_count,
        f"rollout/val/z_norm_mean@{horizon}": total_norm / total_count,
        f"rollout/val/z_cosine@{horizon}": total_cosine / total_count,
    }


def main() -> None:
    args = parse_args()
    history_for_dataset, history_min, history_max = resolve_history_settings(args)
    variant_name = dataset_variant_name(args.dataset_windowing, history_min, history_max)
    use_augmented_index = args.dataset_sampling == "enumerate_history_lengths"

    timestamp = choose_timestamp(args.embeddings_root, args.timestamp)
    model_dir = args.embeddings_root / timestamp / args.input_type / args.embeddings_model_slug
    if not model_dir.exists():
        raise FileNotFoundError(f"embedding model directory not found: {model_dir}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    TORCH.manual_seed(args.seed)

    z_by_fire, w_by_fire, g_by_fire = build_sources(
        model_dir=model_dir,
        landscape_dir=args.landscape_dir,
        max_sequences=args.max_sequences,
    )

    train_ids, val_ids, holdout_ids = split_fire_ids(
        fire_ids=list(z_by_fire.keys()),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    train_sources = select_sources(z_by_fire, w_by_fire, g_by_fire, train_ids)
    val_sources = select_sources(z_by_fire, w_by_fire, g_by_fire, val_ids)
    holdout_sources = select_sources(z_by_fire, w_by_fire, g_by_fire, holdout_ids)
    rollout_val_z_by_fire = val_sources[0]
    if args.normalize_embeddings:
        rollout_val_z_by_fire = WildfireSequenceDataset._normalize_per_fire_embeddings(rollout_val_z_by_fire)

    if use_augmented_index:
        train_ds = WildfireAugmentedSequenceDataset(
            *train_sources,
            history_min=history_min,
            history_max=history_max,
            stride=args.dataset_stride,
            normalize_embeddings=args.normalize_embeddings,
            return_tensors=True,
        )
        val_ds = WildfireAugmentedSequenceDataset(
            *val_sources,
            history_min=history_min,
            history_max=history_max,
            stride=args.dataset_stride,
            normalize_embeddings=args.normalize_embeddings,
            return_tensors=True,
        )
        holdout_ds = WildfireAugmentedSequenceDataset(
            *holdout_sources,
            history_min=history_min,
            history_max=history_max,
            stride=args.dataset_stride,
            normalize_embeddings=args.normalize_embeddings,
            return_tensors=True,
        )
    else:
        train_ds = WildfireSequenceDataset(
            *train_sources,
            history=history_for_dataset,
            stride=args.dataset_stride,
            normalize_embeddings=args.normalize_embeddings,
            return_tensors=True,
        )
        val_ds = WildfireSequenceDataset(
            *val_sources,
            history=history_for_dataset,
            stride=args.dataset_stride,
            normalize_embeddings=args.normalize_embeddings,
            return_tensors=True,
        )
        holdout_ds = WildfireSequenceDataset(
            *holdout_sources,
            history=history_for_dataset,
            stride=args.dataset_stride,
            normalize_embeddings=args.normalize_embeddings,
            return_tensors=True,
        )
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(
            f"empty split after history={history_for_dataset}: train={len(train_ds)}, val={len(val_ds)}"
        )

    train_loader = DataLoader(
        cast(Any, train_ds),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        cast(Any, val_ds),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    sample = train_ds[0]
    z_in = np.asarray(sample["z_in"])
    z_target = np.asarray(sample["z_target"])
    config = LSTMConfig(
        input_dim=int(z_in.shape[-1]),
        hidden_dim=args.hidden_dim,
        output_dim=int(z_target.shape[-1]),
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        batch_first=True,
    )

    device = resolve_device(args.device)
    model = LatentLSTMPredictor(config).to(device)
    optimizer = TORCH.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = TORCH.nn.MSELoss()

    run_id = make_run_id(
        component=args.component,
        family=args.family,
        variant=args.variant,
        timestamp=timestamp,
        seed=args.seed,
    )
    run_dir = args.output_dir / args.component / args.family / run_id
    setup_logging(run_dir)
    checkpoint_path = run_dir / "best_model.pt"
    LOGGER.info("run_id=%s", run_id)
    LOGGER.info("config_path=%s", args.config if args.config is not None else "")
    LOGGER.info("model_dir=%s", model_dir)
    LOGGER.info("output_dir=%s", run_dir)
    LOGGER.info(
        "dataset variant=%s windowing=%s history_min=%d history_max=%d stride=%d sampling=%s normalize_embeddings=%s",
        variant_name,
        args.dataset_windowing,
        history_min,
        history_max,
        args.dataset_stride,
        args.dataset_sampling,
        args.normalize_embeddings,
    )
    LOGGER.info(
        "early_stopping enabled=%s metric=%s mode=%s patience=%d min_delta=%g",
        args.early_stop_enabled,
        args.early_stop_metric,
        args.early_stop_mode,
        args.early_stop_patience,
        args.early_stop_min_delta,
    )

    wandb_run_name = args.wandb_run_name if args.wandb_run_name else run_id
    wandb_tags = [x.strip() for x in args.wandb_tags.split(",") if x.strip()]
    if not wandb_tags:
        wandb_tags = ["wildfire", args.component, args.family, args.variant]
    wandb_handler = WandbHandler(
        enabled=args.wandb and args.wandb_mode != "disabled",
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=wandb_run_name,
        output_dir=run_dir,
        tags=wandb_tags,
        mode=args.wandb_mode,
        config={
            "history": history_for_dataset,
            "dataset/windowing": args.dataset_windowing,
            "dataset/history": history_for_dataset,
            "dataset/history_min": history_min,
            "dataset/history_max": history_max,
            "dataset/sampling": args.dataset_sampling,
            "dataset/stride": args.dataset_stride,
            "dataset/normalize_embeddings": args.normalize_embeddings,
            "dataset_variant": variant_name,
            "data/windowing_mode": args.dataset_windowing,
            "data/history_min": history_min,
            "data/history_max": history_max,
            "max_sequences": args.max_sequences,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "holdout_ratio": max(0.0, 1.0 - args.train_ratio - args.val_ratio),
            "seed": args.seed,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "bidirectional": args.bidirectional,
            "input_type": args.input_type,
            "embeddings_model_slug": args.embeddings_model_slug,
            "component": args.component,
            "family": args.family,
            "variant": args.variant,
            "early_stop_enabled": args.early_stop_enabled,
            "early_stop_metric": args.early_stop_metric,
            "early_stop_mode": args.early_stop_mode,
            "early_stop_patience": args.early_stop_patience,
            "early_stop_min_delta": args.early_stop_min_delta,
            "timestamp": timestamp,
            "device": str(device),
        },
    )
    wandb_handler.watch_model(model)

    early_stopper = EarlyStopper(
        mode=args.early_stop_mode,
        patience=args.early_stop_patience,
        min_delta=args.early_stop_min_delta,
    )
    monitor_metric = args.early_stop_metric
    monitor_rollout_horizon = required_rollout_horizon(monitor_metric)
    best_metric_value: float | None = None
    best_epoch = -1
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            loss_fn,
            device,
            optimizer,
            show_progress=not args.no_progress,
            desc=f"train e{epoch:03d}",
            variable_windowing=(args.dataset_windowing != "fixed" and not use_augmented_index),
            history_min=history_min,
            history_max=history_max,
        )
        with TORCH.no_grad():
            val_metrics = run_epoch(
                model,
                val_loader,
                loss_fn,
                device,
                optimizer=None,
                show_progress=not args.no_progress,
                desc=f"val   e{epoch:03d}",
                variable_windowing=False,
                history_min=history_min,
                history_max=history_max,
            )

        epoch_metrics: dict[str, float] = {
            "train/z_mse": train_metrics["z_mse"],
            "train/z_delta_mse": train_metrics["z_delta_mse"],
            "train/grad_norm": train_metrics.get("grad_norm", float("nan")),
            "val/z_mse": val_metrics["z_mse"],
            "val/z_delta_mse": val_metrics["z_delta_mse"],
            "val/z_cosine": val_metrics["z_cosine"],
        }
        if monitor_rollout_horizon is not None:
            epoch_metrics.update(
                rollout_metrics_at_horizon(
                    model,
                    z_by_fire=rollout_val_z_by_fire,
                    history=history_for_dataset,
                    horizon=monitor_rollout_horizon,
                    device=device,
                    normalize_embeddings=args.normalize_embeddings,
                )
            )
        if monitor_metric not in epoch_metrics:
            raise KeyError(f"early-stop metric '{monitor_metric}' is not available in epoch metrics")
        monitor_value = float(epoch_metrics[monitor_metric])

        LOGGER.info(
            "[epoch %03d] train/z_mse=%.6f train/z_delta_mse=%.6f train/grad_norm=%.6f val/z_mse=%.6f val/z_delta_mse=%.6f val/z_cosine=%.6f (monitor/best=%.6f)",
            epoch,
            train_metrics["z_mse"],
            train_metrics["z_delta_mse"],
            train_metrics.get("grad_norm", float("nan")),
            val_metrics["z_mse"],
            val_metrics["z_delta_mse"],
            val_metrics["z_cosine"],
            best_metric_value if best_metric_value is not None else float("nan"),
        )
        improved, should_stop = early_stopper.update(monitor_value)
        if best_metric_value is None or improved:
            best_metric_value = monitor_value
            best_epoch = epoch
            TORCH.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "epoch": epoch,
                    "monitor_metric": monitor_metric,
                    "monitor_value": monitor_value,
                },
                checkpoint_path,
            )
        epoch_log = {
            "meta/epoch": float(epoch),
            "meta/monitor_value": monitor_value,
            "meta/monitor_best": best_metric_value if best_metric_value is not None else float("nan"),
            **epoch_metrics,
        }
        wandb_handler.log_metrics(epoch_log, step=epoch)
        if args.early_stop_enabled and should_stop:
            LOGGER.info(
                "early stopping at epoch=%d monitor=%s value=%.6f best=%.6f patience=%d",
                epoch,
                monitor_metric,
                monitor_value,
                best_metric_value if best_metric_value is not None else float("nan"),
                args.early_stop_patience,
            )
            break

    state = TORCH.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    with TORCH.no_grad():
        train_eval = run_epoch(
            model,
            train_loader,
            loss_fn,
            device,
            optimizer=None,
            show_progress=not args.no_progress,
            desc="eval train",
            variable_windowing=False,
            history_min=history_min,
            history_max=history_max,
        )
        val_eval = run_epoch(
            model,
            val_loader,
            loss_fn,
            device,
            optimizer=None,
            show_progress=not args.no_progress,
            desc="eval val",
            variable_windowing=False,
            history_min=history_min,
            history_max=history_max,
        )
    rollout_val = rollout_metrics_at_horizon(
        model,
        z_by_fire=rollout_val_z_by_fire,
        history=history_for_dataset,
        horizon=10,
        device=device,
        normalize_embeddings=args.normalize_embeddings,
    )
    summary = {
        "run_id": run_id,
        "config_path": str(args.config) if args.config is not None else "",
        "component": args.component,
        "family": args.family,
        "variant": args.variant,
        "dataset_variant": variant_name,
        "dataset/windowing": args.dataset_windowing,
        "dataset/history": history_for_dataset,
        "dataset/history_min": history_min,
        "dataset/history_max": history_max,
        "dataset/sampling": args.dataset_sampling,
        "dataset/stride": args.dataset_stride,
        "dataset/normalize_embeddings": args.normalize_embeddings,
        "timestamp": timestamp,
        "model_dir": str(model_dir),
        "output_root": str(args.output_dir),
        "output_dir": str(run_dir),
        "history": history_for_dataset,
        "splits": {
            "train_fire_ids": len(train_ids),
            "val_fire_ids": len(val_ids),
            "holdout_fire_ids": len(holdout_ids),
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "holdout_samples": len(holdout_ds),
        },
        "best_epoch": best_epoch,
        "monitor_metric": monitor_metric,
        "monitor_value_best": best_metric_value,
        "metrics": {
            "train/z_mse": train_eval["z_mse"],
            "train/z_delta_mse": train_eval["z_delta_mse"],
            "val/z_mse": val_eval["z_mse"],
            "val/z_delta_mse": val_eval["z_delta_mse"],
            "val/z_cosine": val_eval["z_cosine"],
            **rollout_val,
        },
        "device": str(device),
        "config": asdict(config),
        "checkpoint": str(checkpoint_path),
    }

    summary_path = run_dir / "metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    wandb_handler.log_metrics(
        {
            "train/z_mse": train_eval["z_mse"],
            "train/z_delta_mse": train_eval["z_delta_mse"],
            "val/z_mse": val_eval["z_mse"],
            "val/z_delta_mse": val_eval["z_delta_mse"],
            "val/z_cosine": val_eval["z_cosine"],
            "meta/monitor_best": best_metric_value if best_metric_value is not None else float("nan"),
            "meta/best_epoch": float(best_epoch),
            **rollout_val,
        }
    )
    wandb_handler.log_summary(summary)
    wandb_handler.finish()

    LOGGER.info("checkpoint=%s", checkpoint_path)
    LOGGER.info("metrics=%s", summary_path)
    LOGGER.info(
        "final train/z_mse=%.6f train/z_delta_mse=%.6f val/z_mse=%.6f val/z_delta_mse=%.6f val/z_cosine=%.6f",
        train_eval["z_mse"],
        train_eval["z_delta_mse"],
        val_eval["z_mse"],
        val_eval["z_delta_mse"],
        val_eval["z_cosine"],
    )


if __name__ == "__main__":
    main()
