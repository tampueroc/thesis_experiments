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
from wildfire.data.dataset import WildfireSequenceDataset
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

    def cfg_value(key: str, default: Any) -> Any:
        return cfg.get(key, default)

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
        "--history", type=int, default=int(cfg_value("history", 5)), help="History window length."
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=int(cfg_value("max_sequences", 5000)),
        help="Maximum number of manifest sequences to load.",
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
) -> float:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_items = 0
    for batch in maybe_tqdm(loader, enabled=show_progress, desc=desc):
        z_in = batch["z_in"].to(device)
        z_target = batch["z_target"].to(device)

        if training:
            if optimizer is None:
                raise RuntimeError("optimizer is required during training")
            optimizer.zero_grad(set_to_none=True)

        preds = model(z_in)
        loss = loss_fn(preds, z_target)

        if training:
            if optimizer is None:
                raise RuntimeError("optimizer is required during training")
            loss.backward()
            optimizer.step()

        batch_items = z_in.shape[0]
        total_loss += float(loss.item()) * batch_items
        total_items += int(batch_items)

    if total_items == 0:
        raise RuntimeError("dataloader yielded zero items")
    return total_loss / total_items


def main() -> None:
    args = parse_args()
    if args.history < 1:
        raise ValueError("history must be >= 1")

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

    train_ds = WildfireSequenceDataset(
        *select_sources(z_by_fire, w_by_fire, g_by_fire, train_ids),
        history=args.history,
        return_tensors=True,
    )
    val_ds = WildfireSequenceDataset(
        *select_sources(z_by_fire, w_by_fire, g_by_fire, val_ids),
        history=args.history,
        return_tensors=True,
    )
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(
            f"empty split after history={args.history}: train={len(train_ds)}, val={len(val_ds)}"
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
            "history": args.history,
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
            "timestamp": timestamp,
            "device": str(device),
        },
    )
    wandb_handler.watch_model(model)

    best_val = float("inf")
    best_epoch = -1
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            loss_fn,
            device,
            optimizer,
            show_progress=not args.no_progress,
            desc=f"train e{epoch:03d}",
        )
        with TORCH.no_grad():
            val_loss = run_epoch(
                model,
                val_loader,
                loss_fn,
                device,
                optimizer=None,
                show_progress=not args.no_progress,
                desc=f"val   e{epoch:03d}",
            )

        LOGGER.info(
            "[epoch %03d] train/mse=%.6f val/mse=%.6f (val/best_mse=%.6f)",
            epoch,
            train_loss,
            val_loss,
            best_val,
        )
        wandb_handler.log_metrics(
            {
                "meta/epoch": float(epoch),
                "train/mse": train_loss,
                "val/mse": val_loss,
                "val/best_mse": min(best_val, val_loss),
            },
            step=epoch,
        )
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            TORCH.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "epoch": epoch,
                    "val/mse": val_loss,
                },
                checkpoint_path,
            )

    state = TORCH.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    with TORCH.no_grad():
        train_mse = run_epoch(
            model,
            train_loader,
            loss_fn,
            device,
            optimizer=None,
            show_progress=not args.no_progress,
            desc="eval train",
        )
        val_mse = run_epoch(
            model,
            val_loader,
            loss_fn,
            device,
            optimizer=None,
            show_progress=not args.no_progress,
            desc="eval val",
        )
    summary = {
        "run_id": run_id,
        "config_path": str(args.config) if args.config is not None else "",
        "component": args.component,
        "family": args.family,
        "variant": args.variant,
        "timestamp": timestamp,
        "model_dir": str(model_dir),
        "output_root": str(args.output_dir),
        "output_dir": str(run_dir),
        "history": args.history,
        "splits": {
            "train_fire_ids": len(train_ids),
            "val_fire_ids": len(val_ids),
            "holdout_fire_ids": len(holdout_ids),
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
        },
        "best_epoch": best_epoch,
        "val/best_mse": best_val,
        "metrics": {
            "train/mse": train_mse,
            "val/mse": val_mse,
        },
        "device": str(device),
        "config": asdict(config),
        "checkpoint": str(checkpoint_path),
    }

    summary_path = run_dir / "metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    wandb_handler.log_metrics(
        {
            "train/mse": train_mse,
            "val/mse": val_mse,
            "val/best_mse": best_val,
            "meta/best_epoch": float(best_epoch),
        }
    )
    wandb_handler.log_summary(summary)
    wandb_handler.finish()

    LOGGER.info("checkpoint=%s", checkpoint_path)
    LOGGER.info("metrics=%s", summary_path)
    LOGGER.info(
        "final train/mse=%.6f val/mse=%.6f",
        train_mse,
        val_mse,
    )


if __name__ == "__main__":
    main()
