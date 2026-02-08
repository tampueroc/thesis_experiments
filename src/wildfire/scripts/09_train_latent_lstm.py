from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train LatentLSTMPredictor on sequence embeddings with fire_id-level "
            "train/val/test split."
        )
    )
    parser.add_argument(
        "--embeddings-root",
        type=Path,
        default=Path("/home/tampuero/data/thesis_data/embeddings"),
        help="Root folder containing timestamped embeddings outputs.",
    )
    parser.add_argument(
        "--landscape-dir",
        type=Path,
        default=Path("/home/tampuero/data/thesis_data/landscape"),
        help="Landscape folder containing WeatherHistory.csv, Weathers/, and indices.json.",
    )
    parser.add_argument(
        "--timestamp",
        default="",
        help="Specific embedding timestamp folder. If omitted, uses latest lexicographic timestamp.",
    )
    parser.add_argument(
        "--input-type",
        choices=["fire_frames", "isochrones"],
        default="fire_frames",
        help="Embedding modality to train on.",
    )
    parser.add_argument(
        "--model-slug",
        default="facebook__dinov2-small",
        help="Embedding model slug under modality folder.",
    )
    parser.add_argument("--history", type=int, default=5, help="History window length.")
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=5000,
        help="Maximum number of manifest sequences to load.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for split and training.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--hidden-dim", type=int, default=256, help="LSTM hidden size.")
    parser.add_argument("--num-layers", type=int, default=2, help="LSTM layer count.")
    parser.add_argument("--dropout", type=float, default=0.1, help="LSTM dropout for stacked LSTM.")
    parser.add_argument(
        "--bidirectional",
        action="store_true",
        help="Use bidirectional LSTM.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Dataloader worker count.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Training device.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/wildfire/model_01"),
        help="Directory to store best checkpoint and metrics.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "--wandb-project",
        default="wildfire-latent-lstm",
        help="W&B project name.",
    )
    parser.add_argument(
        "--wandb-entity",
        default="",
        help="W&B entity/team (optional).",
    )
    parser.add_argument(
        "--wandb-run-name",
        default="",
        help="W&B run name. If empty, generated from timestamp and model slug.",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="online",
        help="W&B mode for run syncing.",
    )
    parser.add_argument(
        "--wandb-tags",
        default="wildfire,lstm,model_01",
        help="Comma-separated W&B tags.",
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


def split_fire_ids(
    fire_ids: list[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[set[str], set[str], set[str]]:
    if min(train_ratio, val_ratio, test_ratio) <= 0.0:
        raise ValueError("train/val/test ratios must all be > 0")
    ratio_sum = train_ratio + val_ratio + test_ratio
    if not np.isclose(ratio_sum, 1.0, atol=1e-6):
        raise ValueError(f"train+val+test must sum to 1.0, got {ratio_sum:.6f}")

    ordered_ids = sorted(fire_ids)
    rng = random.Random(seed)
    rng.shuffle(ordered_ids)

    n_total = len(ordered_ids)
    if n_total < 3:
        raise ValueError(f"need at least 3 sequences to build all splits, got {n_total}")

    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    if n_train == 0:
        n_train = 1
    if n_val == 0:
        n_val = 1
    n_test = n_total - n_train - n_val
    if n_test <= 0:
        n_test = 1
        if n_train > n_val:
            n_train -= 1
        else:
            n_val -= 1

    train_ids = set(ordered_ids[:n_train])
    val_ids = set(ordered_ids[n_train : n_train + n_val])
    test_ids = set(ordered_ids[n_train + n_val :])
    return train_ids, val_ids, test_ids


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
) -> float:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_items = 0
    for batch in loader:
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
    model_dir = args.embeddings_root / timestamp / args.input_type / args.model_slug
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

    train_ids, val_ids, test_ids = split_fire_ids(
        fire_ids=list(z_by_fire.keys()),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
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
    test_ds = WildfireSequenceDataset(
        *select_sources(z_by_fire, w_by_fire, g_by_fire, test_ids),
        history=args.history,
        return_tensors=True,
    )

    if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
        raise RuntimeError(
            f"empty split after history={args.history}: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
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
    test_loader = DataLoader(
        cast(Any, test_ds),
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best_model.pt"

    wandb_run_name = (
        args.wandb_run_name
        if args.wandb_run_name
        else f"{timestamp}-{args.input_type}-{args.model_slug}-h{args.history}"
    )
    wandb_tags = [x.strip() for x in args.wandb_tags.split(",") if x.strip()]
    wandb_handler = WandbHandler(
        enabled=args.wandb and args.wandb_mode != "disabled",
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=wandb_run_name,
        output_dir=args.output_dir,
        tags=wandb_tags,
        mode=args.wandb_mode,
        config={
            "history": args.history,
            "max_sequences": args.max_sequences,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "bidirectional": args.bidirectional,
            "input_type": args.input_type,
            "model_slug": args.model_slug,
            "timestamp": timestamp,
            "device": str(device),
        },
    )
    wandb_handler.watch_model(model)

    best_val = float("inf")
    best_epoch = -1
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, loss_fn, device, optimizer)
        with TORCH.no_grad():
            val_loss = run_epoch(model, val_loader, loss_fn, device, optimizer=None)

        print(
            f"[epoch {epoch:03d}] train_mse={train_loss:.6f} val_mse={val_loss:.6f} "
            f"(best_val={best_val:.6f})"
        )
        wandb_handler.log_metrics(
            {
                "epoch": float(epoch),
                "train_mse": train_loss,
                "val_mse": val_loss,
                "best_val_mse": min(best_val, val_loss),
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
                    "val_mse": val_loss,
                },
                checkpoint_path,
            )

    state = TORCH.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    with TORCH.no_grad():
        train_mse = run_epoch(model, train_loader, loss_fn, device, optimizer=None)
        val_mse = run_epoch(model, val_loader, loss_fn, device, optimizer=None)
        test_mse = run_epoch(model, test_loader, loss_fn, device, optimizer=None)

    summary = {
        "timestamp": timestamp,
        "model_dir": str(model_dir),
        "history": args.history,
        "splits": {
            "train_fire_ids": len(train_ids),
            "val_fire_ids": len(val_ids),
            "test_fire_ids": len(test_ids),
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "test_samples": len(test_ds),
        },
        "best_epoch": best_epoch,
        "best_val_mse": best_val,
        "metrics": {
            "train_mse": train_mse,
            "val_mse": val_mse,
            "test_mse": test_mse,
        },
        "device": str(device),
        "config": asdict(config),
        "checkpoint": str(checkpoint_path),
    }

    summary_path = args.output_dir / "metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    wandb_handler.log_metrics(
        {
            "final_train_mse": train_mse,
            "final_val_mse": val_mse,
            "final_test_mse": test_mse,
            "best_epoch": float(best_epoch),
            "best_val_mse": best_val,
        }
    )
    wandb_handler.log_summary(summary)
    wandb_handler.finish()

    print(f"[ok] checkpoint={checkpoint_path}")
    print(f"[ok] metrics={summary_path}")
    print(
        f"[ok] final train_mse={train_mse:.6f} val_mse={val_mse:.6f} "
        f"test_mse={test_mse:.6f}"
    )


if __name__ == "__main__":
    main()
