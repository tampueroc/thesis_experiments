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
from PIL import Image
import torch
from torch.utils.data import DataLoader

from wildfire.data.decoder_dataset import WildfireDecoderDataset, build_decoder_sources
from wildfire.data.real_data import choose_timestamp
from wildfire.logging.wandb_handler import WandbHandler
from wildfire.model_latent_decoder.conditional_unet_01 import (
    ConditionalUNetConfig,
    ConditionalUNetDecoder,
)

TORCH = cast(Any, torch)
LOGGER = logging.getLogger("wildfire.train.latent_decoder_conditional_unet")


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
    pre_parser.add_argument("--config", type=Path, default=None)
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
        description="Train conditional U-Net latent decoder on aligned wildfire frames and embeddings.",
    )
    parser.add_argument(
        "--embeddings-root",
        type=Path,
        default=Path(cfg_value("embeddings_root", "/home/tampuero/data/thesis_data/embeddings")),
    )
    parser.add_argument("--timestamp", default=cfg_value("timestamp", ""))
    parser.add_argument(
        "--input-type",
        choices=["fire_frames", "isochrones"],
        default=cfg_value("input_type", "fire_frames"),
    )
    parser.add_argument(
        "--embeddings-model-slug",
        default=cfg_value("embeddings_model_slug", cfg_value("model_slug", "facebook__dinov2-small")),
    )
    parser.add_argument("--component", default=cfg_value("component", "latent_decoder"))
    parser.add_argument("--family", default=cfg_value("family", "conditional_unet"))
    parser.add_argument("--variant", default=cfg_value("variant", "model_01"))
    parser.add_argument("--max-sequences", type=int, default=int(cfg_value("max_sequences", 0)))
    parser.add_argument("--train-ratio", type=float, default=float(cfg_value("train_ratio", 0.7)))
    parser.add_argument("--val-ratio", type=float, default=float(cfg_value("val_ratio", 0.15)))
    parser.add_argument("--seed", type=int, default=int(cfg_value("seed", 7)))
    parser.add_argument("--batch-size", type=int, default=int(cfg_value("batch_size", 8)))
    parser.add_argument("--epochs", type=int, default=int(cfg_value("epochs", 20)))
    parser.add_argument("--learning-rate", type=float, default=float(cfg_value("learning_rate", 1e-3)))
    parser.add_argument("--base-channels", type=int, default=int(cfg_value("base_channels", 32)))
    parser.add_argument("--num-pool-layers", type=int, default=int(cfg_value("num_pool_layers", 4)))
    parser.add_argument(
        "--conditioning-channels-per-embedding",
        type=int,
        default=int(cfg_value("conditioning_channels_per_embedding", 32)),
    )
    parser.add_argument(
        "--bottleneck-spatial-size",
        type=int,
        default=int(cfg_value("bottleneck_spatial_size", 25)),
    )
    parser.set_defaults(normalize_embeddings=bool(cfg_value("normalize_embeddings", False)))
    parser.add_argument("--normalize-embeddings", dest="normalize_embeddings", action="store_true")
    parser.add_argument("--no-normalize-embeddings", dest="normalize_embeddings", action="store_false")
    parser.add_argument(
        "--dice-loss-weight",
        type=float,
        default=float(cfg_value("dice_loss_weight", 1.0)),
    )
    parser.add_argument(
        "--recon-log-interval",
        type=int,
        default=int(cfg_value("recon_log_interval", 10)),
        help="Save qualitative reconstruction panels every N epochs. Use 0 to disable.",
    )
    parser.add_argument(
        "--recon-log-samples",
        type=int,
        default=int(cfg_value("recon_log_samples", 4)),
        help="Number of validation samples to include in each reconstruction panel.",
    )
    parser.add_argument("--num-workers", type=int, default=int(cfg_value("num_workers", 0)))
    parser.add_argument(
        "--device",
        default=cfg_value("device", "auto"),
        choices=["auto", "cpu", "cuda", "mps"],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(cfg_value("output_dir", "artifacts/wildfire")),
    )
    parser.add_argument("--wandb", action="store_true", default=bool(cfg_value("wandb", False)))
    parser.add_argument("--wandb-project", default=cfg_value("wandb_project", "latent_wildfire"))
    parser.add_argument("--wandb-entity", default=cfg_value("wandb_entity", ""))
    parser.add_argument("--wandb-run-name", default=cfg_value("wandb_run_name", ""))
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default=cfg_value("wandb_mode", "online"),
    )
    parser.add_argument("--wandb-tags", default=cfg_value("wandb_tags", ""))
    parser.add_argument("--no-progress", action="store_true", default=bool(cfg_value("no_progress", False)))
    parser.add_argument("--early-stop-metric", default=str(early_value("metric", "val/loss")))
    parser.add_argument(
        "--early-stop-mode",
        choices=["min", "max"],
        default=str(early_value("mode", "min")),
    )
    parser.add_argument("--early-stop-patience", type=int, default=int(early_value("patience", 12)))
    parser.add_argument("--early-stop-min-delta", type=float, default=float(early_value("min_delta", 1e-4)))
    parser.add_argument(
        "--early-stop-enabled",
        action="store_true",
        default=bool(early_value("enabled", False)),
    )
    parser.add_argument("--no-early-stop", action="store_false", dest="early_stop_enabled")
    return parser.parse_args()


def resolve_device(device_name: str) -> Any:
    if device_name == "auto":
        if TORCH.cuda.is_available():
            return TORCH.device("cuda")
        if hasattr(TORCH.backends, "mps") and TORCH.backends.mps.is_available():
            return TORCH.device("mps")
        return TORCH.device("cpu")
    return TORCH.device(device_name)


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


def make_run_id(component: str, family: str, variant: str, timestamp: str, seed: int) -> str:
    launch_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{component}-{family}-{variant}-{timestamp}-s{seed}-{launch_time}"


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

    n_train = max(1, int(n_total * train_ratio))
    n_val = max(1, int(n_total * val_ratio))
    if n_train + n_val >= n_total:
        n_val = max(1, n_total - n_train - 1)
        if n_train + n_val >= n_total:
            n_train = max(1, n_total - n_val - 1)
    if n_train + n_val >= n_total:
        raise ValueError("not enough sequences to keep holdout set separate")
    return (
        set(ordered_ids[:n_train]),
        set(ordered_ids[n_train : n_train + n_val]),
        set(ordered_ids[n_train + n_val :]),
    )


def select_sources(
    z_by_fire: dict[str, np.ndarray],
    frames_by_fire: dict[str, list[Path]],
    keys: set[str],
) -> tuple[dict[str, np.ndarray], dict[str, list[Path]]]:
    return (
        {k: z_by_fire[k] for k in keys},
        {k: frames_by_fire[k] for k in keys},
    )


def dice_coefficient_from_logits(logits: Any, target: Any, eps: float = 1e-6) -> Any:
    probs = TORCH.sigmoid(logits)
    probs_flat = probs.reshape(int(probs.shape[0]), -1)
    target_flat = target.reshape(int(target.shape[0]), -1)
    intersection = TORCH.sum(probs_flat * target_flat, dim=1)
    denom = TORCH.sum(probs_flat, dim=1) + TORCH.sum(target_flat, dim=1)
    return ((2.0 * intersection) + eps) / (denom + eps)


def iou_from_logits(logits: Any, target: Any, eps: float = 1e-6) -> Any:
    probs = TORCH.sigmoid(logits)
    probs_flat = probs.reshape(int(probs.shape[0]), -1)
    target_flat = target.reshape(int(target.shape[0]), -1)
    intersection = TORCH.sum(probs_flat * target_flat, dim=1)
    union = TORCH.sum(probs_flat, dim=1) + TORCH.sum(target_flat, dim=1) - intersection
    return (intersection + eps) / (union + eps)


def dice_loss_from_logits(logits: Any, target: Any) -> Any:
    return 1.0 - TORCH.mean(dice_coefficient_from_logits(logits, target))


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


def run_epoch(
    model: Any,
    loader: Any,
    bce_loss_fn: Any,
    device: Any,
    optimizer: Any | None,
    dice_loss_weight: float,
    show_progress: bool,
    desc: str,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_bce = 0.0
    total_dice_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_items = 0
    for batch in maybe_tqdm(loader, enabled=show_progress, desc=desc):
        prev_image = batch["prev_image"].to(device)
        prev_embedding = batch["prev_embedding"].to(device)
        target_embedding = batch["target_embedding"].to(device)
        target_image = batch["target_image"].to(device)

        if training:
            if optimizer is None:
                raise RuntimeError("optimizer is required during training")
            optimizer.zero_grad(set_to_none=True)

        logits = model(prev_image, prev_embedding, target_embedding)
        bce = bce_loss_fn(logits, target_image)
        dice_loss = dice_loss_from_logits(logits, target_image)
        loss = bce + (dice_loss_weight * dice_loss)
        dice = TORCH.mean(dice_coefficient_from_logits(logits, target_image))
        iou = TORCH.mean(iou_from_logits(logits, target_image))

        if training:
            if optimizer is None:
                raise RuntimeError("optimizer is required during training")
            loss.backward()
            optimizer.step()

        batch_items = int(prev_image.shape[0])
        total_loss += float(loss.item()) * batch_items
        total_bce += float(bce.item()) * batch_items
        total_dice_loss += float(dice_loss.item()) * batch_items
        total_dice += float(dice.item()) * batch_items
        total_iou += float(iou.item()) * batch_items
        total_items += batch_items

    if total_items == 0:
        raise RuntimeError("dataloader yielded zero items")
    return {
        "loss": total_loss / total_items,
        "bce": total_bce / total_items,
        "dice_loss": total_dice_loss / total_items,
        "dice": total_dice / total_items,
        "iou": total_iou / total_items,
    }


def tensor_chw_to_uint8_hwc(image: np.ndarray) -> np.ndarray:
    clipped = np.clip(image, 0.0, 1.0)
    hwc = np.transpose(clipped, (1, 2, 0))
    return np.rint(hwc * 255.0).astype(np.uint8, copy=False)


def make_reconstruction_panel(
    *,
    prev_images: list[np.ndarray],
    pred_images: list[np.ndarray],
    target_images: list[np.ndarray],
) -> np.ndarray:
    if not prev_images or not (len(prev_images) == len(pred_images) == len(target_images)):
        raise ValueError("reconstruction panel inputs must be non-empty and aligned")

    sample_h = int(prev_images[0].shape[1])
    sample_w = int(prev_images[0].shape[2])
    spacer = 4
    columns = 3
    rows = len(prev_images)
    panel_h = (rows * sample_h) + ((rows + 1) * spacer)
    panel_w = (columns * sample_w) + ((columns + 1) * spacer)
    panel = np.full((panel_h, panel_w, 3), 24, dtype=np.uint8)

    for row_idx, (prev_image, pred_image, target_image) in enumerate(
        zip(prev_images, pred_images, target_images, strict=True)
    ):
        top = spacer + (row_idx * (sample_h + spacer))
        left = spacer
        triplet = [prev_image, pred_image, target_image]
        for col_idx, image in enumerate(triplet):
            col_left = left + (col_idx * (sample_w + spacer))
            panel[top : top + sample_h, col_left : col_left + sample_w, :] = tensor_chw_to_uint8_hwc(image)
    return panel


def log_reconstruction_panel(
    *,
    model: Any,
    dataset: WildfireDecoderDataset,
    device: Any,
    run_dir: Path,
    split_name: str,
    tag: str,
    num_samples: int,
    wandb_handler: WandbHandler,
    step: int | None,
) -> Path | None:
    if num_samples <= 0 or len(dataset) == 0:
        return None

    sample_count = min(num_samples, len(dataset))
    prev_images: list[np.ndarray] = []
    pred_images: list[np.ndarray] = []
    target_images: list[np.ndarray] = []
    model.eval()
    with TORCH.no_grad():
        for sample_idx in range(sample_count):
            sample = dataset[sample_idx]
            prev_image_tensor = cast(Any, sample["prev_image"])
            prev_embedding_tensor = cast(Any, sample["prev_embedding"])
            target_embedding_tensor = cast(Any, sample["target_embedding"])
            target_image_tensor = cast(Any, sample["target_image"])
            prev_image = prev_image_tensor.unsqueeze(0).to(device)
            prev_embedding = prev_embedding_tensor.unsqueeze(0).to(device)
            target_embedding = target_embedding_tensor.unsqueeze(0).to(device)
            target_image = target_image_tensor.cpu().numpy()

            pred_logits = model(prev_image, prev_embedding, target_embedding)
            pred_image = TORCH.sigmoid(pred_logits).squeeze(0).detach().cpu().numpy()

            prev_images.append(prev_image_tensor.cpu().numpy())
            pred_images.append(pred_image)
            target_images.append(target_image)

    panel = make_reconstruction_panel(
        prev_images=prev_images,
        pred_images=pred_images,
        target_images=target_images,
    )
    recon_dir = run_dir / "reconstructions"
    recon_dir.mkdir(parents=True, exist_ok=True)
    output_path = recon_dir / f"{split_name}_{tag}.png"
    Image.fromarray(panel, mode="RGB").save(output_path)
    LOGGER.info("saved reconstruction panel: %s", output_path)

    wandb_image = wandb_handler.image(
        output_path,
        caption=f"{split_name} {tag} | columns=prev,pred,target",
    )
    if wandb_image is not None:
        wandb_handler.log_payload({f"{split_name}/recon_panel": wandb_image}, step=step)
    return output_path


def main() -> None:
    args = parse_args()
    timestamp = choose_timestamp(args.embeddings_root, args.timestamp)
    model_dir = args.embeddings_root / timestamp / args.input_type / args.embeddings_model_slug
    if not model_dir.exists():
        raise FileNotFoundError(f"embedding model directory not found: {model_dir}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    TORCH.manual_seed(args.seed)

    z_by_fire, frames_by_fire = build_decoder_sources(model_dir=model_dir)
    if args.max_sequences > 0:
        limited_ids = sorted(z_by_fire)[: args.max_sequences]
        z_by_fire = {fire_id: z_by_fire[fire_id] for fire_id in limited_ids}
        frames_by_fire = {fire_id: frames_by_fire[fire_id] for fire_id in limited_ids}

    train_ids, val_ids, holdout_ids = split_fire_ids(
        fire_ids=list(z_by_fire.keys()),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    train_ds = WildfireDecoderDataset(
        *select_sources(z_by_fire, frames_by_fire, train_ids),
        normalize_embeddings=args.normalize_embeddings,
        return_tensors=True,
    )
    val_ds = WildfireDecoderDataset(
        *select_sources(z_by_fire, frames_by_fire, val_ids),
        normalize_embeddings=args.normalize_embeddings,
        return_tensors=True,
    )
    holdout_ds = WildfireDecoderDataset(
        *select_sources(z_by_fire, frames_by_fire, holdout_ids),
        normalize_embeddings=args.normalize_embeddings,
        return_tensors=True,
    )
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(f"empty split: train={len(train_ds)} val={len(val_ds)}")

    train_loader = DataLoader(cast(Any, train_ds), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(cast(Any, val_ds), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    holdout_loader = DataLoader(
        cast(Any, holdout_ds),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    sample = train_ds[0]
    prev_image = np.asarray(sample["prev_image"])
    prev_embedding = np.asarray(sample["prev_embedding"])
    bottleneck_size = args.bottleneck_spatial_size
    expected_bottleneck = int(prev_image.shape[-1]) // (2 ** args.num_pool_layers)
    if bottleneck_size != expected_bottleneck:
        raise ValueError(
            f"bottleneck_spatial_size={bottleneck_size} does not match image_size={prev_image.shape[-1]} "
            f"and num_pool_layers={args.num_pool_layers}; expected {expected_bottleneck}"
        )

    config = ConditionalUNetConfig(
        image_channels=int(prev_image.shape[0]),
        embedding_dim=int(prev_embedding.shape[-1]),
        base_channels=args.base_channels,
        num_pool_layers=args.num_pool_layers,
        conditioning_channels_per_embedding=args.conditioning_channels_per_embedding,
        bottleneck_spatial_size=bottleneck_size,
    )
    model = ConditionalUNetDecoder(config)
    device = resolve_device(args.device)
    model = model.to(device)
    optimizer = TORCH.optim.Adam(model.parameters(), lr=args.learning_rate)
    bce_loss_fn = TORCH.nn.BCEWithLogitsLoss()

    run_id = make_run_id(args.component, args.family, args.variant, timestamp, args.seed)
    run_dir = args.output_dir / args.component / args.family / run_id
    setup_logging(run_dir)
    checkpoint_path = run_dir / "best_model.pt"
    LOGGER.info("run_id=%s", run_id)
    LOGGER.info("model_dir=%s", model_dir)
    LOGGER.info("output_dir=%s", run_dir)
    LOGGER.info("normalize_embeddings=%s", args.normalize_embeddings)
    LOGGER.info(
        "reconstruction logging interval=%d samples=%d",
        args.recon_log_interval,
        args.recon_log_samples,
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
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "base_channels": args.base_channels,
            "num_pool_layers": args.num_pool_layers,
            "conditioning_channels_per_embedding": args.conditioning_channels_per_embedding,
            "bottleneck_spatial_size": args.bottleneck_spatial_size,
            "dice_loss_weight": args.dice_loss_weight,
            "recon_log_interval": args.recon_log_interval,
            "recon_log_samples": args.recon_log_samples,
            "normalize_embeddings": args.normalize_embeddings,
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

    early_stopper = EarlyStopper(args.early_stop_mode, args.early_stop_patience, args.early_stop_min_delta)
    best_metric_value: float | None = None
    best_epoch = -1
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            bce_loss_fn,
            device,
            optimizer,
            args.dice_loss_weight,
            show_progress=not args.no_progress,
            desc=f"train e{epoch:03d}",
        )
        with TORCH.no_grad():
            val_metrics = run_epoch(
                model,
                val_loader,
                bce_loss_fn,
                device,
                optimizer=None,
                dice_loss_weight=args.dice_loss_weight,
                show_progress=not args.no_progress,
                desc=f"val   e{epoch:03d}",
            )
        epoch_metrics = {
            "train/loss": train_metrics["loss"],
            "train/bce": train_metrics["bce"],
            "train/dice_loss": train_metrics["dice_loss"],
            "train/dice": train_metrics["dice"],
            "train/iou": train_metrics["iou"],
            "val/loss": val_metrics["loss"],
            "val/bce": val_metrics["bce"],
            "val/dice_loss": val_metrics["dice_loss"],
            "val/dice": val_metrics["dice"],
            "val/iou": val_metrics["iou"],
        }
        monitor_value = float(epoch_metrics[args.early_stop_metric])
        improved, should_stop = early_stopper.update(monitor_value)
        LOGGER.info(
            "[epoch %03d] train/loss=%.6f val/loss=%.6f val/dice=%.6f val/iou=%.6f",
            epoch,
            train_metrics["loss"],
            val_metrics["loss"],
            val_metrics["dice"],
            val_metrics["iou"],
        )
        if best_metric_value is None or improved:
            best_metric_value = monitor_value
            best_epoch = epoch
            TORCH.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "epoch": epoch,
                    "monitor_metric": args.early_stop_metric,
                    "monitor_value": monitor_value,
                },
                checkpoint_path,
            )
        wandb_handler.log_metrics({"meta/epoch": float(epoch), **epoch_metrics}, step=epoch)
        if args.recon_log_interval > 0 and (epoch == 1 or epoch % args.recon_log_interval == 0):
            log_reconstruction_panel(
                model=model,
                dataset=val_ds,
                device=device,
                run_dir=run_dir,
                split_name="val",
                tag=f"epoch_{epoch:03d}",
                num_samples=args.recon_log_samples,
                wandb_handler=wandb_handler,
                step=epoch,
            )
        if args.early_stop_enabled and should_stop:
            LOGGER.info("early stopping at epoch=%d", epoch)
            break

    state = TORCH.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    with TORCH.no_grad():
        train_eval = run_epoch(
            model,
            train_loader,
            bce_loss_fn,
            device,
            optimizer=None,
            dice_loss_weight=args.dice_loss_weight,
            show_progress=not args.no_progress,
            desc="eval train",
        )
        val_eval = run_epoch(
            model,
            val_loader,
            bce_loss_fn,
            device,
            optimizer=None,
            dice_loss_weight=args.dice_loss_weight,
            show_progress=not args.no_progress,
            desc="eval val",
        )
        holdout_eval = run_epoch(
            model,
            holdout_loader,
            bce_loss_fn,
            device,
            optimizer=None,
            dice_loss_weight=args.dice_loss_weight,
            show_progress=not args.no_progress,
            desc="eval holdout",
        )
    final_val_panel = log_reconstruction_panel(
        model=model,
        dataset=val_ds,
        device=device,
        run_dir=run_dir,
        split_name="val",
        tag="final",
        num_samples=args.recon_log_samples,
        wandb_handler=wandb_handler,
        step=best_epoch if best_epoch > 0 else None,
    )
    final_holdout_panel = log_reconstruction_panel(
        model=model,
        dataset=holdout_ds,
        device=device,
        run_dir=run_dir,
        split_name="holdout",
        tag="final",
        num_samples=args.recon_log_samples,
        wandb_handler=wandb_handler,
        step=best_epoch if best_epoch > 0 else None,
    )

    summary = {
        "run_id": run_id,
        "component": args.component,
        "family": args.family,
        "variant": args.variant,
        "timestamp": timestamp,
        "model_dir": str(model_dir),
        "output_dir": str(run_dir),
        "best_epoch": best_epoch,
        "monitor_metric": args.early_stop_metric,
        "monitor_value_best": best_metric_value,
        "device": str(device),
        "splits": {
            "train_fire_ids": len(train_ids),
            "val_fire_ids": len(val_ids),
            "holdout_fire_ids": len(holdout_ids),
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "holdout_samples": len(holdout_ds),
        },
        "metrics": {
            "train/loss": train_eval["loss"],
            "train/dice": train_eval["dice"],
            "train/iou": train_eval["iou"],
            "val/loss": val_eval["loss"],
            "val/dice": val_eval["dice"],
            "val/iou": val_eval["iou"],
            "holdout/loss": holdout_eval["loss"],
            "holdout/dice": holdout_eval["dice"],
            "holdout/iou": holdout_eval["iou"],
        },
        "config": asdict(config),
        "checkpoint": str(checkpoint_path),
        "qualitative": {
            "val_recon_panel": str(final_val_panel) if final_val_panel is not None else "",
            "holdout_recon_panel": str(final_holdout_panel) if final_holdout_panel is not None else "",
        },
    }
    summary_path = run_dir / "metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    wandb_handler.log_summary(summary)
    wandb_handler.finish()


if __name__ == "__main__":
    main()
