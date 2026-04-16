from __future__ import annotations

import argparse
import json
import logging
import tomllib
from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from wildfire.data.decoder_dataset import WildfireDecoderDataset, build_decoder_sources
from wildfire.logging.wandb_handler import WandbHandler
from wildfire.model_latent_decoder.conditional_unet_binary_01 import (
    BinaryMaskConditionalUNetConfig,
    BinaryMaskConditionalUNetDecoder,
)

TORCH = cast(Any, torch)
LOGGER = logging.getLogger("wildfire.eval.latent_decoder_conditional_unet_binary_predicted")


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

    def cfg_value(key: str, default: Any) -> Any:
        return cfg.get(key, default)

    parser = argparse.ArgumentParser(
        parents=[pre_parser],
        description="Evaluate a trained binary decoder checkpoint using predicted latent embeddings.",
    )
    parser.add_argument("--decoder-run-dir", type=Path, default=Path(cfg_value("decoder_run_dir", "")))
    parser.add_argument("--predicted-embeddings-dir", type=Path, default=Path(cfg_value("predicted_embeddings_dir", "")))
    parser.add_argument("--component", default=cfg_value("component", "latent_decoder_eval"))
    parser.add_argument("--family", default=cfg_value("family", "conditional_unet_binary_predicted"))
    parser.add_argument("--variant", default=cfg_value("variant", "model_01"))
    parser.add_argument("--max-sequences", type=int, default=int(cfg_value("max_sequences", 0)))
    parser.add_argument("--batch-size", type=int, default=int(cfg_value("batch_size", 8)))
    parser.add_argument("--mask-threshold", type=float, default=float(cfg_value("mask_threshold", 0.5)))
    parser.add_argument("--hard-threshold", type=float, default=float(cfg_value("hard_threshold", 0.4)))
    parser.add_argument("--recon-log-samples", type=int, default=int(cfg_value("recon_log_samples", 4)))
    parser.add_argument("--num-workers", type=int, default=int(cfg_value("num_workers", 0)))
    parser.add_argument("--device", default=cfg_value("device", "auto"), choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--output-dir", type=Path, default=Path(cfg_value("output_dir", "artifacts/wildfire")))
    parser.add_argument("--wandb", action="store_true", default=bool(cfg_value("wandb", False)))
    parser.add_argument("--wandb-project", default=cfg_value("wandb_project", "latent_wildfire"))
    parser.add_argument("--wandb-entity", default=cfg_value("wandb_entity", ""))
    parser.add_argument("--wandb-run-name", default=cfg_value("wandb_run_name", ""))
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=cfg_value("wandb_mode", "online"))
    parser.add_argument("--wandb-tags", default=cfg_value("wandb_tags", ""))
    parser.add_argument("--no-progress", action="store_true", default=bool(cfg_value("no_progress", False)))
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
    log_path = run_dir / "eval.log"
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


def make_run_id(component: str, family: str, variant: str, decoder_run_name: str, predictor_run_id: str) -> str:
    return f"{component}-{family}-{variant}-{decoder_run_name}-{predictor_run_id}"


def soft_dice_coefficient_from_logits(logits: Any, target: Any, eps: float = 1e-6) -> Any:
    probs = TORCH.sigmoid(logits)
    probs_flat = probs.reshape(int(probs.shape[0]), -1)
    target_flat = target.reshape(int(target.shape[0]), -1)
    intersection = TORCH.sum(probs_flat * target_flat, dim=1)
    denom = TORCH.sum(probs_flat, dim=1) + TORCH.sum(target_flat, dim=1)
    return ((2.0 * intersection) + eps) / (denom + eps)


def soft_iou_from_logits(logits: Any, target: Any, eps: float = 1e-6) -> Any:
    probs = TORCH.sigmoid(logits)
    probs_flat = probs.reshape(int(probs.shape[0]), -1)
    target_flat = target.reshape(int(target.shape[0]), -1)
    intersection = TORCH.sum(probs_flat * target_flat, dim=1)
    union = TORCH.sum(probs_flat, dim=1) + TORCH.sum(target_flat, dim=1) - intersection
    return (intersection + eps) / (union + eps)


def hard_dice_coefficient_from_logits(logits: Any, target: Any, threshold: float, eps: float = 1e-6) -> Any:
    probs = TORCH.sigmoid(logits)
    preds = (probs >= threshold).to(dtype=target.dtype)
    preds_flat = preds.reshape(int(preds.shape[0]), -1)
    target_flat = target.reshape(int(target.shape[0]), -1)
    intersection = TORCH.sum(preds_flat * target_flat, dim=1)
    denom = TORCH.sum(preds_flat, dim=1) + TORCH.sum(target_flat, dim=1)
    return ((2.0 * intersection) + eps) / (denom + eps)


def hard_iou_from_logits(logits: Any, target: Any, threshold: float, eps: float = 1e-6) -> Any:
    probs = TORCH.sigmoid(logits)
    preds = (probs >= threshold).to(dtype=target.dtype)
    preds_flat = preds.reshape(int(preds.shape[0]), -1)
    target_flat = target.reshape(int(target.shape[0]), -1)
    intersection = TORCH.sum(preds_flat * target_flat, dim=1)
    union = TORCH.sum(preds_flat, dim=1) + TORCH.sum(target_flat, dim=1) - intersection
    return (intersection + eps) / (union + eps)


def dice_loss_from_logits(logits: Any, target: Any) -> Any:
    return 1.0 - TORCH.mean(soft_dice_coefficient_from_logits(logits, target))


def run_epoch(model: Any, loader: Any, bce_loss_fn: Any, device: Any, hard_threshold: float, show_progress: bool, desc: str) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_bce = 0.0
    total_dice_loss = 0.0
    total_soft_dice = 0.0
    total_soft_iou = 0.0
    total_hard_dice = 0.0
    total_hard_iou = 0.0
    total_items = 0
    with TORCH.no_grad():
        for batch in maybe_tqdm(loader, enabled=show_progress, desc=desc):
            prev_image = batch["prev_image"].to(device)
            prev_embedding = batch["prev_embedding"].to(device)
            target_embedding = batch["target_embedding"].to(device)
            target_image = batch["target_image"].to(device)
            logits = model(prev_image, prev_embedding, target_embedding)
            bce = bce_loss_fn(logits, target_image)
            dice_loss = dice_loss_from_logits(logits, target_image)
            loss = bce + dice_loss
            soft_dice = TORCH.mean(soft_dice_coefficient_from_logits(logits, target_image))
            soft_iou = TORCH.mean(soft_iou_from_logits(logits, target_image))
            hard_dice = TORCH.mean(hard_dice_coefficient_from_logits(logits, target_image, threshold=hard_threshold))
            hard_iou = TORCH.mean(hard_iou_from_logits(logits, target_image, threshold=hard_threshold))
            batch_items = int(prev_image.shape[0])
            total_loss += float(loss.item()) * batch_items
            total_bce += float(bce.item()) * batch_items
            total_dice_loss += float(dice_loss.item()) * batch_items
            total_soft_dice += float(soft_dice.item()) * batch_items
            total_soft_iou += float(soft_iou.item()) * batch_items
            total_hard_dice += float(hard_dice.item()) * batch_items
            total_hard_iou += float(hard_iou.item()) * batch_items
            total_items += batch_items
    if total_items == 0:
        raise RuntimeError("dataloader yielded zero items")
    return {
        "loss": total_loss / total_items,
        "bce": total_bce / total_items,
        "dice_loss": total_dice_loss / total_items,
        "soft_dice": total_soft_dice / total_items,
        "soft_iou": total_soft_iou / total_items,
        "hard_dice": total_hard_dice / total_items,
        "hard_iou": total_hard_iou / total_items,
    }


def tensor_chw_to_uint8_hwc(image: np.ndarray) -> np.ndarray:
    clipped = np.clip(image, 0.0, 1.0)
    if int(clipped.shape[0]) == 1:
        clipped = np.repeat(clipped, 3, axis=0)
    hwc = np.transpose(clipped, (1, 2, 0))
    return np.rint(hwc * 255.0).astype(np.uint8, copy=False)


def make_reconstruction_panel(*, prev_images: list[np.ndarray], pred_images: list[np.ndarray], target_images: list[np.ndarray]) -> np.ndarray:
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
    for row_idx, (prev_image, pred_image, target_image) in enumerate(zip(prev_images, pred_images, target_images, strict=True)):
        top = spacer + (row_idx * (sample_h + spacer))
        left = spacer
        for col_idx, image in enumerate([prev_image, pred_image, target_image]):
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
    num_samples: int,
    wandb_handler: WandbHandler,
    hard_threshold: float,
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
            pred_probs = TORCH.sigmoid(pred_logits)
            pred_image = (pred_probs >= hard_threshold).to(dtype=pred_probs.dtype).squeeze(0).detach().cpu().numpy()
            prev_images.append(prev_image_tensor.cpu().numpy())
            pred_images.append(pred_image)
            target_images.append(target_image)
    panel = make_reconstruction_panel(prev_images=prev_images, pred_images=pred_images, target_images=target_images)
    recon_dir = run_dir / "reconstructions"
    recon_dir.mkdir(parents=True, exist_ok=True)
    output_path = recon_dir / f"{split_name}_final.png"
    Image.fromarray(panel, mode="RGB").save(output_path)
    LOGGER.info("saved reconstruction panel: %s", output_path)
    wandb_image = wandb_handler.image(output_path, caption=f"{split_name} final | predicted prev+target embeddings | pred_threshold={hard_threshold}")
    if wandb_image is not None:
        wandb_handler.log_payload({f"{split_name}/recon_panel": wandb_image})
    return output_path


def select_sources(
    z_by_fire: dict[str, np.ndarray],
    frames_by_fire: dict[str, list[Path]],
    keys: set[str],
) -> tuple[dict[str, np.ndarray], dict[str, list[Path]]]:
    return ({k: z_by_fire[k] for k in keys}, {k: frames_by_fire[k] for k in keys})


def main() -> int:
    args = parse_args()
    if not args.decoder_run_dir:
        raise ValueError("--decoder-run-dir is required")
    if not args.predicted_embeddings_dir:
        raise ValueError("--predicted-embeddings-dir is required")

    decoder_run_dir = args.decoder_run_dir
    predicted_dir = args.predicted_embeddings_dir
    checkpoint_path = decoder_run_dir / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"decoder checkpoint not found: {checkpoint_path}")

    predicted_manifest = json.loads((predicted_dir / "manifest.json").read_text(encoding="utf-8"))
    predicted_splits = json.loads((predicted_dir / "splits.json").read_text(encoding="utf-8"))
    source_model_dir = Path(str(predicted_manifest.get("source_model_dir", "")))
    if not source_model_dir.exists():
        raise FileNotFoundError(f"source model dir not found: {source_model_dir}")
    predictor_run_id = str(predicted_manifest.get("source_run_id", "unknown"))
    predictor_variant = str(predicted_manifest.get("source_variant", "unknown"))
    export_history = int(predicted_manifest.get("history", 1))
    source_timestamp = str(predicted_manifest.get("source_timestamp", "unknown"))

    run_id = make_run_id(args.component, args.family, args.variant, decoder_run_dir.name, predictor_run_id)
    run_dir = args.output_dir / args.component / args.family / run_id
    setup_logging(run_dir)

    LOGGER.info("decoder_run_dir=%s", decoder_run_dir)
    LOGGER.info("checkpoint=%s", checkpoint_path)
    LOGGER.info("predicted_embeddings_dir=%s", predicted_dir)
    LOGGER.info("source_model_dir=%s", source_model_dir)
    LOGGER.info("predictor_run_id=%s predictor_variant=%s", predictor_run_id, predictor_variant)

    predicted_z_by_fire, frames_by_fire = build_decoder_sources(model_dir=predicted_dir)
    train_ids = set(str(x) for x in predicted_splits.get("train_ids", []))
    val_ids = set(str(x) for x in predicted_splits.get("val_ids", []))
    holdout_ids = set(str(x) for x in predicted_splits.get("holdout_ids", []))
    common_ids = set(predicted_z_by_fire) & set(frames_by_fire)
    train_ids &= common_ids
    val_ids &= common_ids
    holdout_ids &= common_ids

    if args.max_sequences > 0:
        limited_ids = set(sorted(common_ids)[: args.max_sequences])
        predicted_z_by_fire = {fire_id: predicted_z_by_fire[fire_id] for fire_id in limited_ids}
        frames_by_fire = {fire_id: frames_by_fire[fire_id] for fire_id in limited_ids}
        train_ids &= limited_ids
        val_ids &= limited_ids
        holdout_ids &= limited_ids

    min_target_idx = export_history + 1
    train_ds = WildfireDecoderDataset(
        *select_sources(predicted_z_by_fire, frames_by_fire, train_ids),
        image_channels=1,
        binarize_masks=True,
        mask_threshold=args.mask_threshold,
        normalize_embeddings=False,
        min_target_idx=min_target_idx,
        return_tensors=True,
    )
    val_ds = WildfireDecoderDataset(
        *select_sources(predicted_z_by_fire, frames_by_fire, val_ids),
        image_channels=1,
        binarize_masks=True,
        mask_threshold=args.mask_threshold,
        normalize_embeddings=False,
        min_target_idx=min_target_idx,
        return_tensors=True,
    )
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(f"empty split after min_target_idx={min_target_idx}: train={len(train_ds)} val={len(val_ds)}")

    train_loader = DataLoader(cast(Any, train_ds), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_loader = DataLoader(cast(Any, val_ds), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = resolve_device(args.device)
    state = TORCH.load(checkpoint_path, map_location=device, weights_only=False)
    config = BinaryMaskConditionalUNetConfig(**state["config"])
    model = BinaryMaskConditionalUNetDecoder(config).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    bce_loss_fn = TORCH.nn.BCEWithLogitsLoss()

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
            "component": args.component,
            "family": args.family,
            "variant": args.variant,
            "decoder_run_dir": str(decoder_run_dir),
            "predicted_embeddings_dir": str(predicted_dir),
            "predictor_run_id": predictor_run_id,
            "predictor_variant": predictor_variant,
            "source_timestamp": source_timestamp,
            "mask_threshold": args.mask_threshold,
            "hard_threshold": args.hard_threshold,
            "export_history": export_history,
            "min_target_idx": min_target_idx,
            "batch_size": args.batch_size,
            "device": str(device),
        },
    )

    LOGGER.info("device=%s", device)
    LOGGER.info("starting inference with min_target_idx=%d", min_target_idx)

    train_eval = run_epoch(model, train_loader, bce_loss_fn, device, args.hard_threshold, show_progress=not args.no_progress, desc="eval train")
    val_eval = run_epoch(model, val_loader, bce_loss_fn, device, args.hard_threshold, show_progress=not args.no_progress, desc="eval val")

    train_panel = log_reconstruction_panel(
        model=model,
        dataset=train_ds,
        device=device,
        run_dir=run_dir,
        split_name="train",
        num_samples=args.recon_log_samples,
        wandb_handler=wandb_handler,
        hard_threshold=args.hard_threshold,
    )
    val_panel = log_reconstruction_panel(
        model=model,
        dataset=val_ds,
        device=device,
        run_dir=run_dir,
        split_name="val",
        num_samples=args.recon_log_samples,
        wandb_handler=wandb_handler,
        hard_threshold=args.hard_threshold,
    )

    metrics = {
        "train/loss": train_eval["loss"],
        "train/bce": train_eval["bce"],
        "train/dice_loss": train_eval["dice_loss"],
        "train/soft_dice": train_eval["soft_dice"],
        "train/soft_iou": train_eval["soft_iou"],
        "train/hard_dice": train_eval["hard_dice"],
        "train/hard_iou": train_eval["hard_iou"],
        "val/loss": val_eval["loss"],
        "val/bce": val_eval["bce"],
        "val/dice_loss": val_eval["dice_loss"],
        "val/soft_dice": val_eval["soft_dice"],
        "val/soft_iou": val_eval["soft_iou"],
        "val/hard_dice": val_eval["hard_dice"],
        "val/hard_iou": val_eval["hard_iou"],
    }
    LOGGER.info(
        "[eval] train/loss=%.6f val/loss=%.6f val/hard_dice=%.6f val/hard_iou=%.6f",
        train_eval["loss"],
        val_eval["loss"],
        val_eval["hard_dice"],
        val_eval["hard_iou"],
    )

    summary = {
        "run_id": run_id,
        "component": args.component,
        "family": args.family,
        "variant": args.variant,
        "decoder_run_dir": str(decoder_run_dir),
        "decoder_checkpoint": str(checkpoint_path),
        "predicted_embeddings_dir": str(predicted_dir),
        "source_model_dir": str(source_model_dir),
        "predictor_run_id": predictor_run_id,
        "predictor_variant": predictor_variant,
        "source_timestamp": source_timestamp,
        "device": str(device),
        "splits": {
            "train_fire_ids": len(train_ids),
            "val_fire_ids": len(val_ids),
            "holdout_fire_ids_reserved": len(holdout_ids),
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
        },
        "pipeline": {
            "prev_image": "real",
            "prev_embedding": "predicted",
            "target_embedding": "predicted",
            "target_image": "real",
            "prediction_mode": str(predicted_manifest.get("prediction_mode", "")),
            "export_history": export_history,
            "min_target_idx": min_target_idx,
        },
        "metrics": metrics,
        "qualitative": {
            "train_recon_panel": str(train_panel) if train_panel is not None else "",
            "val_recon_panel": str(val_panel) if val_panel is not None else "",
        },
    }
    summary_path = run_dir / "metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    wandb_handler.log_metrics(metrics)
    wandb_handler.log_summary(summary)
    wandb_handler.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
