#!/usr/bin/env python3
"""Train a classification head on top of an I-JEPA backbone."""

import argparse
import logging
import os
import random

import numpy as np
import torch
import yaml
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from src.models.ijepa_backbone import IJEPABackbone
from src.models.classifier import IJEPAClassifier
from src.datasets.classification_dataset import ClassificationDataset
from src.train.train_classifier import train_one_epoch, evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _is_no_decay_param(name: str, param: torch.nn.Parameter) -> bool:
    if param.ndim <= 1:
        return True
    lname = name.lower()
    return any(k in lname for k in ("bias", "norm", "bn", "pos_embed"))


def build_optimizer(model, cfg):
    base_lr = cfg["training"]["lr"]
    weight_decay = cfg["training"].get("weight_decay", 0.05)
    head_lr_mult = cfg["training"].get("head_lr_mult", 1.0)
    use_layer_decay = cfg["training"].get("use_layer_decay", False)
    layer_decay = cfg["training"].get("layer_decay", 0.75)

    depth = len(model.backbone.encoder.blocks)
    params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # default: head gets larger LR than backbone
        lr_scale = head_lr_mult if name.startswith("head") else 1.0

        # optional: layer-wise LR decay for backbone blocks
        if use_layer_decay and name.startswith("backbone.encoder"):
            if name.startswith("backbone.encoder.blocks."):
                blk_id = int(name.split(".")[3])  # backbone.encoder.blocks.{i}....
                lr_scale = layer_decay ** (depth - 1 - blk_id)
            elif name.startswith("backbone.encoder.norm"):
                lr_scale = 1.0
            else:
                # patch_embed / pos_embed
                lr_scale = layer_decay ** depth

        params.append({
            "params": [param],
            "lr": base_lr * lr_scale,
            "weight_decay": 0.0 if _is_no_decay_param(name, param) else weight_decay,
        })

    optimizer = torch.optim.AdamW(params, lr=base_lr, weight_decay=weight_decay)
    logger.info(
        "Built optimizer with %d parameter groups (base_lr=%.2e, head_lr_mult=%.2f, layer_decay=%s)",
        len(params), base_lr, head_lr_mult,
        f"{layer_decay:.3f}" if use_layer_decay else "off",
    )
    return optimizer


def build_scheduler(optimizer, cfg, total_epochs: int):
    sched = cfg["training"].get("scheduler", "cosine")
    if sched == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
    return None


def maybe_apply_unfreeze_schedule(backbone, epoch: int, schedule, applied_steps: set[int]) -> bool:
    """Apply staged unfreezing at the beginning of an epoch."""
    changed = False
    for idx, step in enumerate(schedule):
        if idx in applied_steps:
            continue
        trigger_epoch = int(step["epoch"])
        if epoch >= trigger_epoch:
            n_blocks = int(step["unfreeze_last_n"])
            include_norm = bool(step.get("include_norm", True))
            backbone.unfreeze_last_n_blocks(n_blocks=n_blocks, include_norm=include_norm)
            applied_steps.add(idx)
            changed = True
            logger.info(
                "Applied unfreeze schedule at epoch %d: unfreeze_last_n=%d include_norm=%s",
                epoch, n_blocks, include_norm,
            )
    return changed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/classification.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.checkpoint:
        cfg["backbone"]["checkpoint_path"] = args.checkpoint
    if args.model_name:
        cfg["backbone"]["model_name"] = args.model_name
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr:
        cfg["training"]["lr"] = args.lr
    if args.device:
        cfg["training"]["device"] = args.device

    set_seed(cfg.get("seed", 42))

    if cfg["backbone"].get("freeze", True) and not cfg["backbone"].get("checkpoint_path"):
        raise ValueError(
            "backbone.freeze=true 인데 checkpoint_path가 비어 있습니다. "
            "이 경우 랜덤 encoder가 고정되어 학습이 거의 진행되지 않습니다."
        )

    device = torch.device(cfg["training"]["device"])
    logger.info("Config: %s", cfg)

    train_dataset = ClassificationDataset(
        root=cfg["data"]["train_dir"],
        img_size=cfg["backbone"]["img_size"],
        is_train=True,
    )
    val_dataset = ClassificationDataset(
        root=cfg["data"]["val_dir"],
        img_size=cfg["backbone"]["img_size"],
        is_train=False,
    )

    num_classes_data = len(train_dataset.classes)
    if cfg["task"].get("num_classes") is None:
        cfg["task"]["num_classes"] = num_classes_data
    elif cfg["task"]["num_classes"] != num_classes_data:
        raise ValueError(
            f"num_classes mismatch: cfg={cfg['task']['num_classes']} vs dataset={num_classes_data}. "
            "Food-101은 101 클래스로 설정해야 합니다."
        )

    backbone = IJEPABackbone(
        model_name=cfg["backbone"]["model_name"],
        checkpoint_path=cfg["backbone"]["checkpoint_path"],
        img_size=cfg["backbone"]["img_size"],
        patch_size=cfg["backbone"]["patch_size"],
        freeze=cfg["backbone"]["freeze"],
    )

    model = IJEPAClassifier(
        backbone=backbone,
        num_classes=cfg["task"]["num_classes"],
        head_mode=cfg["task"]["head_mode"],
        pool_last_n=cfg["task"].get("pool_last_n", 4),
        head_hidden_dim=cfg["task"].get("head_hidden_dim"),
        local_ratio=cfg["task"].get("local_ratio", 0.5),
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("Parameters: %d trainable / %d total", trainable, total)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
    )

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, total_epochs=cfg["training"]["epochs"])
    scaler = GradScaler("cuda") if (cfg["training"].get("mixed_precision", False) and device.type == "cuda") else None

    unfreeze_schedule = cfg["training"].get("unfreeze_schedule", [])
    applied_unfreeze_steps: set[int] = set()

    save_dir = cfg["training"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    best_acc = 0.0
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        if unfreeze_schedule:
            changed = maybe_apply_unfreeze_schedule(
                model.backbone, epoch, unfreeze_schedule, applied_unfreeze_steps,
            )
            if changed:
                # Rebuild optimizer/scheduler to include newly trainable params.
                optimizer = build_optimizer(model, cfg)
                scheduler = build_scheduler(
                    optimizer,
                    cfg,
                    total_epochs=max(1, cfg["training"]["epochs"] - epoch + 1),
                )

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch,
            scaler=scaler,
            log_interval=cfg["training"].get("log_interval", 50),
            label_smoothing=cfg["training"].get("label_smoothing", 0.0),
            max_grad_norm=cfg["training"].get("max_grad_norm", None),
        )
        val_loss, val_top1, val_top5 = evaluate(
            model, val_loader, device,
            label_smoothing=cfg["training"].get("label_smoothing", 0.0),
        )

        logger.info(
            "Epoch %d  train_loss=%.4f  train_acc=%.2f%%  "
            "val_loss=%.4f  val_top1=%.2f%%  val_top5=%.2f%%",
            epoch, train_loss, train_acc, val_loss, val_top1, val_top5,
        )

        if val_top1 > best_acc:
            best_acc = val_top1
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "best_acc": best_acc, "cfg": cfg},
                os.path.join(save_dir, "best.pth"),
            )
            logger.info("Saved best model (acc=%.2f%%)", best_acc)

    logger.info("Training complete. Best val accuracy: %.2f%%", best_acc)


if __name__ == "__main__":
    main()
