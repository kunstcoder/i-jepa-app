#!/usr/bin/env python3
"""Train sketch inpainting decoder on top of a pretrained I-JEPA backbone."""

import argparse
import logging
import os
import random

import numpy as np
import torch
import yaml
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from src.datasets.sketch_inpainting_dataset import SketchInpaintingDataset
from src.models.ijepa_backbone import IJEPABackbone
from src.models.reconstructor import IJEPASketchReconstructor
from src.train.train_reconstructor import train_one_epoch, evaluate

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/reconstruction.yaml")
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

    device = torch.device(cfg["training"]["device"])
    logger.info("Config: %s", cfg)

    backbone = IJEPABackbone(
        model_name=cfg["backbone"]["model_name"],
        checkpoint_path=cfg["backbone"]["checkpoint_path"],
        img_size=cfg["backbone"]["img_size"],
        patch_size=cfg["backbone"]["patch_size"],
        freeze=cfg["backbone"]["freeze"],
    )

    model = IJEPASketchReconstructor(
        backbone=backbone,
        mask_fill=cfg["task"].get("mask_fill", 0.0),
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("Parameters: %d trainable / %d total", trainable, total)

    train_dataset = SketchInpaintingDataset(
        images_dir=cfg["data"]["train_images_dir"],
        img_size=cfg["backbone"]["img_size"],
        is_train=True,
        min_mask_ratio=cfg["task"]["min_mask_ratio"],
        max_mask_ratio=cfg["task"]["max_mask_ratio"],
        sketch_blur_radius=cfg["task"]["sketch_blur_radius"],
    )
    val_dataset = SketchInpaintingDataset(
        images_dir=cfg["data"]["val_images_dir"],
        img_size=cfg["backbone"]["img_size"],
        is_train=False,
        min_mask_ratio=cfg["task"]["min_mask_ratio"],
        max_mask_ratio=cfg["task"]["max_mask_ratio"],
        sketch_blur_radius=cfg["task"]["sketch_blur_radius"],
    )

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

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    scheduler = None
    if cfg["training"]["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["training"]["epochs"],
        )

    scaler = GradScaler("cuda") if (cfg["training"].get("mixed_precision", False) and device.type == "cuda") else None

    save_dir = cfg["training"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    loss_weights = cfg["loss"]
    best_psnr = 0.0
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        train_loss, train_psnr = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch,
            loss_weights=loss_weights,
            scaler=scaler, log_interval=cfg["training"]["log_interval"],
        )
        val_loss, val_psnr = evaluate(
            model, val_loader, device,
            loss_weights=loss_weights,
        )

        logger.info(
            "Epoch %d train_loss=%.4f train_masked_psnr=%.2f val_loss=%.4f val_masked_psnr=%.2f",
            epoch, train_loss, train_psnr, val_loss, val_psnr,
        )

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "best_masked_psnr": best_psnr},
                os.path.join(save_dir, "best.pth"),
            )
            logger.info("Saved best model (masked PSNR=%.2f)", best_psnr)

    logger.info("Training complete. Best val masked PSNR: %.2f", best_psnr)


if __name__ == "__main__":
    main()
