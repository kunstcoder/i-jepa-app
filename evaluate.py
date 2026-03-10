#!/usr/bin/env python3
"""Evaluate a trained classification or segmentation model."""

import argparse
import logging

import torch
import yaml
from torch.utils.data import DataLoader

from src.models.ijepa_backbone import IJEPABackbone
from src.models.classifier import IJEPAClassifier
from src.models.segmentor import IJEPASegmentor
from src.datasets.classification_dataset import ClassificationDataset
from src.datasets.segmentation_dataset import SegmentationDataset
from src.train.train_classifier import evaluate as eval_cls
from src.train.train_segmentor import evaluate as eval_seg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to trained downstream model checkpoint")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)
    task_type = cfg["task"]["type"]

    # -- backbone
    backbone = IJEPABackbone(
        model_name=cfg["backbone"]["model_name"],
        checkpoint_path=cfg["backbone"]["checkpoint_path"],
        img_size=cfg["backbone"]["img_size"],
        patch_size=cfg["backbone"]["patch_size"],
        freeze=True,
    )

    if task_type == "classification":
        model = IJEPAClassifier(
            backbone=backbone,
            num_classes=cfg["task"]["num_classes"],
            head_mode=cfg["task"]["head_mode"],
        )

        ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        model = model.to(device)

        val_dataset = ClassificationDataset(
            root=cfg["data"]["val_dir"],
            img_size=cfg["backbone"]["img_size"],
            is_train=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg["training"]["batch_size"],
            shuffle=False,
            num_workers=cfg["training"]["num_workers"],
            pin_memory=True,
        )

        val_loss, top1, top5 = eval_cls(model, val_loader, device)
        logger.info("val_loss=%.4f  top1=%.2f%%  top5=%.2f%%", val_loss, top1, top5)

    elif task_type == "segmentation":
        model = IJEPASegmentor(
            backbone=backbone,
            num_classes=cfg["task"]["num_classes"],
            decoder_type=cfg["task"]["decoder_type"],
        )

        ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        model = model.to(device)

        ignore_index = cfg["task"].get("ignore_index", 255)
        num_classes = cfg["task"]["num_classes"]

        val_dataset = SegmentationDataset(
            images_dir=cfg["data"]["val_images_dir"],
            masks_dir=cfg["data"]["val_masks_dir"],
            img_size=cfg["backbone"]["img_size"],
            is_train=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg["training"]["batch_size"],
            shuffle=False,
            num_workers=cfg["training"]["num_workers"],
            pin_memory=True,
        )

        val_loss, pix_acc, miou = eval_seg(
            model, val_loader, device, num_classes, ignore_index,
        )
        logger.info(
            "val_loss=%.4f  pix_acc=%.2f%%  mIoU=%.2f%%",
            val_loss, pix_acc, miou,
        )
    else:
        raise ValueError(f"Unknown task type: {task_type}")


if __name__ == "__main__":
    main()
