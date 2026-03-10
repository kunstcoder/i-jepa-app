import logging
import time

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

logger = logging.getLogger(__name__)


def compute_miou(pred: torch.Tensor, target: torch.Tensor,
                 num_classes: int, ignore_index: int = 255) -> float:
    """Compute mean Intersection over Union."""
    ious = []
    pred = pred.argmax(dim=1).flatten()
    target = target.flatten()

    for cls in range(num_classes):
        pred_mask = pred == cls
        target_mask = target == cls
        valid = target != ignore_index

        intersection = (pred_mask & target_mask & valid).sum().item()
        union = ((pred_mask | target_mask) & valid).sum().item()

        if union == 0:
            continue
        ious.append(intersection / union)

    return sum(ious) / len(ious) if ious else 0.0


def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch,
                    num_classes, ignore_index=255, scaler=None, log_interval=50):
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    total_loss = 0.0
    total_pixels = 0
    correct_pixels = 0
    start = time.time()

    for i, (images, masks) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None:
            with autocast():
                logits = model(images)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        valid = masks != ignore_index
        predicted = logits.argmax(dim=1)
        correct_pixels += (predicted[valid] == masks[valid]).sum().item()
        total_pixels += valid.sum().item()

        if (i + 1) % log_interval == 0:
            elapsed = time.time() - start
            pix_acc = 100.0 * correct_pixels / max(total_pixels, 1)
            logger.info(
                "Epoch %d [%d/%d]  loss=%.4f  pix_acc=%.2f%%  (%.1fs)",
                epoch, i + 1, len(dataloader),
                total_loss / max(i + 1, 1), pix_acc, elapsed,
            )

    if scheduler is not None:
        scheduler.step()

    avg_loss = total_loss / len(dataloader)
    pix_acc = 100.0 * correct_pixels / max(total_pixels, 1)
    return avg_loss, pix_acc


@torch.no_grad()
def evaluate(model, dataloader, device, num_classes, ignore_index=255):
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    total_loss = 0.0
    total_pixels = 0
    correct_pixels = 0
    total_miou = 0.0
    num_batches = 0

    for images, masks in dataloader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, masks)

        total_loss += loss.item() * images.size(0)
        valid = masks != ignore_index
        predicted = logits.argmax(dim=1)
        correct_pixels += (predicted[valid] == masks[valid]).sum().item()
        total_pixels += valid.sum().item()

        total_miou += compute_miou(logits, masks, num_classes, ignore_index)
        num_batches += 1

    avg_loss = total_loss / len(dataloader)
    pix_acc = 100.0 * correct_pixels / max(total_pixels, 1)
    miou = 100.0 * total_miou / max(num_batches, 1)
    return avg_loss, pix_acc, miou
