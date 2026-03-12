import logging
import time

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

logger = logging.getLogger(__name__)


def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch,
                    scaler=None, log_interval=50, label_smoothing=0.0,
                    max_grad_norm=None):
    model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    total_loss = 0.0
    correct = 0
    total = 0
    start = time.time()

    for i, (images, targets) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with autocast():
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            if max_grad_norm is not None and max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            if max_grad_norm is not None and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if (i + 1) % log_interval == 0:
            elapsed = time.time() - start
            logger.info(
                "Epoch %d [%d/%d]  loss=%.4f  acc=%.2f%%  (%.1fs)",
                epoch, i + 1, len(dataloader),
                total_loss / total, 100.0 * correct / total, elapsed,
            )

    if scheduler is not None:
        scheduler.step()

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, dataloader, device, label_smoothing=0.0):
    model.eval()
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    total_loss = 0.0
    correct = 0
    correct_top5 = 0
    total = 0

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        total_loss += loss.item() * images.size(0)
        _, predicted = logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # top-k (safe when num_classes < 5)
        k = min(5, logits.size(1))
        _, predk = logits.topk(k, dim=1)
        correct_top5 += predk.eq(targets.unsqueeze(1)).any(dim=1).sum().item()

    avg_loss = total_loss / total
    top1 = 100.0 * correct / total
    top5 = 100.0 * correct_top5 / total
    return avg_loss, top1, top5
