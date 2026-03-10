import logging
import time

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

logger = logging.getLogger(__name__)


def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch,
                    scaler=None, log_interval=50):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    start = time.time()

    for i, (images, targets) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None:
            with autocast():
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
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
def evaluate(model, dataloader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
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

        # top-5
        _, pred5 = logits.topk(5, dim=1)
        correct_top5 += pred5.eq(targets.unsqueeze(1)).any(dim=1).sum().item()

    avg_loss = total_loss / total
    top1 = 100.0 * correct / total
    top5 = 100.0 * correct_top5 / total
    return avg_loss, top1, top5
