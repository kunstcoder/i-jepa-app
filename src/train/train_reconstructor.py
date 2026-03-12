import logging
import math
import time

import torch
import torch.nn.functional as F
from torch.amp import autocast

logger = logging.getLogger(__name__)


def sobel_edges(x: torch.Tensor) -> torch.Tensor:
    kx = torch.tensor([
        [-1.0, 0.0, 1.0],
        [-2.0, 0.0, 2.0],
        [-1.0, 0.0, 1.0],
    ], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    ky = torch.tensor([
        [-1.0, -2.0, -1.0],
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 1.0],
    ], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


def reconstruction_loss(pred, target, mask, w_mask=5.0, w_global=1.0, w_edge=1.0):
    masked = (pred - target).abs() * mask
    loss_mask = masked.sum() / mask.sum().clamp(min=1.0)
    loss_global = (pred - target).abs().mean()
    edge_pred = sobel_edges(pred)
    edge_target = sobel_edges(target)
    loss_edge = (edge_pred - edge_target).abs().mean()
    total = w_mask * loss_mask + w_global * loss_global + w_edge * loss_edge
    return total, {
        "loss_mask": loss_mask.item(),
        "loss_global": loss_global.item(),
        "loss_edge": loss_edge.item(),
    }


def compute_masked_psnr(pred, target, mask):
    mse = (((pred - target) ** 2) * mask).sum() / mask.sum().clamp(min=1.0)
    if mse.item() <= 0:
        return 99.0
    return 10.0 * math.log10(1.0 / mse.item())


def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch,
                    loss_weights, scaler=None, log_interval=50):
    model.train()
    total_loss = 0.0
    total_psnr = 0.0
    start = time.time()

    for i, (images, masks, targets) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None:
            with autocast(device_type=device.type):
                pred = model(images, masks)
                loss, parts = reconstruction_loss(
                    pred, targets, masks,
                    w_mask=loss_weights["masked"],
                    w_global=loss_weights["global"],
                    w_edge=loss_weights["edge"],
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(images, masks)
            loss, parts = reconstruction_loss(
                pred, targets, masks,
                w_mask=loss_weights["masked"],
                w_global=loss_weights["global"],
                w_edge=loss_weights["edge"],
            )
            loss.backward()
            optimizer.step()

        psnr = compute_masked_psnr(pred.detach(), targets, masks)
        total_loss += loss.item() * images.size(0)
        total_psnr += psnr

        if (i + 1) % log_interval == 0:
            elapsed = time.time() - start
            logger.info(
                "Epoch %d [%d/%d] loss=%.4f mask=%.4f global=%.4f edge=%.4f masked_psnr=%.2f (%.1fs)",
                epoch, i + 1, len(dataloader),
                total_loss / max(i + 1, 1),
                parts["loss_mask"], parts["loss_global"], parts["loss_edge"],
                total_psnr / max(i + 1, 1),
                elapsed,
            )

    if scheduler is not None:
        scheduler.step()

    return total_loss / len(dataloader), total_psnr / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, device, loss_weights):
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0

    for images, masks, targets in dataloader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        pred = model(images, masks)
        loss, _ = reconstruction_loss(
            pred, targets, masks,
            w_mask=loss_weights["masked"],
            w_global=loss_weights["global"],
            w_edge=loss_weights["edge"],
        )
        total_loss += loss.item() * images.size(0)
        total_psnr += compute_masked_psnr(pred, targets, masks)

    return total_loss / len(dataloader), total_psnr / len(dataloader)
