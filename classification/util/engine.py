from typing import Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
import time
import datetime
import math
import sys
from timm.utils import accuracy, AverageMeter
from util import utils


def train_once(
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scalar: GradScaler,
        device: torch.device,
        logger: Any,
) -> Dict[str, float]:
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    batch_num = len(train_loader)
    end = time.time()
    for batch_id, (samples, targets) in enumerate(train_loader):
        data_time.update(time.time() - end)
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        scalar.scale(loss).backward()
        scalar.step(optimizer)
        scalar.update()

        torch.cuda.synchronize()
        losses.update(loss.detach(), samples.size(0))
        batch_time.update(torch.tensor(time.time() - end).to(device))
        end = time.time()

        if batch_id % 100 == 0:
            eta = batch_time.avg * (len(train_loader) - batch_id)
            eta = datetime.timedelta(seconds=int(eta))
            logger.info(f'> {batch_id + 1:{len(str(len(train_loader)))}d}/{batch_num}  loss: {losses.val:.4f}  '
                        f'time: {batch_time.val:.2f}s/it  eta: {eta}')

    total_time = utils.reduce_tensor(batch_time.sum)
    total_time = datetime.timedelta(seconds=int(total_time))
    global_lr = optimizer.param_groups[0]["lr"]
    global_loss = utils.reduce_tensor(losses.avg).item()

    logger.info(f"* train lr: {global_lr:.6f}\tloss: {global_loss:.4f}\tcost: {total_time}")
    return {"lr": global_lr, "loss": global_loss}


@torch.no_grad()
def evaluate_once(
        model: nn.Module,
        val_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        device: torch.device,
        logger: Any,
) -> Dict[str, float]:
    model.eval()
    top1 = AverageMeter()
    top5 = AverageMeter()
    losses = AverageMeter()

    for images, target in val_loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1, n=images.size(0))
        top5.update(acc5, n=images.size(0))
        losses.update(loss.detach(), n=images.size(0))

    global_top1 = utils.reduce_tensor(top1.avg).item()
    global_top5 = utils.reduce_tensor(top5.avg).item()
    global_loss = utils.reduce_tensor(losses.avg).item()

    logger.info(f"* eval  loss: {global_loss:.4f}\ttop1: {global_top1:.2f}\ttop5: {global_top5:.2f}")
    return {"loss": global_loss, "top1": global_top1, "top5": global_top5}
