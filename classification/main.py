import json
import math, sys, argparse, tabulate
import time
import datetime
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import torch.backends.cudnn as cudnn
from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, accuracy, AverageMeter
from torchmetrics.functional.classification import multiclass_accuracy
from config import get_args_parser
from util import *
from models.menu import *

# initial
args = get_args_parser().parse_args()
utils.init_distributed_mode(args)
device = torch.device(args.device)
cudnn.benchmark = True
# seed
seed = args.seed + utils.get_rank()
torch.manual_seed(seed)
np.random.seed(seed)

num_tasks = utils.get_world_size()
global_rank = utils.get_rank()
# dataset
train_dataset, _ = build_dataset(True, args)
val_dataset, num_classes = build_dataset(False, args)
train_sampler = torch.utils.data.DistributedSampler(
    train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
val_sampler = torch.utils.data.DistributedSampler(
    val_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False)
train_loader = torch.utils.data.DataLoader(
    train_dataset, sampler=train_sampler, batch_size=args.batch_size,
    num_workers=args.num_workers, pin_memory=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(
    val_dataset, sampler=val_sampler, batch_size=int(1.5 * args.batch_size),
    num_workers=args.num_workers, pin_memory=False, drop_last=False)

# create model
model = create_model(args.model, num_classes=num_classes, distillation=(args.distillation_type != 'none'))
model.to(device)
model_without_ddp = model
if args.distributed:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model_without_ddp = model.module

mixup_fn = None
mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
if mixup_active:
    mixup_fn = Mixup(
        mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
        prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
        label_smoothing=args.smoothing, num_classes=num_classes)

# create optimizer scheduler criterion
optimizer = create_optimizer(args, model_without_ddp)
loss_scaler = NativeScaler()
lr_scheduler, _ = create_scheduler(args, optimizer)
if args.mixup > 0.:
    # smoothing is handled with mixup label transform
    criterion = SoftTargetCrossEntropy()
elif args.smoothing:
    criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
else:
    criterion = torch.nn.CrossEntropyLoss()

# distillation model
teacher_model = None
if args.distillation_type != 'none':
    teacher_model = create_model(args.teacher_model, num_classes=num_classes, pretrained=False, global_pool='avg')
    checkpoint = torch.load(args.teacher_path, map_location='cpu')
    try:
        teacher_model.load_state_dict(checkpoint)
    except:
        teacher_model.load_state_dict(checkpoint['model'])
    teacher_model.to(device)
    teacher_model.eval()
criterion = DistillationLoss(
    criterion, teacher_model, args.distillation_type,
    args.distillation_alpha, args.distillation_tau)


def train():
    model.train()
    # lr = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    batch_num = len(train_loader)
    end = time.time()
    for batch_id, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images, target = images.to(device), target.to(device)

        if mixup_fn is not None:
            images, target = mixup_fn(images, target)

        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(images, output, target)
            if hasattr(model.module, 'loss'):
                loss = loss + model.module.loss * 1.0
        if not math.isfinite(loss.item()):
            logger.info("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        optimizer.zero_grad()
        # This attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(
            optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=args.clip_grad, clip_mode=args.clip_mode,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()

        # measure elapsed time
        losses.update(loss.detach(), images.size(0))
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
def evaluate():
    model.eval()
    top1 = AverageMeter()
    top5 = AverageMeter()
    losses = AverageMeter()
    loss_fn = nn.CrossEntropyLoss()
    all_targets = []
    all_predicted = []

    for images, target in val_loader:
        images, target = images.to(device), target.to(device)

        with torch.cuda.amp.autocast():
            output = model(images)
            loss = loss_fn(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1, n=images.size(0))
        top5.update(acc5, n=images.size(0))
        losses.update(loss.detach(), n=images.size(0))
        predicted = torch.max(output, 1)[1]
        all_predicted.append(predicted)
        all_targets.append(target)
    all_predicted = torch.cat(all_predicted)
    all_targets = torch.cat(all_targets)
    class_accs = multiclass_accuracy(all_predicted, all_targets, num_classes, average=None) * 100

    global_top1 = utils.reduce_tensor(top1.avg).item()
    global_top5 = utils.reduce_tensor(top5.avg).item()
    global_loss = utils.reduce_tensor(losses.avg).item()
    class_accs = utils.reduce_tensor(class_accs)
    logger.info(f"* eval  loss: {global_loss:.4f}\ttop1: {global_top1:.2f}\ttop5: {global_top5:.2f}")
    return {"loss": global_loss, "top1": global_top1, "top5": global_top5, "mutil_acc": class_accs}


if __name__ == "__main__":
    if args.eval:
        args.output_dir = '/'.join(args.resume.split('/')[:-1])
        args.output_dir = f'{args.output_dir}/eval'
    elif args.resume:
        args.output_dir = '/'.join(args.resume.split('/')[:-1])
    elif args.output_dir:
        args.output_dir = args.output_dir + f"/{args.model}/" + datetime.datetime.now().strftime(
            '%Y_%m_%d_%H') + "H"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)

    logger = create_logger(output_dir, utils.get_rank(), args.model, "a+" if args.resume else 'w+')
    logger.info(f"train dataset {len(train_dataset)} images, val dataset {len(val_dataset)} images")
    logger.info(f"model: {args.model}")
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    logger.info(json.dumps(args.__dict__, indent=2))
    if args.output_dir and utils.is_main_process():
        if args.srcpy:
            (output_dir / "model.py").write_bytes(Path(args.srcpy).read_bytes())
        with (output_dir / "model.txt").open("w") as f:
            f.write(str(model))
        with (output_dir / "config.json").open("w") as f:
            f.write(json.dumps(args.__dict__, indent=2) + "\n")
        Path(output_dir / "note.txt").touch()

    log_dict = {'epoch': [], 'learning_rate': [], 'train_loss': [], 'test_loss': [], 'top1': [], 'top5': []}
    mutil_acc = []
    best_top1 = .0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        loss_scaler.load_state_dict(checkpoint['scaler'])
        log_dict = checkpoint['log_dict']
        best_top1 = max(log_dict['top1'])

    if args.eval:
        evaluate()
        sys.exit()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        logger.info(f'[Epoch {epoch}]')
        train_stats = train()
        lr_scheduler.step(epoch)
        test_stats = evaluate()
        if utils.is_main_process():
            log_dict['epoch'].append(epoch)
            log_dict['learning_rate'].append(train_stats['lr'])
            log_dict['train_loss'].append(train_stats['loss'])
            log_dict['test_loss'].append(test_stats['loss'])
            log_dict['top1'].append(test_stats["top1"])
            log_dict['top5'].append(test_stats["top5"])
            mutil_acc.append(test_stats["mutil_acc"])
            table = tabulate.tabulate(log_dict, headers='keys', tablefmt='grid', floatfmt='.6f')
            with (output_dir / "stats.log").open("w+") as f:
                f.write(table)
                f.write(f'\nbest top1: {max(log_dict["top1"]):.2f}%\t'
                        f'\tbest top5: {max(log_dict["top5"]):.2f}%')

        checkpoint_paths = [
            [output_dir / 'checkpoint_latest.pth', True],
            [output_dir / f'checkpoint_{epoch}.pth', (epoch + 1) % 50 == 0],
            [output_dir / 'checkpoint_best.pth', best_top1 < test_stats["top1"]],
        ]
        for checkpoint_path, need_save in checkpoint_paths:
            if need_save:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                    'log_dict': log_dict,
                    'mutil_acc': mutil_acc
                }, checkpoint_path)

        best_top1 = max(best_top1, test_stats["top1"])
        logger.info(f'* best top1: {best_top1:.2f}%')
