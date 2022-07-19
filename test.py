import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import json
import os
import sys
import copy
import pickle

from pathlib import Path
from contextlib import suppress
from collections import OrderedDict

from torch.nn.parallel import DistributedDataParallel as NativeDDP

from timm.models import create_model
from timm.utils import *
from timm.data import create_dataset, create_loader, resolve_data_config
from timm.models import create_model, resume_checkpoint

import logging

import net
import arch
from cal_flops_params import cal_flops_parameters


log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
_log = logging.getLogger('search')


def get_args_parser():
    parser = argparse.ArgumentParser('evolutionary search', add_help=False)
    parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N', help='input batch size for training (default: 128)')

    # model
    parser.add_argument('--model', default='burgerformer', type=str, help='Model type to evaluate')
    parser.add_argument('--net_config', default='burgerformer_tiny', type=str, help='arch_name')
    parser.add_argument('--num-classes', type=int, default=1000, metavar='N', help='number of label classes (Model default if None)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='Resume full model and optimizer state from checkpoint (default: none)')
    parser.add_argument('--dataset-download', action='store_true', default=False, help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
    parser.add_argument('--class-map', default='', type=str, metavar='FILENAME', help='path to class to idx mapping file (default: "")')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--pin-mem', action='store_true', default=False, help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    parser.add_argument('--seed', default=0, type=int)

    # Dataset parameters
    parser.add_argument('--data-dir', default='/home/public/imagenet/', type=str, help='dataset path')
    parser.add_argument('--dataset', '-d', metavar='NAME', default='', help='dataset type (default: ImageFolder/ImageTar if empty)')
    parser.add_argument('--val-split', metavar='NAME', default='validation', help='dataset validation split (default: validation)')
    parser.add_argument('--no-prefetcher', action='store_true', default=False, help='disable fast prefetcher')
    parser.add_argument('--workers', default=8, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--sync-bn', action='store_true', help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument("--local_rank", default=0, type=int)

    # Misc
    parser.add_argument('--print-freq', default=50, type=int, help='Interval of iterations to print training/eval info.')

    # for searching
    parser.add_argument('--no-use-holdout', action='store_false', dest='use_holdout', default=True, help='Use sub-train and sub-eval set for evolutionary search.')

    parser.add_argument('--infer_device', default='gpu', type=str, help='gpu | cpu')
    return parser


def validate(model, loader, args, amp_autocast=suppress, log_suffix=''):
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    model.eval()
    top1_sum = 0

    # path_acc = []

    start_time = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if args.infer_device == 'gpu':
                input = input.cuda()
                target = target.cuda()
            else:
                input = input.cpu()
                target = target.cpu()

            with amp_autocast():
                output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]

            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            # path_acc.append((loader.dataset.parser.samples[batch_idx][0][13:], acc1))

            if args.distributed:
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)

            torch.cuda.synchronize()

            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))
            
            if args.local_rank == 0 and last_batch:
                log_name = 'Test' + log_suffix
                _log.info('{}: ' 'Time: {:.3f}s  ' 'Acc@1: {:>7.4f}  ' 'Acc@5: {:>7.4f}'.format(log_name, time.time() - start_time, top1_m.avg, top5_m.avg))

    metrics = OrderedDict([('top1', top1_m.avg), ('top5', top5_m.avg)])
    return metrics


def init_multigpu(args):
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        if args.local_rank == 0:
            _log.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.' % (args.rank, args.world_size))
    else:
        _log.info('Training with a single process on 1 GPUs.')


def multigpu_model(model, args):
    assert args.rank >= 0
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = NativeDDP(model, device_ids=[args.local_rank], broadcast_buffers=True, find_unused_parameters=True)
    return model


def main(args):

    if not hasattr(args, 'gpu'):
        args.gpu = 0

    if args.local_rank == 0:
        _log.info(args)

    # fix the seed for reproducibility
    seed = args.seed

    random_seed(seed, 0)

    cudnn.benchmark = False
    cudnn.deterministic = True

    if args.infer_device == 'gpu':
        init_multigpu(args)
    else:
        args.distributed = False

    model = create_model(
        args.model,
        num_classes=args.num_classes,
        net_config=eval("arch.%s" % args.net_config),
    )

    if args.infer_device == 'gpu':
        model = model.cuda()

    if args.resume:
        resume_checkpoint(model, args.resume, optimizer=None, loss_scaler=None, log_info=args.local_rank == 0)

    if args.infer_device == 'gpu':
        model = multigpu_model(model, args)

    data_config = resolve_data_config(vars(args), model=model, verbose=args.local_rank == 0)
    dataset_eval = create_dataset(args.dataset, root=args.data_dir, split=args.val_split, is_training=False, class_map=args.class_map, download=args.dataset_download, batch_size=args.batch_size)
    data_loader_val = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=False,
        use_prefetcher=not args.no_prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
    )

    flops, params = cal_flops_parameters(eval("arch.%s" % args.net_config))
    _log.info('FLOPs: {:.1f} G ' 'Parameters: {:.0f} M'.format(flops/1e9, params/1e6))


    validate(model, data_loader_val, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('test', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
