# Copyright (c) Hangzhou Hikvision Digital Technology Co., Ltd. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Purne, Train and eval functions used in main.py, adapted from deit, thanks.
"""
import math
import sys
import time
from typing import Iterable, Optional
import numpy as np

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils

def prune_one_shot(args, logger, print_freq, 
                    model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    for samples, targets in metric_logger.log_every(data_loader, logger, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        # make sure that all gradients are zero
        for p in model.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
            
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        # with torch.cuda.amp.autocast():
        ## autocast will cause nan in importance calculation
        outputs = model(samples)
        loss = criterion(samples, outputs, targets)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            logger.info("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        is_second_order = is_second_order or args.controller.need_hessian
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
        if is_second_order:
            args.controller.compute_hessian()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if len(args.pruning_pickle_from) > 0:
            break
        if args.controller.pruning_engines:
            for pruning_engine in args.controller.pruning_engines:
                pruning_engine.do_step(logger, loss=loss_value)
            if args.controller.pruning_engines[0].res_pruning == -1:
                args.controller.save_hessian()
                break
    
    if args.distributed:
        model_without_ddp = model.module
    else:
        model_without_ddp = model
    # reset all pruning_engines
    for i, pruning_engine in enumerate(reversed(args.controller.pruning_engines)):
        pruning_engine.reset()
    
    args.controller.pruning_ea(model_without_ddp, logger)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info("Averaged stats:{}".format(metric_logger))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train_one_epoch(args, logger, print_freq, 
                    model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    if args.distributed:
        model_without_ddp = model.module
    else:
        model_without_ddp = model
    for samples, targets in metric_logger.log_every(data_loader, logger, print_freq, header):  
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        # make sure that all gradients are zero
        # for p in model.parameters():
        #     if p.grad is not None:
        #         p.grad.detach_()
        #         p.grad.zero_()
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(samples, outputs, targets)
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            logger.info("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if args.controller.pruning_engines:
            for pruning_engine in args.controller.pruning_engines:
                pruning_engine.do_step(logger, loss=loss_value)
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info("Averaged stats:{}".format(metric_logger))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train_one_epoch_training_time(args, logger, print_freq, 
                        model: torch.nn.Module, criterion: DistillationLoss,
                        data_loader: Iterable, optimizer: torch.optim.Optimizer,
                        device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                        model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                        set_training_mode=True):
    """get training time of one epochs
    """
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    if args.distributed:
        model_without_ddp = model.module
    else:
        model_without_ddp = model
    total_time = 0
    ite_step = 0
    for samples, targets in metric_logger.log_every(data_loader, logger, print_freq, header):  
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        with torch.cuda.amp.autocast():
            start = time.time()
            outputs = model(samples)
            loss = criterion(samples, outputs, targets)
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            logger.info("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        end = time.time()
        total_time += end-start
        ite_step += 1
        if ite_step % 50 == 0 and ite_step != 0:
            logger.info('{} elapsed every 100 iterations'.format(total_time))
            break
            total_time = 0
        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if args.controller.pruning_engines:
            for pruning_engine in args.controller.pruning_engines:
                pruning_engine.do_step(logger, loss=loss_value)
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info("Averaged stats:{}".format(metric_logger))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(args, logger, data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    print_freq=args.log_period
    for images, target in metric_logger.log_every(data_loader, logger, print_freq, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_time(args, logger, data_loader, model, device):
    """get inference time for evaluating speedup.
    """
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    print_freq=args.log_period

    warmup_iter = 100
    num_iter = 100
    total_time = []
    # updated implementation with help from 
    # NViT (Global Vision Transformer Pruning with Hessian-Aware Saliency)
    input_shape = (256, 3,224,224)
    image = torch.rand(size=input_shape).to(device)
    with torch.cuda.amp.autocast():
        for i in range(warmup_iter):
            output = model(image)
            torch.cuda.synchronize()
        for i in range(num_iter):
            start = time.time()
            output = model(image)
            torch.cuda.synchronize()
            end = time.time()
            total_time.append(end-start)
    logger.info('total time elapsed: {}'.format(np.sum(total_time)))
    logger.info('time per op: {}'.format(np.median(total_time)))