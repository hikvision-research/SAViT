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
prune deit model family, adapted from deit, thanks.
"""
import argparse
import datetime
from typing import NoReturn
import numpy as np
import time
import os
import sys
import torch
import torch.backends.cudnn as cudnn
import json

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma
from utils import NativeScaler_No_Optimizer

from datasets import build_dataset
from engine import train_one_epoch, evaluate, prune_one_shot
from losses import DistillationLoss
from samplers import RASampler
import models
import utils
import json

from logger import create_logger
from pruner import prepare_pruning_list, BaseUnitPruner, get_model_complexity_info, pruning_config, Controller

def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)

    # Model parameters
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # prunning parameters
    parser.add_argument('--pruning_method', type=str, default='2',
                        help='pruning metric when pruning single component')
    parser.add_argument('--pruning_layers', type=int, default=3,
                        help='which components for pruning, 0(attention head), 1(hidden dimension), 2(embedding dimension), 3(all)')
    parser.add_argument('--pruning_feed_percent', type=float, default=0.1,
                        help='how much data for calculating metric and hessian, 0.1 means 10%')
    parser.add_argument('--pruning_per_iteration', type=int, default=20)
    parser.add_argument('--maximum_pruning_iterations',type=int, default=24000)
    parser.add_argument('--pruning_flops_percentage', type=float, default=0.50,
                        help='set the flops percentage for pruning')
    parser.add_argument('--pruning_flops_threshold', type=float, default=0.001,
                        help='when threshold> abs(pruned_percentage) > 0, exit prune')                    
    parser.add_argument('--need_hessian', action='store_true')
    parser.add_argument('--hessian_embed', type=float, default=6.0,
                        help='used for adjusting the hessian sum of embedding dim')
    
    parser.add_argument('--pruning_momentum', type=float, default=0.9)
    parser.add_argument('--pruning_silent', action='store_true', 
                        help='whether to show some pruning details')
    parser.add_argument('--pruning_pickle_from', type=str, default='',
                        help='load pre-calculated metric and hessian information if you already get')
    parser.add_argument('--pruning_normalize_by_layer', action='store_true',
                        help='whether to normalize the metric')
    parser.add_argument('--pruning_normalize_type', type=int, default=2,
                        help='1(l1 normalization), 2(l2 normalization)')
    parser.add_argument('--pruning_protect', type=bool, default=True,
                        help='to prevent every layer from pruning all neurons')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None, help='used in evaluation')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train')
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, 
                        help="distillation loss type")
    parser.add_argument('--distillation-alpha', default=0.5, type=float,
                        help="distillation loss weight")
    parser.add_argument('--distillation-tau', default=1.0, type=float, 
                        help="distillation softmax temperature")

    # * Finetuning params
    parser.add_argument('--finetune', default=None, help='finetune from checkpoint')
    parser.add_argument('--finetune_op', type=int, default=0,
                        help='1(only finetune), 2(only prune)')

    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')
    parser.add_argument('--not_imagenet_default_mean_and_std', action='store_true')

    parser.add_argument('--output_dir', default='./output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--save_freq', default=50, type=int,
                        help='save checkpoint frequency')                        
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # logging 
    parser.add_argument('--log', type=str, default='same',
                        help='log file path')
    parser.add_argument('--log_period', type=int, default=10,
                        help='log period in training or finetuning')                   
    return parser


def main(args):
    utils.init_distributed_mode(args)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    if args.log == 'same':
        args.log = args.output_dir
    log_file = os.path.join(args.log, '{}.log'.format(timestamp))
    logger = create_logger(output_dir=log_file, dist_rank=utils.get_rank())
    logger.warning("GPU {} run on process {}".format(utils.get_rank(),os.getpid()))
    args_txt='----------running configuration----------\n'
    for key, value in vars(args).items():
        args_txt+=('{}: {} \n'.format(key, str(value)))
    logger.info(args_txt)
    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)
  
    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                logger.info('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(2 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    logger.info(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )
    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')
        checkpoint_model = checkpoint
        if 'model' in checkpoint_model.keys():
            checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                logger.info(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed

        model.load_state_dict(checkpoint_model, strict=False)

        if hasattr(model, 'finetune'):
            model.finetune()
    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    # in order to create optimizer without gate params, which should be freezed during fine-tuning
    for name, m in model.named_parameters():
        if "gate" in name:
            m.requires_grad = False
            logger.info("skipping parameter:{} shape:{}".format(name, m.shape))
    optimizer = create_optimizer(args, model_without_ddp)
    # recover gate gradient calculate
    for name, m in model.named_parameters():
        if "gate" in name:
            m.requires_grad = True

    loss_scaler = NativeScaler()
    loss_scaler_no_optimizer = NativeScaler_No_Optimizer()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = LabelSmoothingCrossEntropy()

    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    teacher_model = None
    if args.distillation_type != 'none':
        assert args.teacher_path, 'need to specify teacher-path when using distillation'
        logger.info(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            num_classes=args.nb_classes,
            global_pool='avg',
        )
        if args.teacher_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.teacher_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model'])
        teacher_model.to(device)
        teacher_model.eval()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    criterion = DistillationLoss(
        criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    )
    # pruning initialize
    pruning_engine = None
    args.pruning = args.pruning_flops_percentage > 0
    if args.pruning:
        args.pruning_method = [int(args.pruning_method) for _ in range(args.pruning_layers)]
        if args.pruning_layers <= 2:
            args.pruning_layers = [args.pruning_layers]
        else:
            args.pruning_layers = list(range(args.pruning_layers))

        args_dict=vars(args)
        args_dict['max_iter'] = len(data_loader_train)
        pruning_settings = pruning_config(args_dict)
        logger.info('model complexity before pruning')
        attn_flops, fc_flops, total_flops = get_model_complexity_info(model_without_ddp, logger)
        # config pruning flops
        pruning_settings.flops = [attn_flops, fc_flops, total_flops]
        pruning_engines = []
        for i, _ in enumerate(args.pruning_method):
            pruning_settings.pruning_method = args.pruning_method[i]
            pruning_settings.pruning_layers = args.pruning_layers[i]
            pruning_parameters_list = prepare_pruning_list(model_without_ddp, logger, pruning_settings)
            logger.info("Total pruning layers:{}".format(len(pruning_parameters_list)))
            pruning_engine = BaseUnitPruner(model_without_ddp, pruning_parameters_list, pruning_settings, logger)
            pruning_engines.append(pruning_engine)
        args.controller = Controller(pruning_engines)
    
    output_dir = Path(args.output_dir)
    if args.resume:
        logger.info('resume training from:{}'.format(args.resume))
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
        if hasattr(model, 'finetune'):
            model.finetune()
    if args.eval:
        test_stats = evaluate(args, logger, data_loader_val, model, device)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return
    # one-shot pruning
    if args.finetune_op == 2:
        if args.mixup == 0:
            mixup_fn = None
        logger.info(f"Start one-shot pruning")
        train_stats = prune_one_shot(
            args, logger, args.log_period,
            model, criterion, data_loader_train,
            optimizer, device, 0, loss_scaler_no_optimizer,
            args.clip_grad, model_ema, mixup_fn,
            set_training_mode=(args.teacher_path!=''),  # keep in eval mode during finetuning or pruning
        )
        # test_stats = evaluate(args, logger, data_loader_val, model, device)
        # logger.info(f"test after purning without finetune")
        # logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        args.controller.pruning_engines = None
        # save pruning model
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint_pruned_00.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                }, checkpoint_path)
        
        logger.info('pruned model complexity:')
        cur_flops = get_model_complexity_info(model_without_ddp, logger)
        original_flops = pruning_settings.flops 
        logger.info('pruning percentage:')
        logger.info('attn percentage:{}'.format((original_flops[0]-cur_flops[0])/original_flops[0]*100))
        logger.info('ffn percentage:{}'.format((original_flops[1]-cur_flops[1])/original_flops[1]*100))
        logger.info('total percentage:{}'.format((original_flops[2]-cur_flops[2])/original_flops[2]*100))
        return

    logger.info(f"Start finetune for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    # finetune
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            args, logger, args.log_period,
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn,
            set_training_mode=(args.teacher_path!=''),  # keep in eval mode during finetuning or pruning
        )

        lr_scheduler.step(epoch)
        if args.output_dir:
            if epoch % args.save_freq == 0 and epoch!=0:
                checkpoint_paths = [output_dir / 'checkpoint_{}.pth'.format(epoch)] 
            else:
                checkpoint_paths = [output_dir / 'checkpoint.pth'] 
            # only save pruned model
            if args.finetune_op==2:
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                    }, checkpoint_path)    
                break
            # save the whole finetune state
            else:
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        # 'model_ema': get_state_dict(model_ema),
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                    }, checkpoint_path)

        test_stats = evaluate(args, logger, data_loader_val, model, device)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')
        # recalculate params and Gflops
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch}

        # if args.output_dir and utils.is_main_process():
        # changed by zhengchuanyang, so we can stop it in training platform
        if utils.is_main_process():   
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))
    logger.info('pruned model complexity:')
    cur_flops = get_model_complexity_info(model_without_ddp, logger)
    if args.pruning:
        original_flops = pruning_settings.flops 
        logger.info('pruning percentage:')
        logger.info('attn percentage:{}'.format((original_flops[0]-cur_flops[0])/original_flops[0]*100))
        logger.info('ffn percentage:{}'.format((original_flops[1]-cur_flops[1])/original_flops[1]*100))
        logger.info('total percentage:{}'.format((original_flops[2]-cur_flops[2])/original_flops[2]*100))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    os.umask(0)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True, mode=0o777)
    main(args)
