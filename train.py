import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
from pathlib import Path
import timm
from timm.data import Mixup
from timm.models import create_model
from timm.utils import NativeScaler, get_state_dict, accuracy,ModelEma
from augmentation import DataWrapper
from dataload import Sentinel_Dataset
from contextlib import suppress
import utils
import os
import mlflow
from ConvLSTM import ConvLSTMModel
import math
import sys
from typing import Iterable, Optional
from einops import rearrange
from scipy.stats import pearsonr

def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--gpu', default=1, type=int)
    parser.add_argument('--device', default='cuda:1',
                        help='device to use for training / testing')
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--bce-loss', action='store_true')
    parser.add_argument('--unscale-lr', action='store_true')
    parser.add_argument('--output_dir', default='yieldpredicion',
                        help='path where to save, empty for no saving')
    #parser.add_argument('--weight-decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay', default=0.001, type=float)

    parser.add_argument('-dft', '--image_dir', type=str, default="yieldpredicion/data")
    parser.add_argument('-dfv', '--label_path', type=str, default="yieldpredicion/yieldlabel.txt")

    # Model parameters
    parser.add_argument('--model', default='vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2', type=str, metavar='MODEL',
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

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-5, metavar='LR',
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
    parser.add_argument('--color-jitter', type=float, default=0.3, metavar='PCT',
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)
    
    parser.add_argument('--train-mode', action='store_true')
    parser.add_argument('--no-train-mode', action='store_false', dest='train_mode')
    parser.set_defaults(train_mode=True)
    
    parser.add_argument('--ThreeAugment', action='store_true') #3augment
    
    parser.add_argument('--src', action='store_true') #simple random crop
    
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
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")
    
    # * Cosub params
    parser.add_argument('--cosub', action='store_true') 
    
    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--attn-only', action='store_true') 
    
    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--distributed', action='store_true', default=False, help='Enabling distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    # amp about
    parser.add_argument('--if_amp', action='store_true')
    parser.add_argument('--no_amp', action='store_false', dest='if_amp')
    parser.set_defaults(if_amp=False)

    # if continue with inf
    parser.add_argument('--if_continue_inf', action='store_true')
    parser.add_argument('--no_continue_inf', action='store_false', dest='if_continue_inf')
    parser.set_defaults(if_continue_inf=False)

    # if use nan to num
    parser.add_argument('--if_nan2num', action='store_true')
    parser.add_argument('--no_nan2num', action='store_false', dest='if_nan2num')
    parser.set_defaults(if_nan2num=False)

    # if use random token position
    parser.add_argument('--if_random_cls_token_position', action='store_true')
    parser.add_argument('--no_random_cls_token_position', action='store_false', dest='if_random_cls_token_position')
    parser.set_defaults(if_random_cls_token_position=False)    

    # if use random token rank
    parser.add_argument('--if_random_token_rank', action='store_true')
    parser.add_argument('--no_random_token_rank', action='store_false', dest='if_random_token_rank')
    parser.set_defaults(if_random_token_rank=False)

    parser.add_argument('--local-rank', default=0, type=int)

    parser.add_argument('--nb_classes', default=2, type=int)
    return parser


def main(args):
    torch.cuda.empty_cache()
    utils.init_distributed_mode(args)
    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)
    seed = args.seed + utils.get_rank() 
    torch.manual_seed(seed) 
    np.random.seed(seed) 
    cudnn.benchmark = True

    run_name = args.output_dir.split("/")[-1]
    if args.local_rank == 0 and args.gpu == 1: 
        mlflow.start_run(run_name=run_name) 
        for key, value in vars(args).items():
            mlflow.log_param(key, value)

    dataset_sentinel_train = Sentinel_Dataset(args.image_dir, args.label_path)
    dataset_sentinel_val = Sentinel_Dataset(args.image_dir, args.label_path)

    if True:  
        num_tasks = utils.get_world_size() 
        global_rank = utils.get_rank() 
        sampler_sentinel_train = torch.utils.data.DistributedSampler(
            dataset_sentinel_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        ) 
        sampler_sentinel_val = torch.utils.data.DistributedSampler(
            dataset_sentinel_val, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

    data_loader_sentinel_train = torch.utils.data.DataLoader(
        dataset_sentinel_train, sampler=sampler_sentinel_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    data_loader_sentinel_val = torch.utils.data.DataLoader(
        dataset_sentinel_val, sampler=sampler_sentinel_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    mixup_fn = None  
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None  
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    print(f"Creating model: {args.model}")
    model = ConvLSTMModel(img_size=224, embed_dim=192)
                    
    model.to(device)

    model_ema = None
    if args.model_ema: 
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    amp_autocast = suppress
    loss_scaler = "none" 
    if args.if_amp:
        amp_autocast = torch.cuda.amp.autocast 
        loss_scaler = NativeScaler()

    all_params = [p for p in model.parameters()] 
    optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5) #学习率每隔lr_drop个epoch衰减一次

    criterion = torch.nn.MSELoss()      
        
    teacher_model = None
    if args.distillation_type != 'none':
        assert args.teacher_path, 'need to specify teacher-path when using distillation'
        print(f"Creating teacher model: {args.teacher_model}")
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

    output_dir = Path(args.output_dir)
    #args.resume="yieldpredicion/checkpoint.pth"
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        model.load_state_dict(checkpoint['model'])

        if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint and args.if_amp: # change loss_scaler if not amp
                loss_scaler.load_state_dict(checkpoint['scaler'])
            elif 'scaler' in checkpoint and not args.if_amp:
                loss_scaler = 'none'
        lr_scheduler.step(args.start_epoch)
        
    if args.eval:
        test_stats = evaluate(data_loader_sentinel_val, model, device, args)
        print(f"Accuracy of the network on the {len(data_loader_sentinel_val)} test images: {test_stats['acc1']:.1f}%")

        test_stats = evaluate(data_loader_sentinel_val, model_ema.ema, device, amp_autocast)
        print(f"Accuracy of the ema network on the {len(data_loader_sentinel_val)} test images: {test_stats['acc1']:.1f}%")
        return

    if args.local_rank == 0 and args.gpu == 0:
        mlflow.log_param("n_parameters", n_parameters)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    min_rmse = 10000.0
    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed:
            # train
            data_loader_sentinel_train.sampler.set_epoch(epoch)
            data_loader_sentinel_val.sampler.set_epoch(epoch)
 
        train_stats = train_one_epoch(
            model, criterion, data_loader_sentinel_train,
            optimizer, device, epoch, loss_scaler, amp_autocast,
            args.clip_grad, model_ema, mixup_fn,
            set_training_mode=args.train_mode,  # keep in eval mode for deit finetuning / train mode for training and deit III finetuning
            args=args,
        )

        lr_scheduler.step(epoch)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'model_ema': get_state_dict(model_ema),
                    'scaler': loss_scaler.state_dict() if loss_scaler != 'none' else loss_scaler,
                    'args': args,
                }, checkpoint_path)
             
             

        test_stats = evaluate(model, data_loader_sentinel_val, device, args)
        print(f"rmse of the network on the {len(data_loader_sentinel_val)} test images: {test_stats['rmse']:.1f}%")
        
        if min_rmse > test_stats["rmse"]:
            min_rmse = test_stats["rmse"]
            if args.output_dir:
                checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'model_ema': get_state_dict(model_ema),
                        'scaler': loss_scaler.state_dict() if loss_scaler != 'none' else loss_scaler,
                        'args': args,
                    }, checkpoint_path)
            
        print(f'Min RMSE: {min_rmse:.2f}%')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        
        
        # log about
        if args.local_rank == 0 and args.gpu == 0:
            for key, value in log_stats.items():
                mlflow.log_metric(key, value, log_stats['epoch'])
        
        
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

# RMSE, R_Squared, Corr
best_metrics = [float("inf"), 0, 0]

def RMSE(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    return np.sqrt(mse)


def R2_Score(y_true, y_pred):
    corr_matrix = np.corrcoef(y_true, y_pred)
    corr = corr_matrix[0, 1]
    R2 = corr ** 2

    return R2


def PCC(y_true, y_pred):
    corr, _ = pearsonr(y_true, y_pred)
    return corr


def cal_evaluate(y_true, y_pred):
    rmse = RMSE(y_true, y_pred)
    r2 = R2_Score(y_true, y_pred)
    pcc = PCC(y_true, y_pred)

    return rmse, r2, pcc


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.BCEWithLogitsLoss,
                    data_loader_sentinel: Iterable,                  
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, amp_autocast, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None):
    
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # accum_iter = args.accum_iter
    # data augmentation by following SimCLR
    data_wrapper = DataWrapper(train=True)
    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()
    true_labels = torch.empty(0)
    pred_labels = torch.empty(0)
    ndvi_criterion = torch.nn.L1Loss(reduction='mean') 
    g_min = 1000
    # debug
    # count = 0
    total_step = len(data_loader_sentinel) - 1

    #for samples, FIPS, year in metric_logger.log_every(data_loader, print_freq, header):
    for data_iter_step, x_ in enumerate(data_loader_sentinel):

        x = x_[0].to(device, non_blocking=True) # [1, 12, 29, 224, 224, 3]
        targets = x_[-1].to(device, non_blocking=True)

        b, t, g, _, _, _ = x.shape
        x = rearrange(x, 'b t g h w c -> (b t g) c h w') # [72, 3, 224, 224]
        x = data_wrapper(x)  #torch.Size([660, 7, 256, 256]) [72, 3, 224, 224]

        x = rearrange(x, '(b t g) c h w -> b t g c h w', b=b, t=t, g=g) # [1, 6, 12, 3, 224, 224]
        x = x.to(device, non_blocking=True)
        
        with amp_autocast():
            outputs = model(x)
            loss = criterion(outputs.squeeze(), targets)

        if args.if_nan2num:
            with amp_autocast():
                loss = torch.nan_to_num(loss)

        loss_value = loss.item()
      
        
        # 检查计算的损失值是否为有限的数值（即是否不是无穷大或 NaN）。如果损失无效，则停止训练。
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            if args.if_continue_inf:
                optimizer.zero_grad()
                continue
            else:
                sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        if isinstance(loss_scaler, timm.utils.NativeScaler):
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
        else:
            loss.backward()
            if max_norm != None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        #torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        true_labels = torch.cat([true_labels, targets.detach().cpu()], dim=0)
        pred_labels = torch.cat([pred_labels, outputs.detach().cpu()], dim=0)

        print("Epoch: [{}]  [ {} / {}]  s2_l: {} outputs:{} gt:{}"
              .format(epoch, data_iter_step, total_step, outputs.item()-targets.item(),outputs.item(), targets.item()))


        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    print('g_min=',g_min)
    true_labels = torch.flatten(true_labels, start_dim=0).detach().cpu().numpy()
    pred_labels = torch.flatten(pred_labels, start_dim=0).detach().cpu().numpy()
    rmse, r2, corr= cal_evaluate(true_labels, pred_labels)
    print("train_rmse:{}   train_r2:{}   train_corr:{}".format(rmse, r2, corr))
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def evaluate(model: torch.nn.Module, data_loader_sentinel: Iterable,
             device: torch.device, args):
    criterion = torch.nn.MSELoss() 
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    # switch to evaluation mode
    model.eval()

    # data augmentation by following SimCLR
    data_wrapper = DataWrapper(train=False)
    ndvi_criterion = torch.nn.L1Loss(reduction='mean') 

    true_labels = torch.empty(0)
    pred_labels = torch.empty(0)

    total_step = len(data_loader_sentinel) - 1
    for data_iter_step, x_ in enumerate(data_loader_sentinel):

        # satellite imagery
        x = x_[0].to(device, non_blocking=True)
        targets = x_[-1].to(device, non_blocking=True)
    
        b, t, g, _, _, _ = x.shape

        x = rearrange(x, 'b t g h w c -> (b t g) c h w') 
        x = data_wrapper(x)

        x = rearrange(x, '(b t g) c h w -> b t g c h w', b=b, t=t, g=g) # [1, 6, 12, 3, 224, 224]
        x = x.to(device, non_blocking=True)

        outputs= model(x)
        loss = criterion(outputs.squeeze(), targets)
        loss_value = loss.item()

        print("[ {} / {}]  yeild gap: {} outputs:{} gt:{}"
              .format(data_iter_step, total_step, outputs.item()-targets.item(), outputs.item(), targets.item()))
                
        true_labels = torch.cat([true_labels, targets.detach().cpu()], dim=0)
        pred_labels = torch.cat([pred_labels, outputs.detach().cpu()], dim=0)

        metric_logger.update(loss=loss_value)

    true_labels = torch.flatten(true_labels, start_dim=0).detach().cpu().numpy()
    pred_labels = torch.flatten(pred_labels, start_dim=0).detach().cpu().numpy()

    rmse, r2, corr = cal_evaluate(true_labels, pred_labels)
    print("train_rmse:{}   train_r2:{}   train_corr:{}".format(rmse, r2, corr))

    metric_logger.meters['rmse'].update(rmse.item(), n=1)
    metric_logger.meters['corr'].update(corr.item(), n=1)
    metric_logger.meters['r2'].update(r2.item(), n=1)
    global best_metrics
    best_metrics = [min(best_metrics[0], rmse), max(best_metrics[1], r2), max(best_metrics[2], corr)]
    print("Metrics: RMSE: {}  R_Squared: {}  Corr: {}".format(f"{best_metrics[0]:.2f}", f"{ best_metrics[1]:.2f}", f"{best_metrics[2]:.2f}"))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
