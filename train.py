# YOLOv5 by Ultralytics, GPL-3.0 license
"""
Train a YOLOv5 model on a custom dataset

Usage:
    $ python path/to/train.py --data coco128.yaml --weights yolov5s.pt --img 640
"""

import argparse
import logging
import math
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, AdamW, SGD, lr_scheduler
from tqdm import tqdm
import warnings

os.environ['KMP_DUPLICATE_LIB_OK']='True'

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

import val  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.common import use_flash_attn_flag
from models.yolo import Model, OUT_LAYER
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader,resize_and_save_images
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr, methods, check_amp, TORCH_2_4, autocast
from utils.downloads import attempt_download
from utils.loss import ComputeLoss
from utils.plots import plot_labels, plot_evolve
from utils.torch_utils import EarlyStopping, ModelEMA, de_parallel, intersect_dicts, select_device, \
    torch_distributed_zero_first
from utils.metrics import fitness
from utils.loggers import Loggers
from utils.callbacks import Callbacks

import shutil, re
from collections import OrderedDict

import socket
import uuid
import platform
import psutil
from general.gpu import get_gpu_info

# seed = 0
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe
torch.set_printoptions(linewidth=320, precision=4, profile="default")
np.set_printoptions(linewidth=320, formatter={"float_kind": "{:11.5g}".format})  # format short g, %precision=5

LOGGER = logging.getLogger(__name__)
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

def copy_weights(model):
    return {name:param.clone() for name,param in model.named_parameters()}

def check_model_changed(old_weights,new_model,epoch):
    not_changed=set()
    should_change=set()
    new_weights = copy_weights(new_model)
    for name, params in new_weights.items():
        if torch.equal(old_weights[name],params):
            if params.requires_grad:
                should_change.add(name)
            else:
                not_changed.add(name)
            
    print(f"...\nepoch:{epoch} layer not changed len:{len(not_changed)} shoud change but not len:{len(should_change)}\n...")

def train(hyp,  # path/to/hyp.yaml or hyp dictionary
          opt,
          device,
          callbacks=Callbacks()
          ):
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze

    # Directories
    w = save_dir / 'weights'  # weights dir
    w.mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pt', w / 'best.pt'

    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp) as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))

    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        from general.global_cfg import get_machine_info
        opt.machine_info = get_machine_info()
        # opt.hostname = socket.gethostname()
        # opt.platform = (f"platform: {platform.node()}") #OS
        # opt.cpu_info = (f"cpu_info: {psutil.cpu_freq()}")# 获取CPU的唯一ID（如果支持）
        # opt.unique_id = (f"unique_id: {uuid.uuid1()}") #unique_id
        # opt.get_gpu_info = get_gpu_info()
        # opt.current_dir = os.path.dirname(os.path.abspath(__file__))# 获取当前脚本所在的目录
        # # opt.root_name = os.path.basename(opt.current_dir)# 获取根目录的名称
        # opt.disk_info = (f"disk_info: {psutil.disk_partitions()}")# 获取磁盘的唯一ID（如果支持）
        opt.version = os.path.dirname(os.path.abspath(__file__))
        yaml.safe_dump(vars(opt), f, sort_keys=False)
    data_dict = None

    # Loggers
    if RANK in [-1, 0]:
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # loggers instance
        if loggers.wandb:
            data_dict = loggers.wandb.data_dict
            if resume:
                weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp

        # Register actions
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))

    # Config
    plots = not evolve if opt.plots else False # create plots
    cuda = device.type != 'cpu'
    init_seeds(1 + RANK)
    with torch_distributed_zero_first(RANK):
        data_dict = data_dict or check_dataset(data)  # check if None
    train_path, val_path = data_dict['train'], data_dict['val']
    names = ['item'] if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    nc = 1 if single_cls else int(data_dict.get('nc', len(names)))  # number of classes
    assert len(names) == nc, f'{len(names)} names found for nc={nc} dataset in {data}'  # check
    is_coco = data.endswith('coco.yaml') and nc == 80  # COCO dataset

    # Model
    pretrained = weights.endswith('.pt')
    if pretrained:
        with torch_distributed_zero_first(RANK):
            weights = attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location=device, weights_only=False)  # load checkpoint
        model = Model(cfg or ckpt['model'].yaml, ch=6, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(csd, strict=False)  # load
        LOGGER.info(f'\033[32mTransferred {len(csd)}/{len(model.state_dict())} items from {weights}\033[0m')  # report
    else:
        model = Model(cfg, ch=6, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
    if weights.endswith('.pth'):
        csd = torch.load(weights, map_location=device, weights_only=True)
        new_csd = OrderedDict()
        for k, v in csd.items():
            # if re.match('model\.23\.cv2\.\d\.2.*', k) is not None:
            #     pattern = r'^model\.23\.cv2\.(\d)\.2(.*)'
            #     replacement = r'model.23.cv2.\1.2.conv_expand\2'
            #     new_key = re.sub(pattern, replacement, k)
            #     new_csd[new_key] = v.float()
            #     print(f'{k} --> {new_key}')
            # else:
                new_csd[k] = v.float()
        del csd
        csd = new_csd
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude + ['num_batches_tracked'])  # intersect
        model.load_state_dict(csd, strict=False)
        LOGGER.info(f'\033[32mTransferred {len(csd)}/{len(model.state_dict())} items from {weights}\033[0m')  # report
    # Freeze
    freeze = [f'model.{x}.' for x in range(freeze)]  # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print(f'freezing {k}')
            v.requires_grad = False

    dfl_flag = False
    for mname in OUT_LAYER.keys():
        m = model.get_module_byname(mname)
        if m is not None:
            pts = mname in ['DetectROT', 'OBB', 'OBBText', 'OBB_xn']
            dfl_flag = mname in ['DetectDFL', 'OBB', 'YoloText', 'OBBText', 'DetectDFL_xn', 'OBB_xn']
            if dfl_flag:
                for _i, dfl_module in enumerate(m.cv2):
                    try:
                        dfl_module[-1].conv_merge.weight.requires_grad_(False)
                        print(f'Freezing {mname}.dfl.{_i}.conv_merge.weight')
                    except AttributeError:
                        pass
            break
    
    # EMA
    ema = ModelEMA(model) if RANK in [-1, 0] else None

    # Image sizes
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

    # Trainloader
    train_loader, dataset = create_dataloader(train_path, imgsz, batch_size // WORLD_SIZE, gs, single_cls,
                                              hyp=hyp, augment=opt.augment, cache=opt.cache, rect=opt.rect, rank=RANK,
                                              workers=workers, image_weights=opt.image_weights, quad=opt.quad,
                                              prefix=colorstr('train: '),shuffle=True,
                                              debug_samples=20,pts=pts,save_dir=Path(save_dir) / 'debug_sample',
                                              data_dict=data_dict)
    #mlc = int(np.concatenate(dataset.labels, 0)[:, 0].max())  # max label class
    master_path = data_dict.get('master_path','')
    if master_path!='':
        master_mask = np.array([master_path in img_file for img_file in dataset.img_files])
    else:
        master_mask = np.ones(len(dataset.img_files)).astype(bool)
    # 使用列表推导式过滤 dataset_labels
    assert master_mask.shape[0]==len(dataset.labels)
    master_labels = [label for label, mask in zip(dataset.labels, master_mask) if mask]
    mlc = int(np.concatenate(master_labels, 0)[:, 0].max())  # max label class
    #
    nb = len(train_loader)  # number of batches
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'
    
    # 获取上一级目录
    # images_parent = os.path.dirname(train_path.rstrip('/\\'))
    # if not os.path.exists(images_parent+'/qnt_imgs'):
    #     resize_and_save_images(images_parent,64,[imgsz,imgsz] if isinstance(imgsz,int) else imgsz)

    # Process 0
    if RANK in [-1, 0]:
        val_count = hyp.get('val_count',0)
        data_dict['names_vec'] = dataset.names_vec
        val_loader = create_dataloader(val_path, imgsz, batch_size // WORLD_SIZE, gs, single_cls,
                                       hyp=hyp, augment=False, cache=None if noval else opt.cache, rect=opt.rect, rank=-1,
                                       workers=workers, pad=0.5,
                                       prefix=colorstr('val: '),shuffle=False,
                                       debug_samples=0,sample_count=val_count,pts=pts,data_dict=data_dict)[0]

        if not resume:
            labels = np.concatenate(dataset.labels, 0)
            # c = torch.tensor(labels[:, 0])  # classes
            # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
            # model._initialize_biases(cf.to(device))
            if plots:
                plot_labels(labels, names, save_dir)

            # Anchors
            if not opt.noautoanchor and (not dfl_flag):
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
            model.half().float()  # pre-reduce anchor precision
        
        #.....
        reg_max = m.reg_max

        if opt.key_scale > 0:
            from utils.KAL import compute_gauss_keys_half
            keys = compute_gauss_keys_half(reg_max,opt.key_scale)
            m.update_dfl_keys_base(keys.tolist())
            print(f'proj={m.proj}')

        callbacks.on_pretrain_routine_end()

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    LOGGER.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)
    g0, g1, g2 = [], [], []  # optimizer parameter groups

    if not dfl_flag:
        for v in model.modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
                if v.bias.requires_grad:
                    g2.append(v.bias)
            elif isinstance(v, bn):  # weight (no decay)
                if v.weight.requires_grad:
                    g0.append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
                if v.weight.requires_grad:
                    g1.append(v.weight)
            elif hasattr(v,'lora_A'):
                if v.lora_A.requires_grad:
                    g1.append(v.lora_A)
                    g1.append(v.lora_B)
            else:
                if v.requires_grad:
                    g1.append(v.weight)
    else:
        g = [[], [], []]
        gn = [[], [], []]
        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:  # bias (no decay)
                    g[2].append(param)
                    gn[2].append(fullname)
                elif isinstance(module, bn):  # weight (no decay)
                    g[1].append(param)
                    gn[1].append(fullname)
                else:  # weight (with decay)
                    g[0].append(param)
                    gn[0].append(fullname)

        g0 = g[1]
        g1 = g[0]
        g2 = g[2]

    if dfl_flag:
        # nc = max(nc, 80)
        iterations = math.ceil(len(train_loader.dataset) / max(batch_size, nbs)) * epochs
        lr_fit = round(0.002 * 5 / (4 + nc), 6)  # lr0 fit equation to 6 decimal places
        optM, lr, momentum = ("SGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9) # v11 auto:0.9; v5: hyp['momentum']
        if optM == 'AdamW':
            optimizer = AdamW(g2, lr=lr, betas=(momentum, 0.999), weight_decay=0.0)  # adjust beta1 to momentum
        else:
            optimizer = SGD(g2, lr=lr, momentum=momentum, nesterov=True)
            
        optimizer.add_param_group({'params': g1, 'weight_decay': hyp['weight_decay']})  # add g0 with weight_decay
        optimizer.add_param_group({'params': g0, 'weight_decay': 0.0})  # add g1 (BatchNorm2d weights)
        hyp['warmup_bias_lr'] = 0.0
        opt.linear_lr = True
    else:
        if opt.adam:
            optimizer = Adam(g0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
        else:
            optimizer = SGD(g0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
        lr = hyp['lr0']
        momentum = hyp['momentum']
        optimizer.add_param_group({'params': g1, 'weight_decay': hyp['weight_decay']})  # add g1 with weight_decay
        optimizer.add_param_group({'params': g2})  # add g2 (biases)

    LOGGER.info(f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
                f"{len(g0)} weight(no decay), {len(g1)} weight (decay={hyp['weight_decay']}), {len(g2)} bias (no decay)")
    del g0, g1, g2

    # Scheduler
    if opt.linear_lr:
        if dfl_flag:
            lf = lambda x: max(1 - x / epochs, 0) * (1.0 - hyp['lrf']) + hyp['lrf']  # v11
        else:
            lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    # Resume
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        if resume:
            # Optimizer
            if ckpt['optimizer'] is not None:
                optimizer.load_state_dict(ckpt['optimizer'])
                best_fitness = ckpt['best_fitness']

            # EMA
            if ema and ckpt.get('ema'):
                ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
                ema.updates = ckpt['updates']

            # Epochs
            start_epoch = ckpt['epoch'] + 1     
            assert start_epoch > 0, f'{weights} training to {epochs} epochs is finished, nothing to resume.'
        
        if epochs < start_epoch:
            LOGGER.info(f"{weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {epochs} more epochs.")
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt, csd

    # DP mode
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        logging.warning('DP not recommended, instead use torch.distributed.run for best DDP Multi-GPU results.\n'
                        'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info('Using SyncBatchNorm()')

    # DDP mode
    if cuda and RANK != -1:
        model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)

    # Model parameters
    #nl = model.model[-1].nl 


    nl = m.nl  # number of detection layers (to scale hyps)
    if not dfl_flag:
        hyp['box'] *= 3 / nl  # scale to layers
        hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
        hyp['obj'] *= ((imgsz if isinstance(imgsz,int) else max(imgsz)) / 640) ** 2 * 3. / nl  # scale to image size and layers

    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(master_labels, nc).to(device) * nc  # attach class weights
    model.names = names

    # Start training
    t0 = time.time()
    if dfl_flag:
        nw = max(round(hyp['warmup_epochs'] * nb), 100) if hyp['warmup_epochs']> 0 else -1
    else:
        nw = max(round(hyp['warmup_epochs'] * nb), 1000) if hyp['warmup_epochs'] > 0 else -1  # number of warmup iterations, max(3 epochs, 1k iterations)

    last_opt_step = -1
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0, 0, 0) if mname=='YoloText' or mname=='OBBText' else (0, 0, 0, 0, 0, 0, 0, 0) # P, R, mAP@.5, mAP@.5-.95, f1, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    if dfl_flag:
        amp_flag = check_amp(model, LOGGER)
        scaler = (
            torch.amp.GradScaler("cuda", enabled=amp_flag) if TORCH_2_4 else torch.cuda.amp.GradScaler(amp_flag)
        )
    else:
        amp_flag = cuda
        scaler =  torch.amp.GradScaler(enabled=cuda) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=cuda)
    stopper = EarlyStopping(patience=opt.patience)
    compute_loss = ComputeLoss(model,use_ioa=hyp.get('use_ioa', 0),vec_names=dataset.names_vec,neg_mode=opt.neg_mode)  # init loss class
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                f'Using {train_loader.num_workers} dataloader workers\n'
                f"Logging results to \033[32m{colorstr('bold', save_dir)}\033[0m\n"
                f'Starting training for {epochs} epochs...')
    optimizer.zero_grad()
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()
        #old_weights = copy_weights(model)
        # Update image weights (optional, single-GPU only)
        if opt.image_weights:
            # model.class_weights[nc]   maps[nc]
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc #class weights: cw[nc]
            #cw[n] 类权重与类数量成反比，与每类map误差成正比
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw, master_mask=master_mask,slave_rate=hyp.get('slave_rate', 0.2))  # image weights: iw[nimages]
             # nn*wn / (iw.sum(0) + n*wn) = neg_alpha--> nwp = neg_alpha * iw.sum(0) / (nn-neg_alpha*n)
            neg_set = iw == 0
            neg_count = iw[neg_set].shape[0]
            neg_alpha = hyp.get('neg_alpha',0.02)
            #
            #if neg_count > 0 and neg_count/iw.shape[0] > neg_alpha:
            #   nwp = neg_alpha * iw.sum(0) / (neg_count-neg_alpha*iw.shape[0])
            #   iw += nwp
            #   neg_alpha_val = iw[neg_set].sum(0) / iw.sum(0)
            #   print(neg_alpha_val)
            if neg_count > 0:
                nwp = neg_alpha * iw.sum(0) / ((1 - neg_alpha)*neg_count)
                iw[neg_set] += nwp
                neg_alpha_val = iw[neg_set].sum(0) / iw.sum(0)
                if start_epoch==epoch:#仅第一个epoch显示负样本比例
                    print("neg_alpha_val=",neg_alpha_val)
            #
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx

        # Update mosaic border (optional)
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders
        #
        mtext = model.get_module_byname('YoloText')
        if mtext is None:
            mtext = model.get_module_byname('OBBText')
        mloss = torch.zeros(3 + (mtext is not None), device=device)  # mean losses
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(iter(train_loader))
        if dfl_flag:
            if mtext is None:
                tags = ('Epoch', 'gpu_mem', 'box', 'cls', 'dfl', 'labels', 'img_size')
            else:
                tags = ('Epoch', 'gpu_mem', 'box', 'cls', 'dfl', 'txt', 'labels', 'img_size')
        else:
            tags = ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'labels', 'img_size')
        tags = getattr(compute_loss, 'tags', tags)
        loggers.update_keys(tags[2:-2])
        assert len(tags)==4+len(mloss)
        LOGGER.info(('%10s' * len(tags)) % tags)
        if RANK in [-1, 0]:
            pbar = tqdm(pbar, total=nb, ncols=max(shutil.get_terminal_size().columns - 40, 10), dynamic_ncols=False) # progress bar  
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
            # Scheduler
            scheduler.step()
        for i, (imgs, targets, paths, _, ioa, samples) in pbar:  # batch--> imgs[b,C,H,W]  targets[ntb,1(b)+1(cls)+4(xywh)]
            assert ioa.shape[0]==targets.shape[0] #ioa[ntb]
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
            A23_lst = samples['A23'] #A23_lst[B]
            assert A23_lst[0].shape==(2,3)

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(
                        ni, xi, [hyp['warmup_bias_lr'] if j == (0 if (dfl_flag) else 2) else 0.0, x['initial_lr'] * lf(epoch)]
                        )
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            # with amp.autocast(enabled=cuda):
            with autocast(enabled=amp_flag): # enabled=cuda
                #img[b,c,h,w]
                pred = model(imgs)  # forward   model(W,imgs)=pred ~ targets
                #pred{x[nl=3][B,C=4*16+nc,H,W] angle[B,1,ntotal]}
                #targets[n_batch,1+1+4]#注意n_batch是一个batch里所有目标的总集，因此1(batch编号)+1(目标类别)+4(目标框坐标)
                # loss, loss_items = compute_loss(pred, targets.to(device), paths=paths, master_path=master_path)  # loss scaled by batch_size
                if dfl_flag: #A23_lst[B]
                    loss, loss_items = compute_loss(pred, targets.to(device), imgs.shape[2:], ioa.to(device), samples, imgs = imgs)  # loss scaled by batch_size
                else:
                    loss, loss_items = compute_loss(pred, targets.to(device), paths=paths, master_path=master_path)  # loss scaled by batch_size
                if RANK != -1:
                    loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                if opt.quad:
                    loss *= 4.

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni - last_opt_step >= accumulate:
                # v11
                if dfl_flag:
                    scaler.unscale_(optimizer)  # unscale gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
                # v5 + v11
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

                last_opt_step = ni

            # Log
            if RANK in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('%10s' * 2 + '%10.4g' * (len(mloss)+1) + '%10s') % (
                    f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], f'{imgs.shape[-1]}x{imgs.shape[-2]}'))
                callbacks.on_train_batch_end(ni, model, imgs, targets, paths, plots, opt.sync_bn)
            torch.cuda.empty_cache()

            # if i>10:#only for debug
            #     # if hasattr(train_loader, '_iterator') and train_loader._iterator is not None:
            #     #     train_loader._iterator._shutdown_workers()
            #     # time.sleep(3.3)  # 等待 3.3 秒
            #     break
            # end batch ------------------------------------------------------------------------------------------------

        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
        #check_model_changed(old_weights,model, epoch)
        if RANK in [-1, 0]:
            # mAP
            callbacks.on_train_epoch_end(epoch=epoch)
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            final_epoch = epoch + 1 == epochs
            if epoch >= data_dict.get('val_epoch', 0) and (not noval or final_epoch):  # Calculate mAP
                T = data_dict.get('val_epoch_T', 1)
                if (epoch+1) % T == 0:
                    results, maps, _ = val.run(data_dict,
                                            batch_size=batch_size // WORLD_SIZE * 2,
                                            imgsz=[imgsz[0]+32,imgsz[1]+32],
                                            model=ema.ema,
                                            iou_thres=hyp.get('nms_iou',0.5),  # NMS IoU threshold for best pycocotools results
                                            single_cls=single_cls,
                                            dataloader=val_loader,
                                            save_dir=save_dir,
                                            save_json=is_coco and final_epoch,
                                            verbose=nc < 50 and final_epoch,
                                            plots=plots and final_epoch,
                                            callbacks=callbacks,
                                            compute_loss=compute_loss)

                    # Update best mAP
                    fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
                    if fi > best_fitness:
                        best_fitness = fi
                    log_vals = list(mloss) + list(results) + lr
                    callbacks.on_fit_epoch_end(log_vals, epoch, best_fitness, fi)
                else:
                    fi = np.zeros((1),dtype=np.float32)
            else:
                fi = np.zeros((1),dtype=np.float32)

            # Save model
            if fi.item()>0 and ((not nosave) or (final_epoch and not evolve)):  # if save
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'model': deepcopy(de_parallel(model)).half(),
                        'ema': deepcopy(ema.ema).half(),
                        'updates': ema.updates,
                        'optimizer': optimizer.state_dict(),
                        'wandb_id': loggers.wandb.wandb_run.id if loggers.wandb else None}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi and fi > 0:
                    torch.save(ckpt, best)
                    if epoch >= data_dict.get('val_epoch', 0):
                        print(f"save {epoch}.pt to best.pt with best_fitness:{best_fitness}")
                    src,dst = str(save_dir / 'threshs.npy'),str(w / 'threshs.npy')
                    #os.system(f'cp {src} {dst}')
                    shutil.copy(src, dst)
                del ckpt
                callbacks.on_model_save(last, epoch, final_epoch, best_fitness, fi)
                # check weight not changed

            torch.cuda.empty_cache()
            # Stop Single-GPU
            if stopper(epoch=epoch, fitness=fi):
                break

            # Stop DDP TODO: known issues shttps://github.com/ultralytics/yolov5/pull/4576
            # stop = stopper(epoch=epoch, fitness=fi)
            # if RANK == 0:
            #    dist.broadcast_object_list([stop], 0)  # broadcast 'stop' to all ranks

        # Stop DPP
        # with torch_distributed_zero_first(RANK):
        # if stop:
        #    break  # must break all DDP ranks

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    
    
    if RANK in [-1, 0]:
        LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is best:
                    LOGGER.info(f'\nValidating {f}...')
                    results, _, _ = val.run(data_dict,
                                        batch_size=batch_size // WORLD_SIZE * 2,
                                        imgsz=[imgsz[0]+32,imgsz[1]+32],
                                        model=attempt_load(f, device),#.half(),
                                        iou_thres=hyp.get('nms_iou',0.5),  # NMS IoU threshold for best pycocotools results
                                        single_cls=single_cls,
                                        dataloader=val_loader,
                                        save_dir=save_dir,
                                        save_json=True,
                                        plots=False,
                                        verbose=True)
                    if is_coco:
                        callbacks.run('on_fit_epoch_end', list(mloss) + list(results) + lr, epoch, best_fitness, fi)
        callbacks.on_train_end(last, best, plots, epoch)
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")

    torch.cuda.empty_cache()
    return results


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--hyp', type=str, default='data/hyps/hyp.scratch-dfl.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', '--img', '--img-size', type=list, default=[640,640], help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--augment', type=int, default=1)
    parser.add_argument('--workers', type=int, default=4, help='maximum number of dataloader workers')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    from general.devices import get_available_cuda_devices
    devices,device_total = get_available_cuda_devices(min_memory_gb=20, max_memory_usage=10, max_gpu_util=15)
    if len(devices)>0:
        if(device_total > 1):
            print(f'available cards: \033[32m {devices} --> {devices[0]}\033[0m')
    else:
        print(f'\033[31mNo available cards{len(devices)}\033[0m')
        exit(0) #devices.append(0)
    parser.add_argument('--device', default=str(devices[0]), help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--freeze', type=int, default=0, help='Number of layers to freeze. backbone=10, all=24')
    parser.add_argument('--patience', type=int, default=300, help='EarlyStopping patience (epochs)')
    parser.add_argument('--plots', type=int, default=0, help='plot_labels')
    parser.add_argument('--check_env', type=int, default=0, help='check_env')
    parser.add_argument('--neg_mode', type=int, default=1, help='neg_mode')
    parser.add_argument('--key_scale', type=float, default=0.0, help='key_scale')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    #
    opt.weights = '../../weights/yolov11/yolov11s.pth'
    opt.imgsz = [640, 640]
    opt.epochs = 100
    opt.batch_size = 16
    #coco2017
    # opt.data = 'data/coco2017.yaml'
    # opt.cfg = 'models/yolov11s.yaml'
    # # opt.cfg = 'models/yolov12s-obb.yaml'
    # opt.weights = '' #'../../weights/yolov11/yolov11s.pth'#'weights/yolov11s.pt' #'weights/yolov5m.pt'
    # opt.hyp = 'data/hyps/hyp.scratch-dfl.yaml'
    #coco2017-txt
    # opt.data = 'data/coco2017-txt.yaml'
    # opt.cfg = 'models/yolov11s-txt-nc1.yaml'
    # # opt.cfg = 'models/yolov12s-obb.yaml'
    # opt.weights = '../../weights/yolov11/yolov11s.pth'#'weights/yolov11s.pt' #'weights/yolov5m.pt'
    # opt.hyp = 'data/hyps/hyp.scratch-dfl-text.yaml'
    #coco128
    # opt.data = 'data/coco128.yaml'
    # opt.cfg = 'models/yolov11s.yaml'
    # opt.hyp = 'data/hyps/hyp.scratch-dfl.yaml'
    # opt.weights = '../../weights/yolov11/yolov11s.pth'
    #coco128-txt
    # opt.data = 'data/coco128-txt.yaml'
    # # opt.cfg = 'models/yolov11s-txt.yaml'
    # opt.cfg = 'models/yolov11s-txt-nc1.yaml'
    # opt.hyp = 'data/hyps/hyp.scratch-dfl-text.yaml'
    # opt.weights = '../../weights/yolov11/yolov11s.pth'
    #
    # opt.nosave = True
    # opt.batch_size = 8
    # opt.epochs = 1000
    
    #dota
    opt.data = 'data/dota-diff.yaml'
    opt.cfg = 'models/yolov11s-obb_chgdet.yaml'
    # opt.cfg = 'models/yolov12s-obb.yaml'
    # opt.cfg = 'models/yolov11n-obb-LA.yaml'
    opt.weights = ''#'../../weights/yolov11/yolov11s.pth'#'weights/yolov11s.pt' #'weights/yolov5m.pt'
    opt.hyp = 'data/hyps/hyp.scratch-dfl.yaml'
    # opt.name = 'exp-DFLAttn'
    #dota-txt
    # opt.data = 'data/dota-txt.yaml'
    # # opt.cfg = 'models/yolov11s-obb-txt.yaml'
    # opt.cfg = 'models/yolov11s-obb-txt-nc1.yaml'
    # opt.weights = ''#'../../weights/yolov11/yolov11s.pth'#'weights/yolov11s.pt' #'weights/yolov5m.pt'
    # opt.hyp = 'data/hyps/hyp.scratch-dfl-text.yaml'
    # opt.neg_mode = 1
    #dota-taylor
    # opt.data = 'data/dota.yaml'
    # opt.cfg = 'models/yolov11n-obb-x4.yaml'
    # # opt.cfg = 'models/yolov11n-obb-x1.yaml'
    # opt.weights = ''#'../../weights/yolov11/yolov11s.pth'#'weights/yolov11s.pt' #'weights/yolov5m.pt'
    # opt.hyp = 'data/hyps/hyp.scratch-dfl.yaml'

    #TZ-ship
    # opt.data = 'data/TZ-ship.yaml'
    # opt.cfg = 'models/yolov11s-obb.yaml'
    # opt.hyp = 'data/hyps/hyp.scratch-dfl-deg180.yaml'
    # opt.epochs = 300
    #SAR-mstar
    # opt.data = 'data/SAR-mstar.yaml'
    # opt.cfg = 'models/yolov11s-obb.yaml'
    # opt.hyp = 'data/hyps/hyp.scratch-dfl-deg180.yaml'
    # opt.epochs = 400
    #SAR-SSDD
    # opt.data = 'data/SAR-SSDD.yaml'
    # opt.cfg = 'models/yolov11s-obb.yaml'
    # opt.hyp = 'data/hyps/hyp.scratch-dfl-deg180.yaml'
    # opt.epochs = 300
    #AirSarShip
    # opt.data = 'data/AirSarShip.yaml'
    # opt.cfg = 'models/yolov11s-obb.yaml'
    # opt.hyp = 'data/hyps/hyp.scratch-dfl-deg180.yaml'
    # opt.epochs = 300
    
    ##nuScene-2D
    # opt.data = 'data/nuScene-2D.yaml'
    # opt.cfg = 'models/yolov11s.yaml'
    # # opt.cfg = 'models/yolov12s-obb.yaml'
    # opt.weights = '../../weights/yolov11/yolov11s.pth'#'weights/yolov11s.pt' #'weights/yolov5m.pt'
    # opt.hyp = 'data/hyps/hyp.scratch-dfl-text.yaml'

    #opt.resume = 'runs/train/exp8/weights/last.pt'
    #test
    #opt.weights = 'runs/train/exp60/weights/best.pt'

    #road damage
    # opt.data = 'data/road_dlsb417-total.yaml'
    # opt.cfg = 'models/yolov5m.yaml'  #F(W,images)=labels
    # #opt.weights = '../runs/train/model30000_92.1/weights/best.pt'#'runs/train/exp/weights/last.pt'#'weights/yolov5m.pt'
    # opt.hyp = 'data/hyps/hyp.scratch_patch.yaml'
    # opt.imgsz = [896, 896]
    # opt.batch_size = 24
    # #opt.augment = 1
    # opt.image_weights = True
    # opt.resume = 'runs/train/exp5/weights/last.pt'
    
    #marine
    # opt.data = 'data/marine.yaml'
    # opt.cfg = 'models/yolov5m.yaml'  #F(W,images)=labels
    # opt.weights = 'weights/yolov5m.pt'#'weights/yolov5m.pt'
    # opt.hyp = 'data/hyps/hyp.scratch_patch.yaml'
    # opt.imgsz = [640, 1024]
    #aquarium
    # opt.data = 'data/aquarium.yaml'
    # opt.cfg = 'models/yolov5s.yaml'  #F(W,images)=labels
    # opt.weights = 'weights/yolov5s.pt'#'weights/yolov5m.pt'
    # opt.hyp = 'data/hyps/hyp.scratch_patch.yaml'
    # opt.imgsz = [896, 896]
    #sona
    # opt.data = 'data/sona.yaml'
    # opt.cfg = 'models/yolov5s.yaml'  #F(W,images)=labels
    # opt.weights = 'weights/yolov5s.pt'#'weights/yolov5m.pt'
    # opt.hyp = 'data/hyps/hyp.scratch_patch.yaml'
    # opt.imgsz = [1024, 896]
    #opt.augment = 1
    #opt.image_weights = True
    #opt.resume = 'runs/train/exp10/weights/last.pt'
    return opt


def main(opt):
    # Checks
    set_logging(RANK)
    if RANK in [-1, 0]:
        print(colorstr('train: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
        if opt.check_env:
            check_git_status()
            check_requirements(requirements=FILE.parent / 'requirements.txt', exclude=['thop'])

    # Resume
    if opt.resume and not opt.evolve:  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        if not os.path.exists(ckpt):
            print(f'\033[91m{ckpt} not exists.\033[0m')
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.safe_load(f))  # replace
        opt.cfg, opt.weights, opt.resume = '', ckpt, True  # reinstate
        LOGGER.info(f'Resuming training from {ckpt}')
    else:
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        if opt.evolve:
            opt.project = 'runs/evolve'
            opt.exist_ok = opt.resume
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    use_flash_attn_flag()
    if LOCAL_RANK != -1:
        from datetime import timedelta
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        assert opt.batch_size % WORLD_SIZE == 0, '--batch-size must be multiple of CUDA device count'
        assert not opt.image_weights, '--image-weights argument is not compatible with DDP training'
        assert not opt.evolve, '--evolve argument is not compatible with DDP training'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    # Train
    if not opt.evolve:
        train(opt.hyp, opt, device)
        if WORLD_SIZE > 1 and RANK == 0:
            _ = [print('Destroying process group... ', end=''), dist.destroy_process_group(), print('Done.')]

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
                'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
                'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
                'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
                'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
                'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
                'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
                'box': (1, 0.02, 0.2),  # box loss gain
                'cls': (1, 0.2, 4.0),  # cls loss gain
                'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
                'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
                'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
                'iou_t': (0, 0.1, 0.7),  # IoU training threshold
                'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
                'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
                'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
                'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
                'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
                'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
                'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
                'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
                'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
                'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
                'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
                'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
                'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
                'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
                'mixup': (1, 0.0, 1.0),  # image mixup (probability)
                'copy_paste': (1, 0.0, 1.0)}  # segment copy-paste (probability)

        with open(opt.hyp) as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            if 'anchors' not in hyp:  # anchors commented in hyp.yaml
                hyp['anchors'] = 3
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)  # only val/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        evolve_yaml, evolve_csv = save_dir / 'hyp_evolve.yaml', save_dir / 'evolve.csv'
        if opt.bucket:
            os.system(f'gsutil cp gs://{opt.bucket}/evolve.csv {save_dir}')  # download evolve.csv if exists

        for _ in range(opt.evolve):  # generations to evolve
            if evolve_csv.exists():  # if evolve.csv exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt(evolve_csv, ndmin=2, delimiter=',', skiprows=1)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min() + 1E-6  # weights (sum > 0)
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([x[0] for x in meta.values()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # Train mutation
            results = train(hyp.copy(), opt, device)

            # Write mutation results
            print_mutation(results, hyp.copy(), save_dir, opt.bucket)

        # Plot results
        plot_evolve(evolve_csv)
        print(f'Hyperparameter evolution finished\n'
              f"Results saved to {colorstr('bold', save_dir)}\n"
              f'Use best hyperparameters example: $ python train.py --hyp {evolve_yaml}')


def run(**kwargs):
    # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
