# YOLOv5 ������ by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 model accuracy on a custom dataset

Usage:
    $ python path/to/val.py --data coco128.yaml --weights yolov5s.pt --img 640
"""

import argparse
import json
import os
import sys
from pathlib import Path
from threading import Thread

import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from models.common import use_flash_attn_flag
from models.yolo import OUT_LAYER
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr, \
    non_max_suppression_dfl,non_max_suppression_txt, non_max_suppression_obb,non_max_suppression_olv, xyxyxyxy2xywhr, xywhr2xyxyxyxy, \
    non_max_suppression_obb_no_nc
from utils.metrics import ap_per_class, ConfusionMatrix, process_batch, process_batch_obb
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_sync
from utils.callbacks import Callbacks
import csv

import pickle
import cv2
from utils.plots import Annotator, colors
from detect import detect

import shutil, re

def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({'image_id': image_id,
                      'category_id': class_map[int(p[5])],
                      'bbox': [round(x, 3) for x in b],
                      'score': round(p[4], 5)})

@torch.no_grad()
def run(data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.45,  # NMS IoU threshold
        task='val',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_json=False,  # save a COCO-JSON results file
        project='runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=False,  # use FP16 half-precision inference
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        callbacks=Callbacks(),
        compute_loss=None,
        save_nms = 0
        ):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        device = select_device(device, batch_size=batch_size)
        use_flash_attn_flag()
        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check image size

        # Multi-GPU disabled, incompatible with .half() https://github.com/ultralytics/yolov5/issues/99
        # if device.type != 'cpu' and torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)

        # Data
        data = check_dataset(data)  # check

    # 判断模型类别 BUGBUGBUGBUGBUG
    if hasattr(model, 'yaml'):
        pts, dfl_flag = False, False
        for mname in OUT_LAYER.keys():
            m = model.get_module_byname(mname)
            if m is not None:
                pts = mname in ['DetectROT', 'OBB', 'OBBText', 'OBB_xn']
                dfl_flag = mname in ['DetectDFL', 'OBB', 'YoloText', 'OBBText', 'DetectDFL_xn', 'OBB_xn']
                break
        mname = OUT_LAYER[mname]
    else:
        pts, dfl_flag = False, True
        mname = 10
    #
    _process_batch = process_batch_obb if pts else process_batch

    # Half
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    is_coco = type(data['val']) is str and data['val'].endswith('coco/val2017.txt')  # COCO dataset
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if not training:
        if device.type != 'cpu':
            imgsz2 = [imgsz,imgsz] if isinstance(imgsz,int) else imgsz    
            model(torch.zeros(1, 6, imgsz2[0], imgsz2[1]).to(device).type_as(next(model.parameters())))  # run once
        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        # dataloader = create_dataloader(data[task], imgsz2, batch_size, gs, single_cls, pad=0.5,
        #                                prefix=colorstr(f'{task}: '),sample_count=600, workers=0)[0]
        data_dict = {}
        if mname==3 or mname==4:
            data_dict['names'] = model.names
            data_dict['nc'] = len(model.names)
            data_dict['TMax'] = 1
        val_count = data.get('val_count',0)
        dataloader = create_dataloader(data[task], imgsz, batch_size, gs, single_cls, augment=False, cache=None, rank=-1,
                                    workers=2, pad=0.5,
                                    prefix=colorstr('val: '),debug_samples=0,sample_count=val_count,pts=pts,data_dict=data_dict)[0]
        if dataloader.dataset.names_vec is not None:
            names_vec = dataloader.dataset.names_vec.to(device='cuda') #names_vec[nc,nembd]
        else:
            names_vec = None
    else:
        names_vec = dataloader.dataset.names_vec.to(device='cuda') if dataloader.dataset.names_vec is not None else None
    
    if isinstance(conf_thres,float) and conf_thres<0:
        train_path = Path(weights).parent
        if(os.path.exists(train_path / 'threshs.npy')):
            threshs = np.load(train_path / 'threshs.npy')
            threshs = torch.from_numpy(threshs)
        else:
            threshs = torch.ones(len(names)) * conf_thres
        conf_thres = threshs.to(device)
        use_threshs = True
    else:
        use_threshs = False

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc, pts=pts)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ('%20s' + '%11s' * 8) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95', 'f1', 'thresh')
    p, r, f1, mp, mr, map50, map, t0, t1, t2 = 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3 +(mname==3 or mname==4), device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    total_count = [0,0]
    for batch_i, (img, targets, paths, shapes, ioa, samples) in enumerate(tqdm(dataloader, desc=s, ncols=max(shutil.get_terminal_size().columns - 40, 10), dynamic_ncols=False)):
        #targets[nt_batch,6=1(batch)+1(cls)+4(xywh)] ioa[ntb]
        t_ = time_sync()
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        t = time_sync()
        t0 += t - t_

        # Run model
        if hasattr(model,'yaml'):
            out, train_out = model(img, augment=augment)  #out[b,ntotal,4(dbox)+nc / 4(dbox)+1(angle)+nc / 596=84+512] train_out[nl][B,C,H,W]
        else:
            out = model(img,targets) #preds[np, 1(objs) + 4 + nc]
        
        t1 += time_sync() - t

        # Compute loss
        if compute_loss:
            if hasattr(model,'yaml'):
                if dfl_flag:
                    loss += compute_loss((out, train_out), targets, img.shape[2:], ioa.to(device), samples, img=img)[1]  # loss scaled by batch_size
                else:
                    loss += compute_loss([x.float() for x in train_out], targets)[1]  # box, obj, cls
            else:
                loss += compute_loss(out, targets.to(device), img.shape[2:], ioa.to(device), samples, imgs = img)[1]  # loss scaled by batch_size

        # Run NMS
        targets[:, 2:] *= torch.Tensor([width, height]).repeat(targets.shape[-1] // 2 - 1).view(1, -1).to(device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        t = time_sync()

        if mname == 1:#DetectDFL out[b,ntotal,4+nc] -> out[b][np,6=4(xyxy)+1(conf)+1(cls)]
            # out = non_max_suppression_dfl(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
            out = non_max_suppression_txt(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls, model=model, names_vec=names_vec)
        if mname == 3:#YoloText out[b,ntotal,4+nc + 512(n_embd)] -> out[b][np,518=4(xyxy)+512(n_embd)  +1(conf)+1(cls)], ptext[b][np,2]
            out = non_max_suppression_txt(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls, model=model, names_vec=names_vec)
            conf_thres = 0.6
            # out = non_max_suppression_olv(out, conf_thres, iou_thres, multi_label=True, agnostic=single_cls, model=model, names_vec=names_vec)
        elif mname == 2:#OBB out[b,ntotal,4(dbox)+1(angle)+nc] -> out[b][np,7=5(box-xywhr)+1(conf)+1(cls)]
            out = non_max_suppression_obb(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls, model=model, names_vec=names_vec)
        elif mname == 4:#OBB out[b,ntotal,4(dbox)+1(angle)+nc] -> out[b][np,7=5(box-xywhr)+1(conf)+1(cls)]
            out = non_max_suppression_obb_no_nc(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls, model=model, names_vec=names_vec) #
        elif mname == 0:#Detect
            # 水平框nms
            out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls,paths=paths if save_nms else None,imgsz=imgsz)
        elif mname == 10:#DetectDFL out[b,ntotal,4+nc] -> out[b][np,6=4(xyxy)+1(conf)+1(cls)]
            # out = non_max_suppression_dfl(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
            objs,pred_boxes,pred_clss = out #objs[B, np_max, 1],pred_boxes[B, np_max, 4],pred_clss[B, np_max, nc]
            objs = torch.sigmoid(objs) # objs[B, np_max, 1]
            #
            # mask_obj = objs[:,:,0] > model.thresh #mask_obj[B,np_max]
            # objs = objs[mask_obj] #objs[mask_obj[B, np_max], 1] -> objs[np, 1(objs)]
            # pred_boxes = pred_boxes[mask_obj] #pred_boxes[mask_obj[B, np_max], 4]->pred_boxes[np, 4]
            # pred_clss = pred_clss[mask_obj] #pred_clss[mask_obj[B, np_max], nc]->pred_clss[np, nc]
            pred_boxes = pred_boxes * torch.as_tensor([img.shape[-2], img.shape[-1], img.shape[-2], img.shape[-1]], device=device).view(1, 1, 4)
            out = torch.cat((pred_boxes,objs,pred_clss),dim=-1) # out[B, np_max, 1(objs) + 4 + nc]
            softmax_conf, j = pred_clss.max(-1, keepdim=True) #x[B,np_max,nc]->conf[B,np_max,1],j[B,np_max,1]
            x = torch.cat((pred_boxes, objs, j.float()), dim=-1) #x[B,np_max, 4 + 1(objs) + 1(nc)]
            out = [] # x[np, 4 + 1(objs) + 1(nc)]  -->  out[b][np,6=4(xyxy)+1(conf)+1(cls)]
            for i in range(x.shape[0]):
                # filter_mask = (conf > conf_thres[j]).view(-1) #filter_mask[np]
                filter_mask = objs[i].squeeze(-1) > model.thresh #filter_mask[np_max]
                tt = x[i][filter_mask] #tt[np, 6 = 4(xyxy) + 1(objs) + 1(nc)]
                tt[:,:4] = xywh2xyxy(tt[:,:4])
                out.append(tt) #
            # out[b][np,6=4(xyxy)+1(conf)+1(cls)]
            # out = non_max_suppression_txt(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls, model=model, names_vec=names_vec)

        t2 += time_sync() - t

        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:] #labels[nt,5=1(cls)+4(xywh)]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path, shape = Path(paths[si]), shapes[si][0]
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            if single_cls:
                pred[:, -1] = 0
            predn = pred.clone()
            if mname == 2 or mname == 4: #obb
                box_size = 5
                pbox = xywhr2xyxyxyxy(predn[:, :box_size]).view(-1, 8) #predn[nt,5(xywhr)+?(text_size)+1(conf)+1(cls)]->pbox[nt,8=4(pts)*2]
                scale_coords(img[si].shape[1:], pbox, shape, shapes[si][1])
                predn[:, :5] = xyxyxyxy2xywhr(pbox)
            else:
                box_size = 4
                scale_coords(img[si].shape[1:], predn[:, :box_size], shape, shapes[si][1]) #predn[nt,4(xyxy)+?(text_size)+1(conf)+1(cls)]
                pbox = predn[:, :4].clone()
            #text
            text_size = predn.shape[-1] - box_size - 2
            assert text_size==0 or text_size==512 or text_size==512+2
            if text_size > 0:
                ptext = predn[:,box_size:-2] #ptext[np,512(n_embd)+2(max,max_id)]
                text_cidmax = ptext.shape[-1] - 512
                if text_cidmax > 0:
                    assert text_cidmax==2
                    tnc = ptext[:,-1].to(torch.int) #tnc[np]
                    pnc = predn[:,-1].to(torch.int) #pnc[np]
                    assert pnc.shape[0]==tnc.shape[0] and pnc.shape[0]==predn.shape[0]
                    diff_idx = torch.where(pnc != tnc)[0]
                    # 找出不相等的位置
                    # for tt in diff_idx:
                    #     print(f'\033[31m{si}{os.path.basename(path)} {names[pnc[tt].item()]}:{names[tnc[tt].item()]}\033[0m')
                    total_count[0]+=diff_idx.shape[0]
                    total_count[1]+=predn.shape[0]

            # Evaluate
            if nl:
                if pts:
                    tbox = labels[:, 1:]#tpts[nt,8=4*2(pts)]
                else:
                    tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_coords(img[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  #labelsn[nt,9=1(cls)+4(pts)*2]
                correct = _process_batch(predn, labelsn, iouv) #predn[nt,5(xywhr)+1(conf)+1(cls)]
                # correct = process_batch(predn, labelsn, iouv)
                
                plot_debug = False
                if plot_debug:
                    name = os.path.splitext(os.path.basename(path))[0]
                    debug_samples_path = save_dir / 'visual'
                    debug_samples_path.mkdir(exist_ok=True)
                    f_name = f'{debug_samples_path}/val_{name}.jpg'
                    if not Path(f_name).exists():
                        im0 = cv2.imdecode(np.fromfile(path, dtype=np.uint8),cv2.IMREAD_COLOR)#cv2.imread(path)  # BGR
                        ds = Annotator(im0, line_width=3, pil=True)            
                        for i_, xyxy in enumerate(pbox.cpu().numpy()):
                            cls_ = int(predn[i_, -1])
                            ds.box_label(xyxy, f'{int(cls_)}', color=colors(int(cls_), True))
                        im0 = ds.result()            
                        cv2.imencode('.jpg', im0)[1].tofile(f_name)
                    f_name = f'{debug_samples_path}/ori_{name}.jpg'
                    if not Path(f_name).exists():
                        im0 = cv2.imdecode(np.fromfile(path, dtype=np.uint8),cv2.IMREAD_COLOR)#cv2.imread(path)  # BGR
                        ds = Annotator(im0, line_width=3, pil=True)            
                        for i_, xyxy in enumerate(tbox.cpu().numpy()):
                            cls_ = int(predn[i_, -1])
                            ds.box_label(xyxy, f'{int(cls_)}', color=colors(int(cls_), True))
                        im0 = ds.result()            
                        cv2.imencode('.jpg', im0)[1].tofile(f_name)

                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            else:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
            stats.append((correct.cpu(), pred[:, -2].cpu(), pred[:, -1].cpu(), tcls))  # (correct, conf, pcls, tcls)

            # Save/log
            if save_txt:
                save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / (path.stem + '.txt'))
            if save_json:
                save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
            callbacks.on_val_image_end(pred, predn, path, names, img[si])

        # Plot images
        if False and plots and batch_i < 3:
            f = save_dir / f'val_batch{batch_i}_labels.jpg'  # labels
            Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
            f = save_dir / f'val_batch{batch_i}_pred.jpg'  # predictions
            Thread(target=plot_images, args=(img, output_to_target(out), paths, f, names), daemon=True).start()

    if total_count[1] > 0:
        print(f'hitrate = {total_count[1]-total_count[0]}/{total_count[1]} = {100*(1-float(total_count[0])/total_count[1]):.4}%')
    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class, threshs, py = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        #p[nc] r[nc] ap[nc,10] f1[nc] ap_class[nc] threshs[nc]
        # 保存到文件
        with open(save_dir / 'status.pkl', 'wb') as f:
            pickle.dump([py,ap,names], f)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
        if weights!=None or save_dir!=Path('') and use_threshs==False:
            train_path = Path(weights).parent if weights!=None else save_dir
            np.save(train_path / 'threshs.npy', threshs)
    else:
        nt = torch.zeros(1)
        f1 = torch.zeros(1)
        threshs = torch.zeros(1)

    # Print results
    pf = '%20s' + '%11i' * 2 + '%11.4g' * 6 # print format
    print('\033[32m',pf % ('all', seen, nt.sum(), mp, mr, map50, map, f1.mean(), threshs.mean()),'\033[0m',sep='')

    # Print results per class          
    if (verbose or (nc < 50 and not training)) and nc >= 1 and len(stats):
        for i, c in enumerate(ap_class):
            print('\033[32m', pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i], f1[i], threshs[i]),'\033[0m',sep='')
        with open(save_dir / 'classes_map.csv', 'w', newline='') as file_map:
            writer = csv.writer(file_map)
            writer.writerow(['Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95', 'f1', 'thresh'])
            writer.writerow([f'{a:.6f}' if not isinstance(a, str) else a for a in ["all", seen, nt.sum(), mp, mr, map50, map, f1.mean(), threshs.mean()] ])
            # Print results per class
            for i, c in enumerate(ap_class):
                writer.writerow([f'{a:.6f}' if not isinstance(a, str) else a for a in [names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i], f1[i], threshs[i]]])


    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t2))  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.on_val_end()

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = str(Path(data.get('path', '../coco')) / 'annotations/instances_val2017.json')  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        print(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            check_requirements(['pycocotools'])
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, f1.mean(), *(loss.cpu() / len(dataloader)).tolist()), maps, t


def parse_opt():
    parser = argparse.ArgumentParser(prog='val.py')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=list, default=[640,640], help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default='runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--save_nms', type=int, default=0, help='save_nms')

    opt = parser.parse_args()
    #general
    # opt.weights = '../../yolov5-dfl/s-62.15-43.63-v11s-dfl/weights/best.pt' #dfl #'weights/yolov5s.pt'
    # opt.weights = '../../yolov5-master/weights/last.pt'
    if 1:
        opt.data = 'data/coco2017.yaml'
    #   opt.imgsz = [640, 640]
    # else:
    #     opt.data = 'data/road_dlsb417-total.yaml'
    #     opt.imgsz = [896,896]
    #coco128
    # opt.data = 'data/coco128.yaml'
    # opt.weights = '../models/coco128-yolov11s-bs16-0.942-0.682-epo99/weights/best.pt'
    #coco128-txt
    # opt.data = 'data/coco128-txt.yaml'
    # opt.weights = '../models/coco128-txt-yolov11s-txt-bs8-0.978-0.887-epo999-31/weights/best.pt'
    #coco2017
    # opt.data = 'data/coco2017.yaml'
    # opt.weights = '../models/coco2017-s-62.15-43.63-v11s-dfl/weights/best.pt'
    #coco2017-txt
    # opt.data = 'data/coco2017-txt.yaml'
    # opt.weights = '../models/coco2017-txt-yolov11s-txt-bs16-0.619-0.428-epo99-12/weights/best.pt'

    #dota
    opt.data = 'data/dota.yaml'
    # opt.weights = '../models/s-obb-80.19-61.04-dfl=0/weights/best.pt'
    opt.weights = '../models/dota-yolov11s-obb-bs16-0.801-0.615-epo99-3-640x640/weights/best.pt' #'runs/train/exp3/weights/last.pt'
    #dota-txt
    # opt.data = 'data/dota-txt.yaml'
    # opt.weights = 'runs/train/exp3/weights/last.pt'
    #mstar
    # opt.data = 'data/SAR-mstar.yaml'
    # opt.weights = '../models/mstar-99.5-89.7/weights/best.pt'
    #obb SSDD
    # opt.data = 'data/SAR-SSDD.yaml'
    # opt.weights = 'runs/train/exp/weights/best.pt'
    # opt.weights = r'runs/train/exp8/weights/best.pt'
    # opt.imgsz = [640, 640]

    #nuScene-2D
    # opt.data = 'data/nuScene-2D-mini.yaml'
    # opt.weights = 'runs/train/exp7/weights/last.pt'

    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    opt.data = check_file(opt.data)  # check file
    return opt


def main(opt):
    set_logging()
    print(colorstr('val: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    # check_requirements(requirements=FILE.parent / 'requirements.txt', exclude=('tensorboard', 'thop'))

    if opt.task in ('train', 'val', 'test'):  # run normally
        run(**vars(opt))

    elif opt.task == 'speed':  # speed benchmarks
        for w in opt.weights if isinstance(opt.weights, list) else [opt.weights]:
            run(opt.data, weights=w, batch_size=opt.batch_size, imgsz=opt.imgsz, conf_thres=.25, iou_thres=.45,
                save_json=False, plots=False)

    elif opt.task == 'study':  # run over a range of settings and save/plot
        # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5s.pt yolov5m.pt yolov5l.pt yolov5x.pt
        x = list(range(256, 1536 + 128, 128))  # x axis (image sizes)
        for w in opt.weights if isinstance(opt.weights, list) else [opt.weights]:
            f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'  # filename to save to
            y = []  # y axis
            for i in x:  # img-size
                print(f'\nRunning {f} point {i}...')
                r, _, t = run(opt.data, weights=w, batch_size=opt.batch_size, imgsz=i, conf_thres=opt.conf_thres,
                              iou_thres=opt.iou_thres, save_json=opt.save_json, plots=False)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        plot_study_txt(x=x)  # plot


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
