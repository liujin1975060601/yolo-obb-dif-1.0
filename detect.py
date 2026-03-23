# YOLOv5 ������ by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import os
from models.yolo import OUT_LAYER

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, is_ascii, non_max_suppression,non_max_suppression_obb, non_max_suppression_obb_no_nc, non_max_suppression_dfl, non_max_suppression_txt,\
    apply_classifier, scale_coords,scale_coords_poly, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box,xywhr2xyxyxyxy
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, load_classifier, time_sync
from models.common import use_flash_attn_flag
from tools.plotbox import plot_one_box,plot_one_rot_box

from utils.general import get_source


def detect(model, im,augment,conf_thres, iou_thres, mname=2,agnostic_nms=False,classes=None,max_det=3000,names_vec=None):
    #classes = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
    # pred = model(im, augment=0, val=True)[0]
    #pred[2=Detect+Detect2][nl][b,a,H,W,Dectct(4(box)+1(conf)+nc) or Detect2(5=1(p)+2(dir)+2(ab))]
    # if detect_mode == "Detect_pts":
    #     pred = rot_nmsv2(pred, conf_thres, iou_thres, ab_thres=ab_thres, fold_angle=fold_angle, threshs=threshs)
    # else:
    #     pred = post_process(pred, conf_thres, iou_thres, ab_thres=ab_thres, fold_angle=fold_angle, mask_dir=mask_dir,threshs=threshs,multi_label=multi_label)
    #->pred[b][nt,10=4(pts)*2+1(conf)+1(cls)]
    visualize = False
    pred = model(im, augment=augment, visualize=visualize)[0]
    #pred[b=1,ntotal=80*80+40*40+20*20,4(xywh)+1(conf)+cls]
    # imgsz = list(im.shape[-2:])#[im.shape[-2],im.shape[-1]]
    # pred[..., 0] *= imgsz[1]  # x
    # pred[..., 1] *= imgsz[0]  # y
    # pred[..., 2] *= imgsz[1]  # w
    # pred[..., 3] *= imgsz[0]  # h
    if not isinstance(pred,torch.Tensor):
        pred = torch.tensor(pred)

    # NMS
    if mname == 0:# pred[b=1,ntotal,4+1+cls]->pred[b=1,nt_nms,6=4+1+1(cls)] 水平框nms
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        # pred = non_max_suppression(pred, conf_thres, iou_thres, labels=classes, multi_label=True, agnostic=agnostic_nms,paths=paths if save_nms else None,imgsz=imgsz)
    elif mname == 1:# 水平框dfl nms
        pred = non_max_suppression_dfl(pred, conf_thres, iou_thres, labels=classes, multi_label=True, agnostic=agnostic_nms) #pred[b,ntotal,4(xywh)+nc]->pred[nt_nms,6=4(xyxy)+1(conf)+1(cls)]
    elif mname == 3: #YoloText pred[b,ntotal,4+nc + 512(n_embd)] -> pred[b][np,518=4(xyxy)+512(n_embd) +1(conf)+1(cls)], ptext[b][np,2(max,max_id)]
        pred = non_max_suppression_txt(pred, conf_thres, iou_thres, labels=classes, multi_label=True, agnostic=agnostic_nms, model=model, names_vec=names_vec)
        # pred[b][np,520=4(xyxy)+512(n_embd) + 2(max,max_id) +1(conf)+1(cls)]
    if mname == 2: #pred[b,ntotal,5(xywhr)+nc] -> pred[b][n_nms,10=8=4(pts)*2+1(conf)+1(cls)]
        pred = non_max_suppression_obb(pred, conf_thres, iou_thres, labels=classes, multi_label=True, agnostic=agnostic_nms)
        pred10=[]
        for i, det in enumerate(pred):#batch det[n_nms,7=5(xywhr)+1(conf)+1(cls)]
            conf_cls = det[:,-2:] #[n_nms,2=1(conf)+1(cls)]
            pts = xywhr2xyxyxyxy(det[:,:-2])#[n_nms,5(xywhr)]->[n_nms,4(pts),2]
            pred10.append(torch.cat([pts.view(-1,8),conf_cls],-1)) #pts[n_nms,4*2]+conf_cls[n_nms,2=1(conf)+1(cls)]->[n_nms,10=4(pts)*2+1(conf)+1(cls)]
        pred = pred10 #->pred[b][n_nms,10=8=4(pts)*2+1(conf)+1(cls)]
    if mname == 4: #pred[b,ntotal,5(xywhr)+nc] -> pred[b][n_nms,10=8=4(pts)*2+1(conf)+1(cls)]
        pred = non_max_suppression_obb_no_nc(pred, conf_thres, iou_thres, labels=classes, multi_label=True, agnostic=agnostic_nms, model=model, names_vec=names_vec)
        pred10=[]
        for i, det in enumerate(pred):#batch det[n_nms,7=5(xywhr)+1(conf)+1(cls)]
            conf_cls = det[:,-2:] #[n_nms,2=1(conf)+1(cls)]
            pts = xywhr2xyxyxyxy(det[:,:-2])#[n_nms,5(xywhr)]->[n_nms,4(pts),2]
            pred10.append(torch.cat([pts.view(-1,8),conf_cls],-1)) #pts[n_nms,4*2]+conf_cls[n_nms,2=1(conf)+1(cls)]->[n_nms,10=4(pts)*2+1(conf)+1(cls)]
        pred = pred10 #->pred[b][n_nms,10=8=4(pts)*2+1(conf)+1(cls)]
    return pred


@torch.no_grad()
def run(weights='yolov5s.pt',  # model.pt path(s)
        source='data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        thresh_scale=1,
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        plot_label=True,
        dir_line=True,
        save_txt=False,  # save results to *.txt
        view_img=False,
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        save_img_count=0
        ):
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    use_flash_attn_flag()
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    pt = True #, onnx, tflite, pb, saved_model = (suffix == x for x in ['.pt', '.onnx', '.tflite', '.pb', ''])  # backend
    classify = False
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16
    if classify:  # second-stage classifier
        modelc = load_classifier(name='resnet50', n=2)  # initialize
        modelc.load_state_dict(torch.load('resnet50.pt', map_location=device, weights_only=False)['model']).to(device).eval()
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size

    train_path = Path(weights).parent
    if(os.path.exists(train_path / 'threshs.npy')):
        threshs = np.load(train_path / 'threshs.npy')
        threshs = torch.from_numpy(threshs)
    else:
        threshs = torch.ones(len(names)) * conf_thres
    threshs = threshs * thresh_scale
    conf_thres = threshs.to(device)

    # 判断模型类别
    pts, dfl_flag = False, False
    for mname in OUT_LAYER.keys():
        m = model.get_module_byname(mname)
        if m is not None:
            pts = mname in ['DetectROT', 'OBB']
            dfl_flag = mname in ['DetectDFL', 'OBB']
            break
    mname = OUT_LAYER[mname]

    colors = [
        (54, 67, 244),
        (99, 30, 233),
        (176, 39, 156),
        (183, 58, 103),
        (181, 81, 63),
        (243, 150, 33),
        (212, 188, 0),
        (136, 150, 0),
        (80, 175, 76),
        (74, 195, 139),
        (57, 220, 205),
        (59, 235, 255),
        (0, 152, 255),
        (34, 87, 255),
        (72, 85, 121),
        (180, 105, 255)]
    
    # root_path = '/media/liu/088d8f6e-fca3-4aed-871f-243ad962413b/datas/coco128' #4090
    root_path = os.path.dirname(os.path.normpath(source))#'/media/liu/f4854541-32b0-4d00-84a6-13d3a5dd30f2/datas/coco128' #4090-2
    names_vec_file = os.path.join(root_path,'names_vec.npz')
    if os.path.exists(names_vec_file):
        names_vec = torch.load(names_vec_file).to(device=device)
    else:
        names_vec = None

    # Run inference
    # if pt and device.type != 'cpu':
    #     model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    cls_count = [0,0]
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        #img[c=3,h,w]
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        #img[1,3,H,W]

        # Inference
        t1 = time_sync()
        pred = detect(model, img, augment,conf_thres, iou_thres, mname=mname,agnostic_nms=False,classes=None,max_det=3000, names_vec=names_vec)
        t2 = time_sync()

        # Second-stage classifier (optional)
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process predictions
        if mname==0 or mname==1 or mname==3: #HBox
            for i, det in enumerate(pred):#batch
                if webcam:  # batch_size >= 1
                    fname, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
                else:
                    fname, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
                name = os.path.basename(fname)

                p = Path(fname)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                s = name
                s += ' %gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):  # detections per image
                    # Rescale boxes from img_size to im0 size
                    # det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape, None)  # native-space pred im0.shape[1]

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    
                    for *xyxy, conf, cls in reversed(det): #det[np,4(xyxy)+512(n_embd)+2(max,max_id)+1(conf)+1(cls)]
                        cls = int(cls)
                        if len(xyxy)>4:#yolo-txt
                            ptext = xyxy[516:518] #ptext[2(max,max_id)]
                            text_cls = int(ptext[1])
                            text_cf = ptext[0]
                            xyxy = xyxy[:4]
                        else:
                            ptext = None
                        xyxy = [x.cpu() for x in xyxy]
                        #
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        #
                        if plot_label:
                            label = '{} {:.2f}'.format(names[cls], conf)
                            if ptext is not None:
                                if text_cls!=cls:
                                    label+=f' {names[text_cls]}'
                                    print(f'\033[31m{name}: {names[cls]}[{conf:.3f}]/{names[text_cls]}[{text_cf:.3f}]\033[0m')
                                    cls_count[0]+=1
                                cls_count[1]+=1
                                label+=f':{text_cf:.3f}'
                        else:
                            label = None
                        plot_one_box(np.array(xyxy), im0, color=colors[cls%len(colors)], label=label)
                    cv2.imwrite(save_path, im0)
                    # Print time (inference + NMS)
                    print(f'{s}Done. ({t2 - t1:.3f}s)')

        elif mname==2 or mname==4: #OBB
            # Process predictions
            for i, det in enumerate(pred):  # per image batch循环
                if webcam:  # batch_size >= 1
                    fname, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    fname, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                name = os.path.basename(fname)
                # if name=='P0000_0_1.jpg':
                #     print(name)

                p = Path(fname)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem)
                if not os.path.exists(str(save_dir / 'labels')):
                    os.makedirs(str(save_dir / 'labels'))
                s = name
                s += ' %gx%g ' % img.shape[2:]  # print string
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :8] = scale_coords_poly(img.shape[2:], det[:, :8], im0.shape).round()
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    # Write results
                    im = im0[:,:,:3]
                    im = np.ascontiguousarray(im)
                    for *xyxy, conf, cls in reversed(det):
                        line = (cls, *xyxy, conf) # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        xyxy = [x.cpu() for x in xyxy]
                        label = '{} {:.2f}'.format(names[int(cls)], conf)
                        if not plot_label:
                            label = None
                        plot_one_rot_box(np.array(xyxy), im, color=colors[int(cls)%len(colors)], label=label, dir_line=dir_line, line_thickness=line_thickness)
                    cv2.imwrite(save_path, im)
                    print(s)
                else:
                    with open(txt_path+'.txt', 'w') as f:
                        pass
                # Print time (inference-only)
                # LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")

    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    print(f'Done. ({time.time() - t0:.3f}s)')
    if(cls_count[1]>0):
        print(f'{cls_count[1]-cls_count[0]}/{cls_count[1]}={1-float(cls_count[0])/cls_count[1]}')
        
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./runs/train/exp40/weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='E:/datas/coco128/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640,640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--thresh_scale', type=float, default=0.2, help='confidence scale')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', default=False, action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--save_img_count', type=int, default=0, help='save_img_count')
    parser.add_argument('--plot_label',default=True, action='store_true', help='plot labels')
    parser.add_argument('--dir_line',action='store_true', help='dir_line')
    opt = parser.parse_args()
    #dfl coco2017
    # opt.weights = '../runs/s-62.15-43.63-v11s-dfl/weights/best.pt' #dfl
    # opt.imgsz = [640,640]
    # opt.source = get_source('','data/coco2017.yaml')
    #txt coco2017
    # opt.weights = '../models/coco2017-txt-yolov11s-txt-bs16-0.619-0.428-epo99-12/weights/best.pt'
    # opt.imgsz = [640,640]
    # opt.source = get_source('','data/coco2017-txt.yaml')
    
    #coco128
    # opt.weights = '../models/coco128-yolov11s-bs16-0.942-0.682-epo99/weights/best.pt'
    # opt.source = ''
    # data = 'data/coco128.yaml'
    #coco128-txt
    # opt.weights = '../models/coco128-txt-yolov11s-txt-bs8-0.978-0.887-epo999-31/weights/best.pt'
    # opt.source = ''
    # data = 'data/coco128-txt.yaml'
    
    #obb dota1.5
    # opt.weights = '../models/dota-yolov11s-obb-bs16-0.801-0.615-epo99-3-640x640/weights/best.pt'
    # opt.source = get_source('','data/dota.yaml')

    #chgdet
    opt.weights = 'runs/train/exp9/weights/best.pt'
    opt.source = get_source('','data/dota-diff.yaml')

    #obb dota1.5-txt
    # opt.weights = 'runs/train/exp3/weights/last.pt'
    # opt.source = get_source('','data/dota-txt.yaml')

    #obb mstar
    # opt.weights = '../models/mstar-99.5-89.7/weights/best.pt'
    # opt.source = get_source('','data/SAR-mstar.yaml')

    #obb SSDD
    # opt.weights = 'runs/train/exp/weights/best.pt'
    # opt.source = get_source('','data/SAR-SSDD.yaml')
    
    opt.save_txt = True
    opt.save_img_count = 200
    return opt


def main(opt):
    print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    # check_requirements(exclude=('tensorboard', 'thop')) #too slow!!!!
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
