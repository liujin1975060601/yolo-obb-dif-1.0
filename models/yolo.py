# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python path/to/models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import sys
from copy import deepcopy
from pathlib import Path

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())  # add yolov5/ to path

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import check_version, make_divisible, check_file, set_logging
from utils.tal import dist2bbox, dist2rbox
from utils.plots import feature_visualization
from utils.torch_utils import time_sync, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr
from .head_text import YoloText,OBBText
from .yolo_base import DetectDFL,OBB, DetectDFL_xn, OBB_xn,HA23
from .LocAtt import LocAtt,LocAttGrouped

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

LOGGER = logging.getLogger(__name__)

OUT_LAYER = {
    'Detect': 0,
    'DetectDFL': 1,
    'OBB': 2,
    'YoloText' : 3,
    'OBBText' : 4,
    'DetectDFL_xn': 1, 
    'OBB_xn': 2,
    'RTDETRDecoder': 10
}

class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), stride=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        # self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)
        self.stride = torch.tensor(stride, requires_grad=False)  # strides computed during build

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):#遍历本层输出的所有anchors
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            #x[i][b=1,na,(no==4+1+nc),ny,nx]
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            #x[i][b=1,na*(no==4+1+nc),ny,nx]-->x[i][b=1,na,ny,nx,no]
            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
                #grid[i][b=1,1,ny,nx,2]

                y = x[i].sigmoid()
                #y[b=1,na,ny,nx,no]
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(1, self.na, 1, 1, 2)  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                # y.view(bs, -1, self.no)的shape是[b=1,[na*ny*nx],(no==4+1+nc)]
                z.append(y.view(bs, -1, self.no))
                # 所有anchors全部一股脑丢给z集合

        return x if self.training else (torch.cat(z, 1), x)


    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)], indexing='ij')
        else:
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)])
        #yv[ny,nx]  xv[ny,nx]
        #torch.stack((xv, yv), 2)-->[ny,nx,2]-->[1,na,ny,nx,2]
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float() #返回grid网格左上角编号
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float() #返回目标anchors的像素数量
        #self.anchors[3,3,2]-->anchor_grid[1,self.na=3,ny, nx,2]
        return grid, anchor_grid

CwCwCwCw = 0
class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=6, nc=None, anchors=None):  # model, input channels, number of classes
        super().__init__()

        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels

        if CwCwCwCw:
            # 添加可学习输入参数CwCwCwCw
            self.cw = CwCwCwCw  # 假设Cw=4
            self.learnable_input = nn.Parameter(torch.randn(self.cw, 640, 640))  # 初始化为随机值，假设默认输入尺寸为640x640
            # 修改输入通道数CwCwCwCw
            ch += self.cw  # 更新输入通道数
            self.yaml['ch'] = ch

        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)
        # LOGGER.info([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        s = 256  # 2x min stride
        m = self.get_module_byname('Detect')  # Detect()水平框输出层
        
        with torch.no_grad():
            if isinstance(m, Detect):
                # stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])[:3]  # forward
                m.inplace = self.inplace
                self.stride = m.stride
                check_anchor_order(m)
                m.anchors /= m.stride.view(-1, 1, 1)
                self._initialize_biases()  # only run once
                # LOGGER.info('Strides: %s' % m.stride.tolist())
            
            for mname in ['DetectDFL', 'OBB', 'YoloText', 'OBBText', 'DetectDFL_xn', 'OBB_xn']:
                m = self.get_module_byname(mname)  # Detect()
                if m is not None:
                    m.inplace = self.inplace
                    self.stride = m.stride
                    self._initialize_biases_dfl(m) #only run once 针对水平框的1(obj)+nc输出做了特殊的bias初始化
                    break
            
            # Init weights, biases
            initialize_weights(self)
            self.eval()
            self.info()
            self.train()
        # LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        #CwCwCwCw
        if CwCwCwCw:
            # 原始输入x的形状: [B,C,H,W]
            b, C, h, w = x.shape
            assert C==3
            # 获取可学习参数并扩展到batch维度
            learnable_part = self.learnable_input.unsqueeze(0).expand(b, -1, -1, -1)  # [B,Cw,H,W]
            # 调整可学习参数的尺寸以匹配输入(如果输入尺寸变化)
            if h != learnable_part.shape[2] or w != learnable_part.shape[3]:
                learnable_part = F.interpolate(learnable_part, size=(h,w), mode='bilinear')
            # 拼接原始输入和可学习部分
            x = torch.cat([x, learnable_part], dim=1)  # [B,3+Cw,H,W]
            
        if augment:
            return self.forward_augment(x)  # augmented inference, None
        return self.forward_once(x, profile, visualize)  # single-scale inference, train

    def forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self.forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        return torch.cat(y, 1), None  # augmented inference, train

    def forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:
                c = isinstance(m, (Detect, DetectDFL))  # copy input as inplace fix
                o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
                t = time_sync()
                for _ in range(10):
                    m(x.copy() if c else x)
                dt.append((time_sync() - t) * 100)
                if m == self.model[0]:
                    LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}")
                LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)

        if profile:
            LOGGER.info('%.1fms total' % sum(dt))
        return x

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _initialize_biases_dfl(self, m):
        for a, b, s in zip(m.cv2, m.cv3, m.stride):
            if isinstance(a[-1], DFLExt):  # from
                dfl_bias_ = a[-1].conv_expand.bias
            #
            elif isinstance(a[-1], nn.Conv2d):
                dfl_bias_ = a[-1].bias
            elif isinstance(a[-2], nn.Conv2d):
                dfl_bias_ = a[-2].bias
            else:
                dfl_bias_  = None
            if dfl_bias_ is not None:
                dfl_bias_.data[:] = 1.0

            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            LOGGER.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             LOGGER.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def autoshape(self):  # add AutoShape module
        LOGGER.info('Adding AutoShape... ')
        m = AutoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)
    
    def get_module_byname(self, name:str):
        if not hasattr(self, 'module_idx'):
            self.module_idx = {
            }
            for i, h in enumerate(self.yaml['head']):
                if h[2] in OUT_LAYER:
                    self.module_idx[h[2]] = i + len(self.yaml['backbone'])
        idx = self.module_idx.get(name, -1)
        return self.model[idx] if idx != -1 else None
    
    def _apply(self, fn):
        self = super()._apply(fn)        
        m = self.get_module_byname('Detect')  # Detect()
        if isinstance(m, Detect):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        for mname in ['DetectDFL', 'OBB', 'YoloText', 'OBBText', 'DetectDFL_xn', 'OBB_xn']:
            m = self.get_module_byname(mname)  # Detect()
            if m is not None:
                m.stride = fn(m.stride)
                m.anchor_points = fn(m.anchor_points)
                m.stride_tensor = fn(m.stride_tensor)
                m.keys = fn(m.keys)
        return self
    

    def predict(self, x, profile=False, visualize=False, batch=None, augment=False, embed=None):
        """
        FOR RT-DETR
        Perform a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor. x[B,C=3,H,W]
            profile (bool): If True, profile the computation time for each layer.
            visualize (bool): If True, save feature maps for visualization.
            batch (dict, optional): Ground truth data for evaluation.
            augment (bool): If True, perform data augmentation during inference.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): Model's output tensor.
        """
        y, dt, embeddings = [], [], []  # outputs
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        for m in self.model[:-1]:  # except the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        head = self.model[-1]
        x = head([y[j] for j in head.f], batch)  # head inference
        return x #x[5][.....]

    def reset_bn_stats(self):
        for m in self.model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.reset_running_stats()

from models.rtdetr.transformer import (
    AIFI,
)
from models.rtdetr.block import (
    RepC3,
    HGBlock,
    HGStem,
    ResNetLayer,
)
from models.rtdetr.rtDetr import (
    RTDETRDecoder
)

def parse_model(d, ch):  # model_dict, input_channels(3)
    LOGGER.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    nc, gd, gw = d['nc'], d.get('depth_multiple',1), d.get('width_multiple',1.0)
    anchors = d.get('anchors', None)
    mc = d.get('max_channels', 10240)

    if anchors is None:
        na = 1
        no = na * (nc + 4)
    else:
        na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
        no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    strides = [1]
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        now_stride = strides[-1]
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                 BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, C3k2, C2PSA, C2f, C3k, A2C2f,LocAtt,LocAttGrouped, RepC3]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                # c2 = make_divisible(c2 * gw, 8)
                c2 = make_divisible(min(c2, mc) * gw, 8)

            args = [c1, c2, *args[1:]]            
            if m in [Focus, Conv, GhostConv, DWConv, GhostBottleneck, MixConv2d, CrossConv]:
                now_stride = strides[f] * args[3]
            if m in [BottleneckCSP, C3, C3TR, C3Ghost, C3k2, C2PSA, C2f, C3k, A2C2f, RepC3]:
                args.insert(2, n)  # number of repeats
                n = 1
                # if m is C3k2: #for M/L/X sizes
                #     if gw >= 1:
                #         args[3] = True
        elif m is nn.Upsample:
            now_stride = int(strides[f] / args[1])
            c2 = ch[f]
        elif m is nn.Identity:
            now_stride = int(strides[f] / 1)
            c2 = ch[f]
        elif m is AIFI:
            args = [ch[f], *args]
        elif m in frozenset({HGStem, HGBlock}):
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)  # number of repeats
                n = 1
        elif m is ResNetLayer:
            c2 = args[1] if args[3] else args[1] * 4
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m in [Detect, DetectDFL, OBB, YoloText, OBBText, DetectDFL_xn, OBB_xn,HA23]:
            args.append([ch[x] for x in f])
            if m not in [DetectDFL, OBB, YoloText, OBBText, DetectDFL_xn, OBB_xn,HA23]:
                if isinstance(args[1], int):  # number of anchors
                    args[1] = [list(range(args[1] * 2))] * len(f)
            else:
                assert args[0] == nc
            args.append([strides[x] for x in f])
        elif m is RTDETRDecoder:  # special case, channels arg must be passed in index 1
            args.insert(1, [ch[x] for x in f])
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]
        strides.append(now_stride)

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n_, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()

    # Profile
    if opt.profile:
        img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
        y = model(img, profile=True)

    # Tensorboard (not working https://github.com/ultralytics/yolov5/issues/2898)
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter('.')
    # LOGGER.info("Run 'tensorboard --logdir=models' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(torch.jit.trace(model, img, strict=False), [])  # add model graph


