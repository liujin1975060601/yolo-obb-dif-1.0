
import torch
import torch.nn as nn
from models.common import *
from utils.tal import dist2bbox, dist2rbox
from utils.general import check_version, make_divisible, check_file, set_logging
from math import factorial

class DetectDFL(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter
    export = False

    def __init__(self, nc=80, legacy=True, ch=(), stride=(), attdfl=0.0):  # detection layer
        super().__init__()        
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.extra_no = 0
        self.no = nc + self.reg_max * (4 + self.extra_no)  # number of outputs per anchor
        self.stride = torch.tensor(stride, requires_grad=False)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        # self.cv2 = nn.ModuleList( #box branch->4(box) * self.reg_max
        #     nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        # )
        self.keys = torch.arange(0, self.reg_max, requires_grad=False).view(1, -1).repeat((4 + self.extra_no, 1)).float()
        self.anchor_points = torch.zeros(1, requires_grad=False)  # init grid
        self.stride_tensor = torch.zeros(1, requires_grad=False)  # init grid
        self.cv2 = (#box branch->4(box) * self.reg_max
            nn.ModuleList(
                nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), 
                              nn.Conv2d(c2, 4 * self.reg_max, 1, bias=True) # -> 4(box) * self.reg_max
                             )
                if attdfl<=0 else
                nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), 
                              nn.Conv2d(c2, 4 * (2 * self.reg_max), 1, bias=True),
                              AttnDFL(attdfl) #4 * (2 * self.reg_max) -> 4(box) * self.reg_max
                             )
              for x in ch)
        )

        self.legacy = legacy
        self.cv3 = ( #class branch->nc
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),
                )
                for x in ch
            )
        )
        self.init_flag = False
        self.register_buffer('proj', torch.arange(self.reg_max, dtype=torch.float))

    def _forward(self, x):
        ys = []
        dists = []
        clses = []
        b, _, _, _ = x[0].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
        # if self.onnx_dynamic or self.anchor_points.shape[0] != sum([x[i].shape[2]*x[i].shape[3] for i in range(self.nl)]):#初始时分配网格坐标
        self.anchor_points, self.stride_tensor = make_anchors(x, self.stride, 0.5)
        for i in range(self.nl):
            y, dist = [o.permute(0, 2, 3, 1) for o in self.cv2[i](x[i])]
            cls_ = self.cv3[i](x[i]).permute((0, 2, 3, 1))
            # x[i] = torch.cat((dist, cls_), -1) # cv2[i](x[i])[B,C=reg_max*4,H,W] for box, cv3[i](x[i])[B,C=nc,H,W] for cls
            dists.append(dist)
            ys.append(y)
            clses.append(cls_)
        clses = torch.cat([x_.view(b, -1, self.nc) for x_ in clses], 1)
        dists = torch.cat([x_.view(b, -1, self.no - self.nc) for x_ in dists], 1)
        ys = torch.cat([y.view(b, -1, 4) for y in ys], 1)
        return dists, clses, ys

    def dfl(self, x):
        b, c, a = x.shape  # batch, channels, anchors
        # x = x.view(b, 4, c // 4, a).transpose(2, 1).softmax(1).matmul(self.proj.type(x.dtype)).view(b, 4, a)
        if getattr(self, 'proj', None) is None:
            self.proj = torch.arange(self.reg_max, dtype=x.dtype).to(x.device)
        x = F.conv2d(x.view(b, 4, c // 4, a).transpose(2, 1).softmax(1), self.proj.view(1, -1, 1, 1)).view(b, 4, a)
        return x

    def forward(self, x): #x[3][B,C,H,W]
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1) #x[i][B,C,H,W]->
        if self.training:  # Training path
            return x
        shape = x[0].shape  # BCHW
        self.anchor_points, self.stride_tensor = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2) #x_cat[b,reg_max*4 + nc,ntotal]
        box, clses = x_cat.split((self.reg_max * 4, self.nc), 1) #box[b,reg_max*4,ntotal] clses[b,nc,ntotal] 
        dbox = dist2bbox(self.dfl(box), self.anchor_points.unsqueeze(0), xywh=True, dim=1) * self.stride_tensor
        return (torch.cat((dbox, clses.sigmoid()), 1).transpose(2, 1), x) #dbox[b,reg_max*4,ntotal] + clses[b,nc,ntotal] --> y[b,reg_max*4+nc,ntotal]--> y[b,ntotal,reg_max*4+nc]
        # dists, clses, ys = self._forward(x)
        # if not self.export:
        #     y = dist2bbox(ys, self.anchor_points, xywh=False, dim=-1)
        #     train_out = (dists, clses, y)
        #     if self.training:  # Training path
        #         return train_out
        #     else:
        #         y1 = dist2bbox(ys, self.anchor_points, xywh=True, dim=-1)
        #         return torch.cat((y1 * self.stride_tensor, clses.sigmoid()), -1), train_out
        # else:
        #     y1 = dist2bbox(ys, self.anchor_points, xywh=True, dim=-1)
        #     return torch.cat((y1 * self.stride_tensor, clses.sigmoid()), -1)
          
    def compute_ecloss_dim3(self, dist, gt, mask):
        # 使用中间 softmax 分布 dist ([B, All(H*W), C*K])
        # 和标注 gt ([B, All(H*W), C]) 来计算 EcLoss。
        # 其中 self.keys.shape=[C,K] 每个通道 c 的关键值。
        # dist.shape=[B*C, K, H, W]，下标 i=(b*C + c)
        # keys.shape=[C, K]
        C, K = self.keys.shape
        device = self.keys.device
        # 只取 mask 
        dist_mask = dist[mask].view(-1, C, K)  # [nt, c*k]
        gt_mask = gt[mask]  # [nt, c]
        nt = dist_mask.shape[0]
        Kmin, Kmax = self.keys.min(-1)[0].repeat((gt_mask.shape[0], 1)), self.keys.max(-1)[0].repeat((gt_mask.shape[0], 1))
        
        # gt_clamped = torch.clamp(gt_mask, min=channel_keys[0], max=channel_keys[-1])
        gt_clamped = torch.maximum(Kmin, torch.minimum(gt_mask, Kmax)).transpose(1, 0).contiguous()  # [c, nt]
        pos = torch.stack([torch.searchsorted(self.keys[i], gt_clamped[i], right=False) for i in range(C)]).transpose(1, 0).contiguous()    # [nt, c]
        pos = torch.clamp(pos, 1, self.reg_max-1)
        left = pos - 1           # [nt, c]
        right= pos               # [nt, c]
        # [nt, c] --> [nt, c, 3(nt_idx, c_idx, k_idx)] --> [nt * c, 3] --> 3 * [nt*c]
        left = torch.cat([torch.arange(nt, device=device).view(-1, 1, 1).repeat(1, C, 1),
                          torch.arange(C, device=device).view(1, -1, 1).repeat(nt, 1, 1),
                          left.view(nt, C, 1)], dim=-1).long().view(-1, 3).split(1, dim=-1)
        # [nt, c] --> [nt, c, 3(nt_idx, c_idx, k_idx)] --> [nt * c, 3] --> 3 * [nt*c]
        right = torch.cat([torch.arange(nt, device=device).view(-1, 1, 1).repeat(1, C, 1),
                          torch.arange(C, device=device).view(1, -1, 1).repeat(nt, 1, 1),
                          right.view(nt, C, 1)], dim=-1).long().view(-1, 3).split(1, dim=-1)

        # p_left  = dist_mask[left[0], left[1], left[2]].view(-1, C)
        # p_right = dist_mask[right[0], right[1], right[2]].view(-1, C)

        # 准备一个容器存各 (b,c) 的 EcLoss
        
        left_val  = self.keys[left[1], left[2]].view(-1, C)   # [nt, c]
        right_val = self.keys[right[1], right[2]].view(-1, C)   # [nt, c]

        denom = (right_val - left_val).clamp_min(1e-8)  # 避免除0
        val_clamped = gt_clamped.transpose(1, 0).contiguous().detach()
        w_left  = (right_val - val_clamped) / denom
        w_right = (val_clamped - left_val) / denom
    
        # eps = 1e-12
        # ce_map = -( w_left  * torch.log(p_left  + eps) +
        #             w_right * torch.log(p_right + eps) )
        ce_map_1 =(F.cross_entropy(dist_mask.view(-1, K), left[2].view(-1), reduction="none").view(nt, C) * w_left
                   + F.cross_entropy(dist_mask.view(-1, K), right[2].view(-1), reduction="none").view(nt, C) * w_right
                   ).mean(-1, keepdim=True)

        ec_loss_map = ce_map_1

        return ec_loss_map
    
    def update_dfl_keys_base(self, keys:list): #keys[rgmax]
        with torch.no_grad():
            self.proj = torch.tensor(keys, device=self.proj.device)

class OBB(DetectDFL):
    """YOLO OBB detection head for detection with rotation models."""

    def __init__(self, nc=80, legacy=True, ch=(), stride=(), attdfl=0.0):
        """Initialize OBB with number of classes `nc` and layer channels `ch`."""
        super().__init__(nc, legacy, ch, stride, attdfl)
        self.ne = 1  # number of extra parameters
        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for x in ch)

    def forward(self, x): #x[nl=3][B,C=16*4(rbox)+nc,H,W]
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        bs = x[0].shape[0]  # batch size
        angle = torch.cat([self.cv4[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2) #angle[B,1,ntotal]
        # NOTE: set `angle` as an attribute so that `decode_bboxes` could use it.
        angle = ((angle.sigmoid() - 0.25) * math.pi)#.permute((0, 2, 1)) #->angle[B,1,ntotal]
        # angle = angle.sigmoid() * math.pi / 2  # [0, pi/2]
        
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return x, angle #x[nl=3][B,C=4*16+nc,H,W] angle[B,1,ntotal]
        shape = x[0].shape  # BCHW
        self.anchor_points, self.stride_tensor = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5)) #anchor_points[2,ntotal] stride_tensor[1,ntotal]
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2) #x_cat[B,C=4*16+nc,ntotal]
        box, clses = x_cat.split((self.reg_max * 4, self.nc), 1)#x_cat[B,C=4*16+nc,ntotal]->box[B,4*16,ntotal] + clses[B,nc,ntotal]
        dbox = dist2rbox(self.dfl(box), angle, self.anchor_points.unsqueeze(0), dim=1) * self.stride_tensor #->dbox[b,4,ntotal]
        return (torch.cat((dbox, angle, clses.sigmoid()), 1).transpose(2, 1), (x, angle))
                #dbox[b,4,ntotal] + angle[b,1,ntotal] + clses[b,nc,ntotal] -> [b,4+1+nc,ntotal] -> [b,ntotal,4(dbox)+1(angle)+nc]
        # if not self.training:
        #     self.angle = angle
        # dists, clses, ys = self._forward(x)
        # y = self.dist2rbox(ys, angle, self.anchor_points, dim=-1)
        # if not self.export:
        #     train_out = (dists, clses, torch.cat([y, angle], -1))
        #     if self.training:  # Training path
        #         return train_out
        #     else:
        #         return torch.cat((y * self.stride_tensor, angle, clses.sigmoid()), -1), train_out
        # else:
        #     return torch.cat((y * self.stride_tensor, angle, clses.sigmoid()), -1)

class HA23(DetectDFL):
    """YOLO OBB detection head for detection with rotation models."""

    def __init__(self, nc=80, legacy=True, ch=(), stride=(), attdfl=0.0):
        """Initialize OBB with number of classes `nc` and layer channels `ch`."""
        super().__init__(nc, legacy, ch, stride, attdfl)
        self.ne = 1  # number of extra parameters
        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for x in ch)

    def forward(self, x): #x[nl=3][B,C=16*4(rbox)+nc,H,W]
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        bs = x[0].shape[0]  # batch size
        angle = torch.cat([self.cv4[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2) #angle[B,1,ntotal]
        # NOTE: set `angle` as an attribute so that `decode_bboxes` could use it.
        angle = ((angle.sigmoid() - 0.25) * math.pi)#.permute((0, 2, 1)) #->angle[B,1,ntotal]
        # angle = angle.sigmoid() * math.pi / 2  # [0, pi/2]
        
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return x, angle #x[nl=3][B,C=4*16+nc,H,W] angle[B,1,ntotal]
        shape = x[0].shape  # BCHW
        self.anchor_points, self.stride_tensor = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5)) #anchor_points[2,ntotal] stride_tensor[1,ntotal]
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2) #x_cat[B,C=4*16+nc,ntotal]
        box, clses = x_cat.split((self.reg_max * 4, self.nc), 1)#x_cat[B,C=4*16+nc,ntotal]->box[B,4*16,ntotal] + clses[B,nc,ntotal]
        dbox = dist2rbox(self.dfl(box), angle, self.anchor_points.unsqueeze(0), dim=1) * self.stride_tensor #->dbox[b,4,ntotal]
        return (torch.cat((dbox, angle, clses.sigmoid()), 1).transpose(2, 1), (x, angle))
                #dbox[b,4,ntotal] + angle[b,1,ntotal] + clses[b,nc,ntotal] -> [b,4+1+nc,ntotal] -> [b,ntotal,4(dbox)+1(angle)+nc]
        # if not self.training:
        #     self.angle = angle
        # dists, clses, ys = self._forward(x)
        # y = self.dist2rbox(ys, angle, self.anchor_points, dim=-1)
        # if not self.export:
        #     train_out = (dists, clses, torch.cat([y, angle], -1))
        #     if self.training:  # Training path
        #         return train_out
        #     else:
        #         return torch.cat((y * self.stride_tensor, angle, clses.sigmoid()), -1), train_out
        # else:
        #     return torch.cat((y * self.stride_tensor, angle, clses.sigmoid()), -1)

def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            sy, sx = torch.meshgrid(sy, sx, indexing='ij')
        else:
            sy, sx = torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)