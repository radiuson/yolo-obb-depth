import torch
import torch.nn as nn
import numpy as np

import math
from ultralytics.utils.tal import TORCH_1_10, dist2bbox, dist2rbox, make_anchors
from ultralytics.nn.modules import Conv, C2f, SPPF, Concat, DWConv, DFL
from ultralytics.utils.ops import make_divisible
W = 0.25
D = 0.33
class YOLOv8OBBDepth_backbone(torch.nn.Module):
    def __init__(self):
        super(YOLOv8OBBDepth_backbone, self).__init__()

        self.layer0 = Conv(3, make_divisible(64 * W,8), 3, 2)
        self.layer1 = Conv(make_divisible(64 * W,8), make_divisible(128 * W,8), 3, 2)
        self.layer2 = C2f(make_divisible(128 * W,8), make_divisible(128 * W,8), n=make_divisible(3 * D,8), shortcut=True)

        self.layer3 = Conv(make_divisible(128 * W,8), make_divisible(256 * W,8), 3, 2)
        self.layer4 = C2f(make_divisible(256 * W,8), make_divisible(256 * W,8), n=make_divisible(6 * D,8), shortcut=True)

        self.layer5 = Conv(make_divisible(256 * W,8), make_divisible(512 * W,8), 3, 2)
        self.layer6 = C2f(make_divisible(512 * W,8), make_divisible(512 * W,8), n=make_divisible(6 * D,8), shortcut=True)

        self.layer7 = Conv(make_divisible(512 * W,8), make_divisible(1024 * W,8), 3, 2)
        self.layer8 = C2f(make_divisible(1024 * W,8), make_divisible(1024 * W,8), n=make_divisible(3 * D,8), shortcut=True)

        self.layer9 = SPPF(make_divisible(1024 * W,8), make_divisible(1024 * W,8), k=5)


    def forward(self, x):
        # Forward pass
        x = self.layer0(x)     # 0
        x = self.layer1(x)     # 1
        x = self.layer2(x)     # 2
        x = self.layer3(x)     # 3
        p3 = self.layer4(x)     # 4
        x = self.layer5(p3)     # 5
        p4 = self.layer6(x)     # 6
        x = self.layer7(p4)     # 7
        x = self.layer8(x)     # 8
        p5 = self.layer9(x)     # 9
        
        return p3,p4,p5
    
class YOLOv8OBBDepth_neck(torch.nn.Module):
    def __init__(self, ch):  # ch = [256, 512, 1024] from backbone output
        super().__init__()

        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.concat1 = Concat(1)
        self.c2f1 = C2f(ch[2] + ch[1], make_divisible(512 * W,8), n=make_divisible(3*D,8))

        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.concat2 = Concat(1)
        self.c2f2 = C2f(make_divisible(512 * W,8) + ch[0], make_divisible(256 * W,8), n=make_divisible(3*D,8))

        self.downsample1 = Conv(make_divisible(256 * W,8), make_divisible(256 * W,8), k=3, s=2)
        self.concat3 = Concat(1)
        self.c2f3 = C2f(make_divisible(256 * W,8) + make_divisible(512 * W,8), make_divisible(512 * W,8), n=make_divisible(3*D,8))

        self.downsample2 = Conv(make_divisible(512 * W,8), make_divisible(512 * W,8), k=3, s=2)
        self.concat4 = Concat(1)
        self.c2f4 = C2f(make_divisible(512 * W,8) + make_divisible(1024 * W,8), make_divisible(1024 * W,8), n=make_divisible(3*D,8))

    def forward(self, p3, p4, p5):
        x = self.upsample1(p5)
        x = self.concat1([x, p4])
        x = self.c2f1(x)

        h3 = self.upsample2(x)
        h3 = self.concat2([h3, p3])
        h3 = self.c2f2(h3)  # P3/80*80

        h4 = self.downsample1(h3)
        h4 = self.concat3([h4, x])
        h4 = self.c2f3(h4)    # P4/16

        h5 = self.downsample2(h4)
        h5 = self.concat4([h5, p5])
        h5 = self.c2f4(h5)  # 最大尺度 P5/32

        return h3, h4, h5


class Detect(nn.Module):
    """YOLO Detect head for detection models."""

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    format = None  # export format
    end2end = False  # end2end
    max_det = 300  # max_det
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init
    legacy = False  # backward compatibility for v3/v5/v8/v9 models
    xyxy = False  # xyxy or xywh output

    def __init__(self, nc=80, ch=()):
        """Initialize the YOLO detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = (
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
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if self.end2end:
            return self.forward_end2end(x)

        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def forward_end2end(self, x):
        """
        Performs forward pass of the v10Detect module.

        Args:
            x (List[torch.Tensor]): Input feature maps from different levels.

        Returns:
            (dict | tuple):

                - If in training mode, returns a dictionary containing outputs of both one2many and one2one detections.
                - If not in training mode, returns processed detections or a tuple with processed detections and raw outputs.
        """
        x_detach = [xi.detach() for xi in x]
        one2one = [
            torch.cat((self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1) for i in range(self.nl)
        ]
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return {"one2many": x, "one2one": one2one}

        y = self._inference(one2one)
        y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)
        return y if self.export else (y, {"one2many": x, "one2one": one2one})

    def _inference(self, x):
        """
        Decode predicted bounding boxes and class probabilities based on multiple-level feature maps.

        Args:
            x (List[torch.Tensor]): List of feature maps from different detection layers.

        Returns:
            (torch.Tensor): Concatenated tensor of decoded bounding boxes and class probabilities.
        """
        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.format != "imx" and (self.dynamic or self.shape != shape):
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        elif self.export and self.format == "imx":
            dbox = self.decode_bboxes(
                self.dfl(box) * self.strides, self.anchors.unsqueeze(0) * self.strides, xywh=False
            )
            return dbox.transpose(1, 2), cls.sigmoid().permute(0, 2, 1)
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
        if self.end2end:
            for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes, anchors, xywh=True):
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, xywh=xywh and not (self.end2end or self.xyxy), dim=1)

    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80):
        """
        Post-processes YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc) with last dimension
                format [x, y, w, h, class_probs].
            max_det (int): Maximum detections per image.
            nc (int, optional): Number of classes. Default: 80.

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6) and last
                dimension format [x, y, w, h, max_class_prob, class_index].
        """
        batch_size, anchors, _ = preds.shape  # i.e. shape(16,8400,84)
        boxes, scores = preds.split([4, nc], dim=-1)
        index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(min(max_det, anchors))
        i = torch.arange(batch_size)[..., None]  # batch indices
        return torch.cat([boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1)


class OBB(Detect):
    """YOLO OBB detection head for detection with rotation models."""

    def __init__(self, nc=80, ne=1, ch=()):
        """Initialize OBB with number of classes `nc` and layer channels `ch`."""
        super().__init__(nc, ch)
        self.ne = ne  # number of extra parameters

        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for x in ch)

    def forward(self, x):
        # for i in range(self.nl):
        #     print(f"x[i].shape:{x[i].shape}")

        """Concatenates and returns predicted bounding boxes and class probabilities."""
        bs = x[0].shape[0]  # batch size
        angle = torch.cat([self.cv4[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2)  # OBB theta logits
        # NOTE: set `angle` as an attribute so that `decode_bboxes` could use it.
        angle = (angle.sigmoid() - 0.25) * math.pi  # [-pi/4, 3pi/4]
        # angle = angle.sigmoid() * math.pi / 2  # [0, pi/2]
        if not self.training:
            self.angle = angle
        x = Detect.forward(self, x)
        if self.training:
            return x, angle
        return torch.cat([x, angle], 1) if self.export else (torch.cat([x[0], angle], 1), (x[1], angle))

    def decode_bboxes(self, bboxes, anchors):
        """Decode rotated bounding boxes."""
        return dist2rbox(bboxes, self.angle, anchors, dim=1)
    
class Depth(nn.Module):
    def __init__(self,ch=()):
        super(Depth, self).__init__()
        self.nl = len(ch)
        c5 = max(ch[0] // 4, 1)
        self.cv = nn.ModuleList(
            nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, 1, 1)) for x in ch
        )

    def forward(self, x):
        for i in range(self.nl):
            x[i] = self.cv[i](x[i])
        if self.training:
            return x
        y = x[0]
        return y

class YOLOv8OBBDepthModel(nn.Module):
    def __init__(self, nc=80):
        super().__init__()
        ch = [make_divisible(256*W, 8), make_divisible(512*W, 8), make_divisible(1024*W, 8)]
        self.backbone = YOLOv8OBBDepth_backbone()
        self.neck = YOLOv8OBBDepth_neck(ch)
        self.head = OBB(nc=nc, ch=ch)
        self.depth = Depth(ch=ch)

    def forward(self, x):
        p3, p4, p5 = self.backbone(x)
        h3, h4, h5 = self.neck(p3, p4, p5)
        det_out = self.head([h3, h4, h5])   # (preds, (raw_x, angle))
        depth_out = self.depth([h3, h4, h5])
        return det_out, depth_out

from ultralytics.nn.tasks import BaseModel

class DetectionModelWithDepth(BaseModel):
    def __init__(self, cfg=None, ch=3, nc=80, verbose=True):
        super().__init__()
        self.model = YOLOv8OBBDepthModel(nc=nc)
        self.save = []
        self.names = {i: f"{i}" for i in range(nc)}
        self.inplace = True

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    backbone = YOLOv8OBBDepth_backbone()
    x = torch.randn(16, 3, 640, 640)
    p3,p4,p5 = backbone(x)
    
    neck = YOLOv8OBBDepth_neck([make_divisible(256*W,8), make_divisible(512*W,8), make_divisible(1024*W,8)])
    h3, h4, h5 = neck(p3, p4, p5)

    head = OBB(nc=80,ch = [make_divisible(256*W,8), make_divisible(512*W,8), make_divisible(1024*W,8)])
    result = head([h3, h4, h5])

    depth = Depth(ch = [make_divisible(256*W,8), make_divisible(512*W,8), make_divisible(1024*W,8)])
    depth_result = depth([h3, h4, h5])
    # print("Detection Output Shape:", result[0][0].shape)

    
