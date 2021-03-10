import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

from model.yolo_loss import loss
from model.darknet import darknet19


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, padding=0,  stride=1, dilation=1, leakyReLU=False):
        super(Conv2d, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, padding=padding, stride=stride, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True) if leakyReLU else nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convs(x)


class ReorgLayer(nn.Module):
    """
    特征重排：[bs, 64, 26, 26]-->[bs, 256, 13, 13]
    """
    def __init__(self, stride):
        super(ReorgLayer, self).__init__()
        self.stride = stride

    def forward(self, x):
        b, c, h_, w_ = x.size()
        h = h_ // self.stride
        w = w_ // self.stride
        x = x.view(b, c, h, self.stride, w, self.stride).transpose(3, 4).contiguous()
        x = x.view(b, c, h * w, self.stride * self.stride).transpose(2, 3).contiguous()
        x = x.view(b, c, self.stride * self.stride, h, w).transpose(1, 2).contiguous()
        x = x.view(b, -1, h, w)
        return x


class SPP(nn.Module):
    """
        Spatial Pyramid Pooling
    """
    def __init__(self, in_ch, out_ch):
        super(SPP, self).__init__()
        self.fuse_conv = Conv2d(in_ch * 4, out_ch, 1, leakyReLU=True)

    def forward(self, x):
        x_1 = F.max_pool2d(x, 5, stride=1, padding=2)
        x_2 = F.max_pool2d(x, 9, stride=1, padding=4)
        x_3 = F.max_pool2d(x, 13, stride=1, padding=6)
        x = torch.cat([x, x_1, x_2, x_3], dim=1)

        return self.fuse_conv(x)


class SAM(nn.Module):
    """ Parallel CBAM """

    def __init__(self, in_ch):
        super(SAM, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """ Spatial Attention Module """
        x_attention = self.conv(x)

        return x * x_attention


class YOLO(nn.Module):
    def __init__(self, device=None, num_classes=None, input_size=None, trainable=True,
                 conf_thresh=0.01, nms_thresh=0.5, anchor_size=None):
        super(YOLO, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.trainable = trainable
        self.stride = 32
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.anchor_size = torch.tensor(anchor_size)
        self.anchor_num = len(anchor_size)
        self.input_size = input_size    # [w, h]
        self.scale = np.array([[[input_size[0], input_size[1], input_size[0], input_size[1]]]])
        self.scale_torch = torch.tensor(self.scale.copy()).float().to(device)
        self.set_grid(input_size)

        # 网络定义
        self.backbone = darknet19(pretrained=False)

        # detection head
        self.convsets_1 = nn.Sequential(
            Conv2d(1024, 1024, 3, 1, leakyReLU=True),
            Conv2d(1024, 1024, 3, 1, leakyReLU=True)
        )

        self.route_layer = Conv2d(512, 64, 1, leakyReLU=True)
        self.reorg = ReorgLayer(stride=2)

        self.convsets_2 = Conv2d(1280, 1024, 3, 1, leakyReLU=True)

        # prediction layer
        self.pred = nn.Conv2d(1024, self.anchor_num * (1 + 4 + self.num_classes), 1)

    # def decode_boxes(self, txtytwth_pred):
    #     """
    #         Input:
    #             txtytwth_pred : [B, H*W*anchor_num, 4] containing [tx, ty, tw, th]
    #         Output:
    #             xywh_pred : [B, H*W*anchor_num, 4] containing [xmin, ymin, xmax, ymax]
    #     """
    #     # [B, H * W * anchor_num, 4] --> [B, H*W, anchor_num, 4]
    #     txtytwth_pred = txtytwth_pred.view(txtytwth_pred.size(0), -1, self.anchor_num, 4)
    #
    #     # 生成每个格子的坐标位置
    #     w_grid, h_grid = self.input_size[0] // self.stride, self.input_size[1] // self.stride
    #     grid_y, grid_x = torch.meshgrid([torch.arange(h_grid), torch.arange(w_grid)])
    #     grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
    #     grid_cell = grid_xy.view(1, h_grid * w_grid, 1, 2).to(self.device)
    #     # 生成anchor_wh tensor: [1, w_grid * h_grid, 5, 2]
    #     anchor_wh = self.anchor_size.repeat(h_grid * w_grid, 1, 1).unsqueeze(0).to(self.device)
    #
    #     # [tx, ty, tw, th]-->[bx, by, bw, bh]
    #     # b_x = sigmoid(tx) + grid_x
    #     # b_y = sigmoid(ty) + grid_y
    #     # b_w = anchor_w * exp(tw)
    #     # b_h = anchor_h * exp(th)
    #     xy_pred = torch.sigmoid(txtytwth_pred[:, :, :, :2]) + grid_cell
    #     wh_pred = torch.exp(txtytwth_pred[:, :, :, 2:]) * anchor_wh
    #     # [B, H*W, anchor_n, 4] -> [B, H*W*anchor_n, 4]
    #     xywh_pred = torch.cat([xy_pred, wh_pred], -1).view(txtytwth_pred.size(0), -1, 4) * self.stride
    #
    #     # [bx, by, bw, bh] -> [xmin, ymin, xmax, ymax]
    #     x1y1x2y2_pred = torch.zeros_like(xywh_pred)
    #     x1y1x2y2_pred[:, :, 0] = (xywh_pred[:, :, 0] - xywh_pred[:, :, 2] / 2)
    #     x1y1x2y2_pred[:, :, 1] = (xywh_pred[:, :, 1] - xywh_pred[:, :, 3] / 2)
    #     x1y1x2y2_pred[:, :, 2] = (xywh_pred[:, :, 0] + xywh_pred[:, :, 2] / 2)
    #     x1y1x2y2_pred[:, :, 3] = (xywh_pred[:, :, 1] + xywh_pred[:, :, 3] / 2)
    #     return xywh_pred
    def create_grid(self, input_size):
        w, h = input_size[1], input_size[0]
        # generate grid cells
        ws, hs = w // self.stride, h // self.stride
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
        grid_xy = grid_xy.view(1, hs * ws, 1, 2).to(self.device)

        # generate anchor_wh tensor
        anchor_wh = self.anchor_size.repeat(hs * ws, 1, 1).unsqueeze(0).to(self.device)

        return grid_xy, anchor_wh

    def set_grid(self, input_size):
        self.input_size = input_size
        self.grid_cell, self.all_anchor_wh = self.create_grid(input_size)
        self.scale = np.array([[[input_size[1], input_size[0], input_size[1], input_size[0]]]])
        self.scale_torch = torch.tensor(self.scale.copy(), device=self.device).float()


    def decode_xywh(self, txtytwth_pred):
        """
            Input:
                txtytwth_pred : [B, H*W, anchor_n, 4] containing [tx, ty, tw, th]
            Output:
                xywh_pred : [B, H*W*anchor_n, 4] containing [xmin, ymin, xmax, ymax]
        """
        # b_x = sigmoid(tx) + gride_x,  b_y = sigmoid(ty) + gride_y
        B, HW, ab_n, _ = txtytwth_pred.size()
        xy_pred = torch.sigmoid(txtytwth_pred[:, :, :, :2]) + self.grid_cell
        # b_w = anchor_w * exp(tw),     b_h = anchor_h * exp(th)
        wh_pred = torch.exp(txtytwth_pred[:, :, :, 2:]) * self.all_anchor_wh
        # [H*W, anchor_n, 4] -> [H*W*anchor_n, 4]
        xywh_pred = torch.cat([xy_pred, wh_pred], -1).view(B, HW * ab_n, 4) * self.stride

        return xywh_pred

    def decode_boxes(self, txtytwth_pred, requires_grad=False):
        """
            Input:
                txtytwth_pred : [B, H*W, anchor_n, 4] containing [tx, ty, tw, th]
            Output:
                x1y1x2y2_pred : [B, H*W*anchor_n, 4] containing [xmin, ymin, xmax, ymax]
        """
        # [H*W*anchor_n, 4]
        xywh_pred = self.decode_xywh(txtytwth_pred)

        # [center_x, center_y, w, h] -> [xmin, ymin, xmax, ymax]
        x1y1x2y2_pred = torch.zeros_like(xywh_pred, requires_grad=requires_grad)
        x1y1x2y2_pred[:, :, 0] = (xywh_pred[:, :, 0] - xywh_pred[:, :, 2] / 2)
        x1y1x2y2_pred[:, :, 1] = (xywh_pred[:, :, 1] - xywh_pred[:, :, 3] / 2)
        x1y1x2y2_pred[:, :, 2] = (xywh_pred[:, :, 0] + xywh_pred[:, :, 2] / 2)
        x1y1x2y2_pred[:, :, 3] = (xywh_pred[:, :, 1] + xywh_pred[:, :, 3] / 2)

        return x1y1x2y2_pred


    def iou_score(self, bboxes_a, bboxes_b):
        """
            bbox_1 : [B*N, 4] = [x1, y1, x2, y2]
            bbox_2 : [B*N, 4] = [x1, y1, x2, y2]
        """
        tl = torch.max(bboxes_a[:, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en  # * ((tl < br).all())
        return area_i / (area_a + area_b - area_i)

    def nms(self, dets, scores):
        """"Pure Python NMS baseline."""
        x1 = dets[:, 0]  # xmin
        y1 = dets[:, 1]  # ymin
        x2 = dets[:, 2]  # xmax
        y2 = dets[:, 3]  # ymax

        areas = (x2 - x1) * (y2 - y1)  # the size of bbox
        order = scores.argsort()[::-1]  # sort bounding boxes by decreasing order

        keep = []  # store the final bounding boxes
        while order.size > 0:
            i = order[0]  # the index of the bbox with highest confidence
            keep.append(i)  # save it to keep
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(1e-28, xx2 - xx1)
            h = np.maximum(1e-28, yy2 - yy1)
            inter = w * h

            # Cross Area / (bbox + particular area - Cross Area)
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            # reserve all the boundingbox whose ovr less than thresh
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep

    def postprocess(self, all_local, all_conf):
        """
        bbox_pred: (HxW*anchor_n, 4), bsize = 1
        prob_pred: (HxW*anchor_n, num_classes), bsize = 1
        """
        bbox_pred = all_local
        prob_pred = all_conf

        # get score
        cls_inds = np.argmax(prob_pred, axis=1)
        prob_pred = prob_pred[(np.arange(prob_pred.shape[0]), cls_inds)]
        scores = prob_pred.copy()

        # confidence threshold filter
        thr_keep = np.where(scores >= self.conf_thresh)
        bbox_pred = bbox_pred[thr_keep]
        scores = scores[thr_keep]
        cls_inds = cls_inds[thr_keep]

        # NMS filter
        keep = np.zeros(len(bbox_pred), dtype=np.int)
        for i in range(self.num_classes):
            inds = np.where(cls_inds == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bbox_pred[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1

        nms_keep = np.where(keep > 0)

        # final pred
        bbox_pred = bbox_pred[nms_keep]
        scores = scores[nms_keep]
        cls_inds = cls_inds[nms_keep]

        return bbox_pred, scores, cls_inds

    def forward(self, x, target=None):
        # backbone
        _, c5, c6 = self.backbone(x)    # [bs, 512, 26, 26], [bs, 1024, 13, 13]
        c6 = self.convsets_1(c6)  # [bs, 1024, 13, 13]

        # route from c6 feature map
        c5 = self.reorg(self.route_layer(c5))  # [bs, 512, 26, 26]-->[bs, 64, 26, 26]-->[bs, 256, 13, 13]

        # route concatenate
        feature_map = self.convsets_2(torch.cat([c5, c6], dim=1))   # [bs, 1024, 13, 13]

        # pred
        prediction = self.pred(feature_map)   # [bs, anchor_num * (5 + num_classes), 13, 13]

        B, _, H, W = prediction.size()
        prediction = prediction.permute(0, 2, 3, 1).contiguous().view(B, H * W * self.anchor_num, 5 + self.num_classes)
        # 分离预测值
        conf_pred = prediction[:, :, :1]     # [B, H*W*anchor_num, 1]
        cls_pred = prediction[:, :, 5:]     # [B, H*W*anchor_num, num_cls]
        txtytwth_pred = prediction[:, :, 1:5].view(B, H*W, self.anchor_num, 4)    # [B, H*W*anchor_num, 4]

        # train
        if self.trainable:
            # 训练标签：
            # txtytwth_pred[tx, ty, tw, th] --> [bx, by, bw, bh] --> x1y1x2y2_pred[xmin. ymin, xmax, ymax]
            x1y1x2y2_pred = (self.decode_boxes(txtytwth_pred) / self.scale_torch).view(-1, 4)
            x1y1x2y2_gt = target[:, :, 7:].view(-1, 4)
            iou = self.iou_score(x1y1x2y2_pred, x1y1x2y2_gt).view(B, H * W * self.anchor_num, 1)
            target = torch.cat([iou, target[:, :, :7]], dim=2)
            # print(iou.min(), iou.max())

            txtytwth_pred = txtytwth_pred.view(B, H*W*self.anchor_num, 4)

            conf_loss, cls_loss, txtytwth_loss, total_loss = loss(pred_conf=conf_pred, pred_cls=cls_pred,
                                                                  pred_txtytwth=txtytwth_pred, label=target)

            return conf_loss, cls_loss, txtytwth_loss, total_loss

        else:
            with torch.no_grad():
                # batch size = 1
                all_obj = torch.sigmoid(conf_pred)[0]  # 0 is because that these is only 1 batch.
                all_bbox = torch.clamp((self.decode_boxes(txtytwth_pred) / self.scale_torch)[0], 0., 1.)
                all_class = (torch.softmax(cls_pred[0, :, :], 1) * all_obj)

                # separate box pred and class conf
                all_bbox = all_bbox.to('cpu').numpy()
                all_class = all_class.to('cpu').numpy()

                bboxes, scores, cls_inds = self.postprocess(all_bbox, all_class)

                return bboxes, scores, cls_inds


if __name__ == "__main__" :
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    ANCHOR_SIZE = [[1.19, 1.98], [2.79, 4.59], [4.53, 8.92], [8.06, 5.29], [10.32, 10.65]]
    yolo = YOLO(device=device, num_classes=20, input_size=[416, 416], trainable=False, anchor_size=ANCHOR_SIZE).to(device)
    input = torch.rand((2, 3, 416, 416)).to(device)
    output = yolo(input)
