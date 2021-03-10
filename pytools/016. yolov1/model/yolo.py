import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

from model.yolo_loss import loss


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride=1, padding=0, dilation=1, leakyReLU=False):
        super(Conv2d, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True) if leakyReLU else nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convs(x)


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
    def __init__(self, device=None, num_classes=None, input_size=None, trainable=True, conf_thresh=0.01, nms_thresh=0.5):
        super(YOLO, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.trainable = trainable
        self.stride = 32
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.input_size = input_size    # [w, h]
        self.scale = np.array([[[input_size[0], input_size[1], input_size[0], input_size[1]]]])
        self.scale_torch = torch.tensor(self.scale.copy()).float().to(device)

        # 网络定义
        resnet18 = models.resnet18(False)
        self.backbone = torch.nn.Sequential(*(list(resnet18.children())[:-2]))
        # print(self.backbone)

        self.SPP = SPP(512, 512)
        self.SAM = SAM(512)
        self.conv_set = nn.Sequential(
            Conv2d(512, 256, 1, leakyReLU=True),
            Conv2d(256, 512, 3, padding=1, leakyReLU=True),
            Conv2d(512, 256, 1, leakyReLU=True),
            Conv2d(256, 512, 3, padding=1, leakyReLU=True),
        )
        # 进入FC之前，flatten方式会破坏特征的空间结构信息， 所以1×1卷积替代FC
        self.pred = nn.Conv2d(512, 1 + self.num_classes + 4, 1)

    def decode_boxes(self, pred):
        """
        input box :  [tx, ty, w, h]
        output box : [xmin, ymin, xmax, ymax]

        生成grid_cell的shape: [1, 49, 2]
        # grid_cell[0, 0, :] = (0, 0)表示第0行第0列的格子
        """
        # 生成每个格子的坐标位置
        w_grid, h_grid = self.input_size[0] // self.stride, self.input_size[1] // self.stride
        grid_y, grid_x = torch.meshgrid([torch.arange(h_grid), torch.arange(w_grid)])
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
        grid_cell = grid_xy.view(1, h_grid * w_grid, 2).to(self.device)

        # tx, ty, 网络用sigmoid输出， w, h, exp(log())
        pred[:, :, :2] = torch.sigmoid(pred[:, :, :2]) + grid_cell  # 每个格子的小偏移量c_x, c_y加上格子的坐标位置
        pred[:, :, 2:] = torch.exp(pred[:, :, 2:])  # 标签是ln(w)和ln(h),所以预测时需要取指数exp（ln(pred_w or pred_h)）

        # [c_x, c_y, w, h] -> [xmin, ymin, xmax, ymax]
        output = torch.zeros_like(pred)
        output[:, :, 0] = pred[:, :, 0] * self.stride - pred[:, :, 2] / 2
        output[:, :, 1] = pred[:, :, 1] * self.stride - pred[:, :, 3] / 2
        output[:, :, 2] = pred[:, :, 0] * self.stride + pred[:, :, 2] / 2
        output[:, :, 3] = pred[:, :, 1] * self.stride + pred[:, :, 3] / 2

        return output

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

    def postprocess(self, all_bbox, all_class):
        """
        bbox_pred: (HxW, 4), bsize = 1
        prob_pred: (HxW, num_classes), bsize = 1
        """
        bbox_pred = all_bbox
        prob_pred = all_class

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
        x = self.backbone(x)
        # neck
        x = self.SPP(x)
        x = self.SAM(x)
        x = self.conv_set(x)
        # 1×1卷积预测head
        prediction = self.pred(x)

        # 分离预测值
        prediction = prediction.view(x.size(0), 1 + self.num_classes + 4, -1).permute(0, 2, 1)  # [B, HW, C]
        # cx, cy表示预测框的中心点与预测grid左上角点的偏差，根据偏差计算预测框的中心点，wh分别为宽高占比
        conf_pred = prediction[:, :, :1]    # [B, H*W, 1]
        txtywh_pred = prediction[:, :, 1:5]     # shape:[B, H*W, 4],[cx, cy, w, h]
        cls_pred = prediction[:, :, 5:]    # [B, H*W, num_cls]

        if self.trainable:
            """
            # 训练标签：
            # 对于中心点落在[grid_x, grid_y]的位置，则认为有物体，因此Pr(objectness)=1:
            # grid_x = math.floor(center_x/stride)
            # grid_y = math.floor(center_y/stride)
            # gt_cx = (center_x/stride)-grid_x
            # gt_cy = (center_y/stride)-grid_y
            # gt_w = w_box/w_image
            # gt_h = h_box/h_image
            """
            conf_loss, cls_loss, xywh_loss, total_loss = loss(pred_conf=conf_pred, pred_cls=cls_pred,
                                                              pred_txtytwth=txtywh_pred, label=target)
            return conf_loss, cls_loss, xywh_loss, total_loss
        else:
            with torch.no_grad():
                """
                测试：
                # pred:[cx, cy, w, h]-->pred_box:[center_x, center_y, w_box, h_box]
                # center_x = (grid_x + cx) * stride
                # center_y = (grid_y + cy) * stride
                # w_box = w * w_image
                # h_box = h * h_image
                # score = conf_pred * cls_pred,用这个得分去做后续的非极大值抑制处理（NMS）
                """
                # batch size = 1
                all_conf = torch.sigmoid(conf_pred)[0]  # 0 is because that these is only 1 batch
                all_bbox = torch.clamp((self.decode_boxes(txtywh_pred) / self.scale_torch)[0], 0., 1.)
                all_class = (torch.softmax(cls_pred[0, :, :], 1) * all_conf)

                # separate bbox pred and class conf
                all_bbox = all_bbox.to('cpu').numpy()
                all_class = all_class.to('cpu').numpy()

                bboxes, scores, cls_inds = self.postprocess(all_bbox, all_class)
                return bboxes, scores, cls_inds


if __name__ == "__main__" :
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    yolo = YOLO(device=device, num_classes=20, input_size=[224, 224],  trainable=False).to(device)
    input = torch.rand((1, 3, 224, 224)).to(device)
    output = yolo(input)
