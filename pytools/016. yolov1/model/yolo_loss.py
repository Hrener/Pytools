import torch
import torch.nn as nn


class BCELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(BCELoss, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, targets):
        pos_id = (targets == 1.0).float()
        neg_id = (targets == 0.0).float()
        pos_loss = -pos_id * torch.log(inputs + 1e-14)
        neg_loss = -neg_id * torch.log(1.0 - inputs + 1e-14)
        if self.reduction == 'mean':
            pos_loss = torch.mean(torch.sum(pos_loss, 1))
            neg_loss = torch.mean(torch.sum(neg_loss, 1))
            return pos_loss, neg_loss
        else:
            return pos_loss, neg_loss


class MSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(MSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, targets):
        pos_id = (targets == 1.0).float()
        neg_id = (targets == 0.0).float()
        pos_loss = pos_id * (inputs - targets) ** 2
        neg_loss = neg_id * (inputs) ** 2
        if self.reduction == 'mean':
            pos_loss = torch.mean(torch.sum(pos_loss, 1))
            neg_loss = torch.mean(torch.sum(neg_loss, 1))
            return pos_loss, neg_loss
        else:
            return pos_loss, neg_loss


class BCE_focal_loss(nn.Module):
    def __init__(self, weight=None, gamma=2, reduction='mean'):
        super(BCE_focal_loss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        pos_id = (targets == 1.0).float()
        neg_id = (1 - pos_id).float()
        pos_loss = -pos_id * (1.0 - inputs) ** self.gamma * torch.log(inputs + 1e-14)
        neg_loss = -neg_id * (inputs) ** self.gamma * torch.log(1.0 - inputs + 1e-14)

        if self.reduction == 'mean':
            return torch.mean(torch.sum(pos_loss + neg_loss, 1))
        else:
            return pos_loss + neg_loss


def loss(pred_conf, pred_cls, pred_txtytwth, label):
    obj = 5.0
    noobj = 1.0

    # create loss_f
    conf_loss_function = MSELoss(reduction='mean')
    cls_loss_function = nn.CrossEntropyLoss(reduction='none')
    txty_loss_function = nn.BCEWithLogitsLoss(reduction='none')
    twth_loss_function = nn.MSELoss(reduction='none')
    # pred
    pred_conf = torch.sigmoid(pred_conf[:, :, 0])
    pred_cls = pred_cls.permute(0, 2, 1)
    pred_txty = pred_txtytwth[:, :, :2]
    pred_twth = pred_txtytwth[:, :, 2:]
    # gt
    gt_obj = label[:, :, 0].float()
    gt_cls = label[:, :, 1].long()
    gt_txtytwth = label[:, :, 2:-1].float()
    gt_box_scale_weight = label[:, :, -1]

    # objectness loss：公式中第三、四项, obj和noobj用于平衡正负样例损失
    pos_loss, neg_loss = conf_loss_function(pred_conf, gt_obj)
    conf_loss = obj * pos_loss + noobj * neg_loss

    # class loss：公式中第五项，只计算有物体的地方
    cls_loss = torch.mean(torch.sum(cls_loss_function(pred_cls, gt_cls) * gt_obj, 1))

    # box loss：公式中第一、二项，只计算有物体的地方
    txty_loss = torch.mean(
        torch.sum(torch.sum(txty_loss_function(pred_txty, gt_txtytwth[:, :, :2]), 2) * gt_box_scale_weight * gt_obj, 1))
    twth_loss = torch.mean(
        torch.sum(torch.sum(twth_loss_function(pred_twth, gt_txtytwth[:, :, 2:]), 2) * gt_box_scale_weight * gt_obj, 1))
    txtytwth_loss = txty_loss + twth_loss

    total_loss = conf_loss + cls_loss + txtytwth_loss

    return conf_loss, cls_loss, txtytwth_loss, total_loss


if __name__ == "__main__":
    pass