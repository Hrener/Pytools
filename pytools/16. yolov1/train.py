import os
import time
import math
import random
import argparse
import logging

import torch.optim as optim
import torch.backends.cudnn as cudnn

from data.voc0712 import *
from data.augmentations import SSDAugmentation
from model.yolo import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Detection')
    parser.add_argument('-input_size', default=[224, 224], help='input size for training')
    parser.add_argument('--batch_size', default=8, type=int, help='Batch size for training')
    parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--lr_epoch', default=[60, 90, 160], help='initial learning rate')
    parser.add_argument('--cos', action='store_true', default=False, help='use cos lr')
    parser.add_argument('--wp_epoch', type=int, default=2, help='The upper bound of warm-up')
    parser.add_argument('--no_warm_up', action='store_true', default=False, help='yes or no to choose warmup')
    parser.add_argument('--max_epoch', type=int, default=160, help='max epoch for training')
    parser.add_argument('--dataset_root', default="./data/VOCdevkit/", help='Location of VOC root directory')
    parser.add_argument('--num_classes', default=20, type=int, help='The number of dataset classes')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--save_folder', default='weights/', type=str, help='Gamma update for SGD')
    parser.add_argument('--resume', type=str, default=None, help='fine tune the model trained on MSCOCO.')

    return parser.parse_args()


class Logger:
    def __init__(self, title="Logger:", filename="Default.log"):
        # 1.生成记录器
        self.logger = logging.getLogger(__name__)
        # 2.记录器配置
        logging.basicConfig(format="%(asctime)s - %(message)s",
                            level=logging.DEBUG,
                            filename=filename,
                            filemode='w')
        # 3.输出到控制台
        console = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(message)s")
        console.setFormatter(formatter)
        self.logger.addHandler(console)

        # 记录标题
        self.logger.info(title)

    def write_info(self, message):
        # 记录info
        self.logger.info(message)

    def write_warning(self, message):
        # 记录warning
        self.logger.warning(message)


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def train():
    log_filename = "./log/yolo_" + time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time())) + ".log"
    logger = Logger(title="This is YOLO Training:\n", filename=log_filename)

    setup_seed(2020)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = parse_args()
    os.makedirs(args.save_folder, exist_ok=True)

    # build data_loader
    dataset = VOCDetection(root=args.dataset_root,  transform=SSDAugmentation(args.input_size,
                           mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)))
    # dataset = VOCDetection(root=args.dataset_root, transform=BaseTransform(args.input_size, mean=(0.406, 0.456, 0.485),
    #                                                                        std=(0.225, 0.224, 0.229)))
    data_loader = data.DataLoader(dataset, args.batch_size, num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate, pin_memory=True)
    # build model
    yolo_net = YOLO(device=device, input_size=args.input_size, num_classes=args.num_classes, trainable=True)

    # finetune the model trained on COCO
    if args.resume is not None:
        print('finetune COCO trained ')
        yolo_net.load_state_dict(torch.load(args.resume, map_location=device), strict=False)

    print("----------------------------------------Object Detection--------------------------------------------")
    model = yolo_net.to(device)

    base_lr = args.lr
    tmp_lr = base_lr
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=5e-4)

    # start training
    t0 = time.time()
    max_loss = 99999.
    epoch_size = len(dataset) // args.batch_size
    for epoch in range(args.max_epoch):
        epoch_loss = 0.
        # use cos lr
        if args.cos and (epoch > 20) and (epoch <= args.max_epoch - 20):
            # use cos lr
            tmp_lr = 0.00001 + 0.5 * (base_lr - 0.00001) * (
                        1 + math.cos(math.pi * (epoch - 20) * 1. / (args.max_epoch - 20)))
            next(iter(optimizer.param_groups))['lr'] = tmp_lr
        elif args.cos and epoch > args.max_epoch - 20:
            tmp_lr = 0.00001
            next(iter(optimizer.param_groups))['lr'] = tmp_lr
        else:
            # use step lr
            if epoch in args.lr_epoch:
                tmp_lr = tmp_lr * 0.1
                next(iter(optimizer.param_groups))['lr'] = tmp_lr

        for iter_i, (images, targets) in enumerate(data_loader):
            # WarmUp strategy for learning rate
            if not args.no_warm_up:
                if epoch < args.wp_epoch:
                    tmp_lr = base_lr * pow((iter_i + epoch * epoch_size) * 1. / (args.wp_epoch * epoch_size), 4)
                    # tmp_lr = 1e-6 + (base_lr-1e-6) * (iter_i+epoch*epoch_size) / (epoch_size * (args.wp_epoch))

                    next(iter(optimizer.param_groups))['lr'] = tmp_lr

                elif epoch == args.wp_epoch and iter_i == 0:
                    next(iter(optimizer.param_groups))['lr'] = base_lr
            # make train label
            targets = [label.tolist() for label in targets]
            targets = gt_creator(input_size=args.input_size, stride=yolo_net.stride, label_lists=targets)

            # to device
            images = images.to(device)
            targets = torch.tensor(targets).float().to(device)

            # forward and loss
            conf_loss, cls_loss, txtytwth_loss, total_loss = model(images, target=targets)
            epoch_loss += total_loss.item()

            # backprop and update
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if iter_i % 10 == 0:
                t1 = time.time()
                logger.write_info('[Epoch %d/%d][Iter %d/%d][lr %.6f]'
                                  '[Loss: obj %.2f || cls %.2f || bbox %.2f || total %.2f || size %d || time: %.2f]'
                      % (epoch + 1, args.max_epoch, iter_i, epoch_size, tmp_lr, conf_loss.item(), cls_loss.item(),
                         txtytwth_loss.item(), total_loss.item(), args.input_size[0], t1 - t0))

                t0 = time.time()

        if (epoch + 1) % 2 == 0:
            print('Saving state, epoch:', epoch + 1)
            torch.save(model.state_dict(), os.path.join(args.save_folder, 'yolo_' + repr(epoch + 1) + '.pth'))
        # save best model
        epoch_loss = (epoch_loss / epoch_size)
        if max_loss > epoch_loss:
            max_loss = epoch_loss
            torch.save(model.state_dict(), os.path.join(args.save_folder, 'yolo_best_model.pth'))


if __name__ == '__main__':
    train()
