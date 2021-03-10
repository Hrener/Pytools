from ensemble_boxes import *
import json
import numpy as np

import cv2

img = cv2.imread('sw_14_000327.jpg')



with open("img_sort.json")as f:
    img_sort = json.load(f)
model_list = [
            "./sub/merge6_0.806.json",
              "./sub/cx101_big_ms_6_ms2_0.777.json",
              "./sub/cx101_large_ms_12_ms4_0.787.json",
              "./sub/merge_large0.787_bigms4001_0.800.json"
              ]
model_res_all = []
for mosel_path in model_list:
    with open(mosel_path)as f:
        model_res = json.load(f)
        model_res_all.append(model_res)
print(len(model_res_all))


weights = [1]*len(model_res_all)
iou_thr = 0.6
skip_box_thr = 0.001
sigma = 0.1
color_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 128, 128)]
predict = []
for i in range(len(img_sort)):
    if img_sort[i]['file_name'] == 'sw_14_000327.jpg':
        print(i)
        boxes_list = [[]]*len(model_res_all)
        scores_list = [[]]*len(model_res_all)
        labels_list = [[]]*len(model_res_all)
        print(boxes_list)
        img_res = []
        for class_id in range(14):
            img_res.append([])

        h, w = img_sort[i]['height'], img_sort[i]['width']
        for model_id in range(len(model_res_all)):
            color = color_list[model_id]
            print(color)
            # print(model_res_all[model_id][i])
            for boxe in model_res_all[model_id][i]:
                # print(boxe)

                if boxe:
                    for box in boxe:
                        # print(box)
                        if (box[4] > 0.3):
                            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 1)
        break
cv2.imshow("img", img)
cv2.waitKey(0)

