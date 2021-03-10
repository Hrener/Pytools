from ensemble_boxes import *
import json
import numpy as np

with open("img_sort.json")as f:
    img_sort = json.load(f)
model_list = ["./syx/detectors_e11_0.73.json",
"./syx/x101_ms5_e9_0.789.json",
"./syx/x101_ms5_result9.json",
"./syx/x101_ms_e9_0.78042.json",
"./syx/x101_ms_e10_0.76248.json",
"./syx/x101_ms_e11_0.7598.json",
"./syx/x101_ms_e12_0.76.json",
              ]
model_res_all = []
for mosel_path in model_list:
    with open(mosel_path)as f:
        model_res = json.load(f)
        model_res_all.append(model_res)
print(len(model_res_all))


weights = [1]*len(model_res_all)
iou_thr = 0.5
skip_box_thr = 0.001
sigma = 0.1

predict = []
for i in range(len(img_sort)):
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
        for class_num, pred in enumerate(model_res_all[model_id][i]):
            for pd in pred:
                pd[0] = pd[0] / w
                pd[2] = pd[2] / w
                pd[1] = pd[1] / h
                pd[3] = pd[3] / h
                boxes_list[model_id].append(pd[:4])
                scores_list[model_id].append(pd[4])
                labels_list[model_id].append(class_num)
    print(labels_list)
    # print(boxes_list, scores_list, labels_list)
    boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights,
                                                  iou_thr=iou_thr, skip_box_thr=skip_box_thr, conf_type='max')
    # print(boxes, scores, labels)
    for box, score, label in zip(boxes, scores, labels):
        # print(box, list(box))
        box = list(box)
        box[0] = box[0] * w
        box[2] = box[2] * w
        box[1] = box[1] * h
        box[3] = box[3] * h
        box.append(score)
        # print(box)
        img_res[int(label)].append(box)
    print(img_res)
    predict.append(img_res)
    # break


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


with open('./merge7_syx.json', 'w', encoding='utf-8') as fw:
    json.dump(predict, fw, ensure_ascii=False, cls=MyEncoder)
