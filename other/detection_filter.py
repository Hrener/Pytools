#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Hren
"""
目标检测去重，保留同一框中最大置信度的目标
"""


def filter(tem_pred):
    """
    Input: tem_pred = [{"img_id":000001,"bbox":[x1, y1, x2, y2],"label":9,"score":0.9888}，{}，{},...]
    Output: 过滤后的检测结果final_pred
    """
    final_pred = []
    removed = []
    for tem in tem_pred:
        tf_list = []
        for rem in removed:
            if (tem['bbox'] == rem['bbox']).all():
                tf_list.append(True)
            else:
                tf_list.append(False)
        if True not in tf_list:
            tem_list = []
            tem_list.append(tem)
            removed.append(tem)
            for pred in tem_pred:
                dis = 10.0
                if abs(tem['bbox'][1] - pred['bbox'][1]) < dis:
                    tem_list.append(pred)
                    removed.append(pred)
            tem_list = sorted(tem_list, key=lambda dict_data: dict_data['score'], reverse=True)
            final_pred.append(tem_list[0])
    return final_pred
