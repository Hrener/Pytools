#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Hren
import pandas as pd
import numpy as np
from shutil import copyfile

data_9654 = pd.read_csv('./res/b5_336_e14_rotated2_36000_merge7.csv')
data_9652 = pd.read_csv('./res/b5_336_rotated2_36000_merge2_0.965.csv')
data_962 = pd.read_csv('./res/b5_224_rotated2_24000_merge8_0.962.csv')
data_955 = pd.read_csv('./res/T_eff_b5_smallsize_result_0.955.csv')
data_952 = pd.read_csv('./res/T_eff_b5_smallsize_aug_final_a9_0.9521.csv')
data_951 = pd.read_csv('./res/T_eff_b5_smallsize_aug_0.95132.csv')
data_95 = pd.read_csv('./res/b5andb4_224_rotated2_24000_merge4_0.95.csv')
different = []
diff1 = []
res = []
for data in zip(data_9654['uuid'], data_9654['label'], data_9652['label'], data_962['label'], data_955['label'], data_952['label'], data_951['label'], data_95['label']):
    if not data[1] == data[2]:
        different.append([data])
        print(data)
        res.append([data[0], max(set(data[1:]), key=data[1:].count)])
    else:
        res.append([data[0], data[1]])
print(len(different))
# test_path = "D:\Hren Files\My Documents\pycharm projects\Github download\Bisai\脑PET/test\AD&CN/"
# for img in different:
#     print(img)
#     img = list(img[0])
#     copyfile(test_path+str(img[0])+".png", "./test_label/"+img[1]+"_"+str(img[0])+"_test.png")


# (297, 'AD', 'CN', 'AD', 'AD')
# (699, 'AD', 'CN', 'AD', 'AD')
# (833, 'CN', 'AD', 'CN', 'CN')
# (875, 'AD', 'CN', 'AD', 'AD')
# (993, 'CN', 'AD', 'CN', 'CN')