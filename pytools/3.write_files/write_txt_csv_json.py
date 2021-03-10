#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Hren
"""
csv、txt、json等文件写入方式
"""
import csv
import json
import pandas as pd
import numpy as np


# txt写入
with open('test1.txt', 'w', encoding='utf-8')as txt_file:
    txt_file.write('test!\ntest!\n测试！')

# csv写入方式一
with open('test1.csv', 'w', encoding='utf-8', newline='')as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['name', 'age', '性别'])
    results = [['Hren', '999', '男'], ['hren', '999', '女']]
    writer.writerows(results)

# csv写入方式二
results_pd = pd.DataFrame()
results_pd['name'] = ['Hren', 'hren']
results_pd['age'] = [999, 999]
results_pd['性别'] = ['男', '女']
results_pd.to_csv('test2.csv', index=None)


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


# json写入
json_results = {'name': 'hren', 'age': 999}
with open('test.json', 'w', encoding='utf-8')as json_file:
    json.dump(json_results, json_file, cls=MyEncoder)


