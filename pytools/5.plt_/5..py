#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Hren
import numpy as np
import matplotlib.pyplot as plt

# 生成数据
x = np.linspace(0, 4*np.pi)
y = np.sin(x)
# 设置rc参数显示中文标题,设置字体为SimHei显示中文
plt.rcParams['font.sans-serif'] = 'SimHei'
# 设置正常显示字符
plt.rcParams['axes.unicode_minus'] = False

plt.title('sin曲线')
# 设置线条样式
plt.rcParams['lines.linestyle'] = '-.'
# 设置线条宽度
plt.rcParams['lines.linewidth'] = 3
# 绘制sin曲线
plt.plot(x, y, label='$sin(x)$')
# plt.savefig('sin.png')
plt.show()
