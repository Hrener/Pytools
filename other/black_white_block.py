#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Hren
"""
黑白块
"""
import cv2
import numpy as np

# max_pixel = 90
# mask = np.zeros((max_pixel, max_pixel, 3))
# flag = 0
# time_n = 0
# for i in range(max_pixel//10):
#     for j in range(max_pixel//10):
#         mask[i*10:(i+1)*10, j*10:(j+1)*10, :] = flag
#
#         if time_n % 2 == 0:
#             flag = not flag
#             time_n = 1
#         time_n += 1
#
# cv2.imshow('img', mask)
# cv2.waitKey()

shang = 10
max_pixel = shang*5   # 奇数
mask = np.zeros((max_pixel, max_pixel, 3))
flag = 1
time_n = 0

for i in range(max_pixel//shang):
    for j in range(max_pixel//shang):
        mask[i*shang:(i+1)*shang, j*shang:(j+1)*shang, :] = flag

        if time_n % 2 == 0:
            flag = not flag
            time_n = 1
        time_n += 1

for i in range(max_pixel//shang):
    for j in range(max_pixel//shang):
        offset = 4
        mask[i * shang+offset:(i + 1) * shang-offset, j * shang+offset:(j + 1) * shang-offset, :] = 1

num_1 = 0
for i in range(max_pixel):
    for j in range(max_pixel):
        if mask[i, j, 0] == 0:
            num_1 += 1
print(num_1)
# img = mask[0:50, 0:60, :]
cv2.imshow('img', mask)
cv2.waitKey()