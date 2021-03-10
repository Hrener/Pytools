#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Hren

matrix = [[1, 2, 3, 4, 5, 6],
          [6, 7, 8, 9, 10, 11],
          [11, 12, 13, 14, 15, 16]]

xmin = 0
xmax = len(matrix)
ymin = 0
ymax = len(matrix[0])
res = []
len_ = xmax * ymax
while True:
    [res.append(matrix[xmin][col]) for col in range(ymin, ymax)]
    xmin += 1
    if len(res) == len_:
        break
    [res.append(matrix[row][ymax-1]) for row in range(xmin, xmax)]
    ymax -= 1
    if len(res) == len_:
        break
    [res.append(matrix[xmax-1][col]) for col in range(ymax-1, ymin-1, -1)]
    xmax -= 1
    if len(res) == len_:
        break
    [res.append(matrix[row][ymin]) for row in range(xmax-1, xmin-1, -1)]
    ymin += 1
    if len(res) == len_:
        break

print(res)
