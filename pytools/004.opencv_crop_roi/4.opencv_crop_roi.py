#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Hren
"""
使用Python和OpenCV检测图像中的物体并将物体裁剪下来
"""

import cv2
import math
import numpy as np


def Nrotate(angle, valuex, valuey, pointx, pointy):
    angle = (angle / 180) * math.pi
    valuex = np.array(valuex)
    valuey = np.array(valuey)
    nRotatex = (valuex - pointx) * math.cos(angle) - (valuey - pointy) * math.sin(angle) + pointx
    nRotatey = (valuex - pointx) * math.sin(angle) + (valuey - pointy) * math.cos(angle) + pointy
    return (nRotatex, nRotatey)


# 顺时针旋转
def Srotate(angle, valuex, valuey, pointx, pointy):
    angle = (angle / 180) * math.pi
    valuex = np.array(valuex)
    valuey = np.array(valuey)
    sRotatex = (valuex - pointx) * math.cos(angle) + (valuey - pointy) * math.sin(angle) + pointx
    sRotatey = (valuey - pointy) * math.cos(angle) - (valuex - pointx) * math.sin(angle) + pointy
    return (sRotatex, sRotatey)


# 将四个点做映射
def rotatecordiate(angle, rectboxs, pointx, pointy):
    output = []
    for rectbox in rectboxs:
        if angle > 0:
            output.append(Srotate(angle, rectbox[0], rectbox[1], pointx, pointy))
        else:
            output.append(Nrotate(-angle, rectbox[0], rectbox[1], pointx, pointy))
    return output


def detection_crop(path, crop_mode=0):
    # step1：加载图片，转成灰度图
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # step2:用Sobel算子计算x，y方向上的梯度，之后在x方向上减去y方向上的梯度，通过这个减法，我们留下具有高水平梯度和低垂直梯度的图像区域。
    gradX = cv2.Sobel(gray, cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, cv2.CV_32F, dx=0, dy=1, ksize=-1)
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    # step3:滤波，二值化
    blurred = cv2.blur(gradient, (9, 9))
    _, thresh = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)

    # step4:做一些形态学方面的操作。
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # step5:执行4次形态学腐蚀与膨胀。
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)

    # step5:找轮廓
    (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

    if crop_mode == 0:
        # step6:直接裁剪最大外接正矩形
        rect = cv2.boundingRect(c)  # [x1, y1, w, h]
        # cv2.rectangle(image, rect, (0, 0, 255), 2)    # 绘制最大外接正矩形
        crop_img = image[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
        return crop_img
    else:
        # step6:裁剪最小外接矩形
        rect = cv2.minAreaRect(c)  # [(center_x, center_y), (w, h), angle]
        box = cv2.boxPoints(rect)   # [point1(x, y), point2, point3，point4]
        # draw_img = cv2.drawContours(image.copy(), [box], -1, (0, 0, 255), 2)  # 绘制最小外接矩形

        # 旋转原图，角度为最小外接矩形的角度，防止旋转后目标区域在图像外面，将图像大小扩大为原来的2倍。
        M = cv2.getRotationMatrix2D(rect[0], rect[2], 1)
        rotated_image = cv2.warpAffine(image, M, (2 * image.shape[0], 2 * image.shape[1]))

        # 得到旋转后的四个点
        rotated_box = rotatecordiate(rect[2], box, rect[0][0], rect[0][1])
        xs = [int(x[1]) for x in rotated_box]
        ys = [int(x[0]) for x in rotated_box]
        crop_img = rotated_image[min(xs):max(xs), min(ys):max(ys)]
        return crop_img


if __name__ == "__main__":
    path = 'pet.png'
    crop_img = detection_crop(path, crop_mode=1)
    cv2.imwrite('crop_' + path.split('\\')[-1], crop_img)

