# import numpy as np
# import cv2
#
#
# def rotate(img, angle):
#     h, w, _ = img.shape
#
#     x = np.array([(np.arange(h)+1) for i in range(w)]).T
#     y = np.array([(np.arange(w)+1) for j in range(h)])
#
#     r = np.sqrt(np.square(x)+np.square(y))
#     theta = np.arctan(y/x)
#     new_theta = theta + angle / 180 * np.pi
#     new_theta = new_theta
#
#     new_x = np.around(r*np.cos(new_theta))
#     new_y = np.around(r*np.sin(new_theta))
#
#     new_x = new_x + abs(np.min(np.min(new_x))) + 1 if np.min(np.min(new_x)) < 0.else new_x
#     new_y = new_y + abs(np.min(np.min(new_y))) + 1 if np.min(np.min(new_y)) < 0.else new_y
#
#     rotated = np.zeros((int(np.max(np.max(new_x))), int(np.max(np.max(new_y))), 3), np.uint8)
#     for i in range(h):
#         for j in range(w):
#             rotated[int(new_x[i, j]-1), int(new_y[i, j]-1), :] = img[i, j, :]
#     return rotated
#
#
# img = cv2.imread("./lena.jpg")
# img = rotate(img, 3)
# cv2.imshow("rot", img)
# cv2.waitKey(0)
import torch
print(torch.cuda.is_available())


