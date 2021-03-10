import cv2


def rotate(image, angle, center=None, scale=1.0):  # 1
    (h, w) = image.shape[:2]  # 2
    if center is None:  # 3
        center = (w // 2, h // 2)  # 4
    M = cv2.getRotationMatrix2D(center, angle, scale)  # 5
    rotated = cv2.warpAffine(image, M, (w, h), borderValue=(255, 255, 255))  # 6
    return rotated  # 7


img = cv2.imread("test0.png")
img = rotate(img, 45)
cv2.imwrite("test_rot10.png", img)