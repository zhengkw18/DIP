import cv2
import math
import numpy as np


class Trans:
    def __init__(self, src_shape, coefficient):
        self.shape = src_shape
        self.coefficient = coefficient

    def __call__(self, r, theta):
        d = self.coefficient * np.exp(r ** (1 / 2.1) / 1.8)
        r = d * math.sin(theta)
        c = d * math.cos(theta)
        return (c + self.shape[1] / 2.0, r + self.shape[0] / 2.0)


def get_color(src, trans, x, y):
    def _in_range(_x, _y):
        return _x >= 0 and _x < src.shape[1] and _y >= 0 and _y < src.shape[0]

    res = trans(x, y)
    x = res[0]
    y = res[1]
    # x = min(max(x, 0), src.shape[1] - 1)
    # y = min(max(y, 0), src.shape[0] - 1)
    x1 = int(np.floor(x))
    x2 = int(np.ceil(x))
    y1 = int(np.floor(y))
    y2 = int(np.ceil(y))
    q11 = src[y1, x1] if _in_range(x1, y1) else np.array([0, 0, 0], dtype=np.uint8)
    q21 = src[y1, x2] if _in_range(x2, y1) else np.array([0, 0, 0], dtype=np.uint8)
    q12 = src[y2, x1] if _in_range(x1, y2) else np.array([0, 0, 0], dtype=np.uint8)
    q22 = src[y2, x2] if _in_range(x2, y2) else np.array([0, 0, 0], dtype=np.uint8)

    if x1 != x2:
        y_1 = ((x2 - x) * q11 + (x - x1) * q21) / (x2 - x1)
        y_2 = ((x2 - x) * q12 + (x - x1) * q22) / (x2 - x1)
    else:
        y_1 = src[y1, x1] if _in_range(x1, y1) else np.array([0, 0, 0], dtype=np.uint8)
        y_2 = src[y2, x1] if _in_range(x1, y2) else np.array([0, 0, 0], dtype=np.uint8)
    if y1 != y2:
        a = ((y2 - y) * y_1 + (y - y1) * y_2) / (y2 - y1)
    else:
        a = y_1
    return np.round(a).astype(np.uint8)


if __name__ == "__main__":
    width, height = 500, 500
    src = cv2.imread("sphere.jpg")
    height = 400
    width = 430
    res = np.zeros((height, width, 3), np.uint8)
    trans = Trans(src.shape, 0.25)
    r_0 = 1.0 / 2 * min(height, width)
    for j in range(height):
        for i in range(width):
            r = math.sqrt((i - width / 2.0) ** 2 + (j - height / 2.0) ** 2)
            theta = math.atan2(j - height / 2.0, i - width / 2.0)
            res[j, i] = get_color(src, trans, r, theta)
    cv2.imwrite("sphere_fisheye.png", res)