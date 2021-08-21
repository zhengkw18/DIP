import cv2
import math
import numpy as np
import scipy.integrate as integrate
from scipy import optimize

eps = 1e-5
h = 8
k = 15


def x_t(t):
    # return 8 * math.sin(t)
    return 16 * math.sin(t) - 4 * math.sin(3 * t)


def y_t(t):
    # return 8 * math.cos(t)
    return 15 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)


t1 = optimize.brentq(y_t, 0, np.pi)
t2 = optimize.brentq(y_t, np.pi, 2 * np.pi)


def get_t0(x, y):
    if abs(x) < eps:
        return 0 if y > 0 else np.pi
    if abs(y) < eps:
        return t1 if x > 0 else t2
    f = lambda t: x_t(t) * y - y_t(t) * x
    if x > 0 and y > 0:
        return optimize.brentq(f, 0, t1)
    if x < 0 and y > 0:
        return optimize.brentq(f, t2, 2 * np.pi)
    if x < 0 and y < 0:
        return optimize.brentq(f, np.pi, t2)
    if x > 0 and y < 0:
        return optimize.brentq(f, t1, np.pi)


def get_color(src, r_out, c_out, d0):
    def _in_range(_x, _y):
        return _x >= 0 and _x < src.shape[1] and _y >= 0 and _y < src.shape[0]

    y_out, x_out = -r_out, c_out
    t0 = get_t0(x_out, y_out)
    x_t0, y_t0 = x_t(t0), y_t(t0)
    sin_theta0 = x_out / (x_t0 * k) if abs(x_t0) > eps else y_out / (y_t0 * k)
    assert sin_theta0 >= 0
    if sin_theta0 > 1:
        return np.array([0, 0, 0], dtype=np.uint8)
    theta0 = math.asin(sin_theta0)
    integrated = lambda phi: math.sqrt((x_t0 ** 2 + y_t0 ** 2) * math.cos(phi) ** 2 + h ** 2 * math.sin(phi) ** 2)
    d = d0 * integrate.quad(integrated, 0, theta0)[0] / integrate.quad(integrated, 0, np.pi / 2)[0]
    x, y = d * math.sin(t0) + src.shape[1] / 2.0, -d * math.cos(t0) + src.shape[0] / 2.0
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
    src = cv2.imread("sphere.jpg")
    R_out, C_out = 500, 600
    res = np.zeros((R_out, C_out, 3), np.uint8)
    R_in, C_in, _ = src.shape
    d0 = 0.5 * max(R_in, C_in)
    for j in range(R_out):
        for i in range(C_out):
            res[j, i] = get_color(src, j - R_out / 2.0, i - C_out / 2.0, d0)
    cv2.imwrite("general_out.png", res)
