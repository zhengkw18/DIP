import argparse
import json
import cv2
import numpy as np


def rotation_matrix_xy(x, y, theta):
    c = np.cos(theta)
    s = np.sin(theta)
    t = 1 - np.cos(theta)
    return np.array(
        [
            [t * x * x + c, t * x * y, s * y],
            [t * x * y, t * y * y + c, -s * x],
            [-s * y, s * x, c],
        ]
    )


def rotation_matrix_z(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array(
        [
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1],
        ]
    )


def extend_lst(lst, H, W):
    lst.extend([[0, 0], [0, W // 2], [0, W - 1], [H // 2, 0], [H // 2, W - 1], [H - 1, 0], [H - 1, W // 2], [H - 1, W - 1]])


def get_epipoles(F):
    value0, vector0 = np.linalg.eig(F)
    value1, vector1 = np.linalg.eig(F.T)
    e0, e1 = vector0[:, np.argmin(value0)], vector1[:, np.argmin(value1)]
    return e0, e1


def get_projective(F):
    e0, e1 = get_epipoles(F)
    d0 = np.array([-e0[1], e0[0], 0])
    Fd0 = F.dot(d0)
    d1 = np.array([-Fd0[1], Fd0[0], 0])
    theta0, theta1 = np.arctan(e0[2] / (d0[1] * e0[0] - d0[0] * e0[1])), np.arctan(e1[2] / (d1[1] * e1[0] - d1[0] * e1[1]))
    R_d0_theta0, R_d1_theta1 = rotation_matrix_xy(d0[0], d0[1], theta0), rotation_matrix_xy(d1[0], d1[1], theta1)
    new_e0, new_e1 = R_d0_theta0.dot(e0), R_d1_theta1.dot(e1)
    phi0, phi1 = -np.arctan(new_e0[1] / new_e0[0]), -np.arctan(new_e1[1] / new_e1[0])
    R_phi0, R_phi1 = rotation_matrix_z(phi0), rotation_matrix_z(phi1)
    H0, H1 = R_phi0.dot(R_d0_theta0), R_phi1.dot(R_d1_theta1)
    return H0, H1


def transform_point(trans, point, h_offset=0, w_offset=0):
    new_p = trans.dot(np.array([point[1], point[0], 1]))
    return [new_p[1] / new_p[2] + h_offset, new_p[0] / new_p[2] + w_offset]


def transform_points(trans, points, h_offset=0, w_offset=0):
    new_ps = []
    for point in points:
        h, w = transform_point(trans, point, h_offset, w_offset)
        new_ps.append([h, w])
    return new_ps


def get_color(src, trans, h, w):
    def _in_range(_x, _y):
        return _x >= 0 and _x < src.shape[1] and _y >= 0 and _y < src.shape[0]

    def _color(_x, _y):
        return src[_y, _x] if _in_range(_x, _y) else np.array([0, 0, 0], dtype=np.uint8)

    y, x = transform_point(trans, [h, w])
    if not _in_range(x, y):
        return np.array([0, 0, 0], dtype=np.uint8)
    x1, x2, y1, y2 = int(np.floor(x)), int(np.ceil(x)), int(np.floor(y)), int(np.ceil(y))
    if x1 != x2:
        y_1 = ((x2 - x) * _color(x1, y1) + (x - x1) * _color(x2, y1)) / (x2 - x1)
        y_2 = ((x2 - x) * _color(x1, y2) + (x - x1) * _color(x2, y2)) / (x2 - x1)
    else:
        y_1 = _color(x1, y1)
        y_2 = _color(x1, y2)
    if y1 != y2:
        a = ((y2 - y) * y_1 + (y - y1) * y_2) / (y2 - y1)
    else:
        a = y_1
    return np.round(a).astype(np.uint8)


def transform_img_with_offset(trans, img, points):
    H, W, _ = img.shape
    trans_inv = np.linalg.inv(trans)
    hmin, hmax, wmin, wmax = float("inf"), float("-inf"), float("inf"), float("-inf")
    for p in [[0, 0], [H - 1, 0], [0, W - 1], [H - 1, W - 1]]:
        new_p = transform_point(trans, p)
        hmin, hmax, wmin, wmax = min(hmin, new_p[0]), max(hmax, new_p[0]), min(wmin, new_p[1]), max(wmax, new_p[1])
    hmin, hmax, wmin, wmax = int(np.floor(hmin)), int(np.ceil(hmax)), int(np.floor(wmin)), int(np.ceil(wmax))
    print(hmin, hmax, wmin, wmax)
    out = np.zeros([hmax - hmin + 1, wmax - wmin + 1, 3])
    for h in range(hmin, hmax + 1):
        for w in range(wmin, wmax + 1):
            out[h - hmin, w - wmin, :] = get_color(img, trans_inv, h, w)

    new_points = transform_points(trans, points, -hmin, -wmin)
    return out, new_points


def pre_warp(src_img, dst_img, src_points, dst_points):
    src_points_r, dst_points_r = [], []
    for p in src_points:
        src_points_r.append([p[1], p[0]])
    for p in dst_points:
        dst_points_r.append([p[1], p[0]])
    # for source1 and target1, use this line
    F = cv2.findFundamentalMat(np.array(src_points_r), np.array(dst_points_r), method=cv2.FM_8POINT)[0]
    # for source2 and target2, use this line
    # F = cv2.findFundamentalMat(np.array(src_points_r[25:]), np.array(dst_points_r[25:]), method=cv2.FM_8POINT)[0]
    H1, W1, _ = src_img.shape
    H2, W2, _ = dst_img.shape
    extend_lst(src_points, H1, W1)
    extend_lst(dst_points, H2, W2)
    H0, H1 = get_projective(F)
    new_src_img, new_src_points = transform_img_with_offset(H0, src_img, src_points)
    new_dst_img, new_dst_points = transform_img_with_offset(H1, dst_img, dst_points)
    return new_src_img, new_dst_img, new_src_points, new_dst_points


if __name__ == "__main__":
    parser = argparse.ArgumentParser("View morphing pre-warp")
    parser.add_argument("--src", type=str, help="Source image", required=True)
    parser.add_argument("--target", type=str, help="Target image", required=True)
    args = parser.parse_args()
    src_img = cv2.imread(args.src)
    target_img = cv2.imread(args.target)
    with open(args.src + ".json", "r") as f:
        src_points = json.load(f)
    with open(args.target + ".json", "r") as f:
        dst_points = json.load(f)
    new_src_img, new_dst_img, new_src_points, new_dst_points = pre_warp(src_img, target_img, src_points, dst_points)
    print(new_src_img.shape, new_dst_img.shape)
    cv2.imwrite(args.src + ".prewarp.png", new_src_img)
    cv2.imwrite(args.target + ".prewarp.png", new_dst_img)
    with open(args.src + ".prewarp.png.json", "w") as f:
        json.dump(new_src_points, f)
    with open(args.target + ".prewarp.png.json", "w") as f:
        json.dump(new_dst_points, f)
    with open(args.src + ".tris.json", "r") as f:
        src_tris = json.load(f)
    with open(args.src + ".prewarp.png.tris.json", "w") as f:
        json.dump(src_tris, f)
