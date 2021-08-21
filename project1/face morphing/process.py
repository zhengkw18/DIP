import argparse
import json
import cv2
import numpy as np
from tqdm import tqdm


class AffineTrans:
    def __init__(self, s0, s1, s2, d0, d1, d2):
        A = np.array(
            [
                [s0[0], s0[1], 1, 0, 0, 0],
                [0, 0, 0, s0[0], s0[1], 1],
                [s1[0], s1[1], 1, 0, 0, 0],
                [0, 0, 0, s1[0], s1[1], 1],
                [s2[0], s2[1], 1, 0, 0, 0],
                [0, 0, 0, s2[0], s2[1], 1],
            ]
        )
        b = np.array([d0[0], d0[1], d1[0], d1[1], d2[0], d2[1]])
        T = np.linalg.solve(A, b)
        self.T = np.vstack([T.reshape(2, 3), [0, 0, 1]])

    def forward(self, s):
        return list(self.T.dot(np.array([s[0], s[1], 1]).T)[:2])


def extend_lst(lst, H, W):
    lst.extend([[0, 0], [0, W // 2], [0, W - 1], [H // 2, 0], [H // 2, W - 1], [H - 1, 0], [H - 1, W // 2], [H - 1, W - 1]])


def in_triangle(p, p0, p1, p2):
    pa, pb, pc = p - p0, p - p1, p - p2
    c1, c2, c3 = np.cross(pa, pb), np.cross(pb, pc), np.cross(pc, pa)
    return c1 * c2 >= 0 and c2 * c3 >= 0


def get_color(src, trans, h, w):
    def _clip1(hh):
        return 0 if hh < 0 else src.shape[0] - 1 if hh >= src.shape[0] else hh

    def _clip2(ww):
        return 0 if ww < 0 else src.shape[1] - 1 if ww >= src.shape[1] else ww

    y, x = trans.forward([h, w])
    x1, x2, y1, y2 = _clip2(int(np.floor(x))), _clip2(int(np.ceil(x))), _clip1(int(np.floor(y))), _clip1(int(np.ceil(y)))
    if x1 != x2:
        y_1 = ((x2 - x) * src[y1, x1] + (x - x1) * src[y1, x2]) / (x2 - x1)
        y_2 = ((x2 - x) * src[y2, x1] + (x - x1) * src[y2, x2]) / (x2 - x1)
    else:
        y_1 = src[y1, x1]
        y_2 = src[y2, x1]
    if y1 != y2:
        a = ((y2 - y) * y_1 + (y - y1) * y_2) / (y2 - y1)
    else:
        a = y_1
    return np.round(a).astype(np.uint8)


def fill_triangle(src_img, dst_img, s0, s1, s2, d0, d1, d2, out, ratio):
    def _get_intermediate(a1, a2):
        return (1 - ratio) * a1 + ratio * a2

    s0, s1, s2, d0, d1, d2 = np.array(s0), np.array(s1), np.array(s2), np.array(d0), np.array(d1), np.array(d2)
    m0, m1, m2 = _get_intermediate(s0, d0), _get_intermediate(s1, d1), _get_intermediate(s2, d2)
    T1, T2 = AffineTrans(m0, m1, m2, s0, s1, s2), AffineTrans(m0, m1, m2, d0, d1, d2)
    hmin, hmax, wmin, wmax = round(min(m0[0], m1[0], m2[0])), round(max(m0[0], m1[0], m2[0])), round(min(m0[1], m1[1], m2[1])), round(max(m0[1], m1[1], m2[1]))
    for h in range(max(0, hmin), min(out.shape[0] - 1, hmax) + 1):
        for w in range(max(0, wmin), min(out.shape[1] - 1, wmax) + 1):
            p = np.array([h, w])
            if in_triangle(p, m0, m1, m2):
                c1, c2 = get_color(src_img, T1, h, w), get_color(dst_img, T2, h, w)
                out[h, w, :] = _get_intermediate(c1, c2).clip(0, 255).astype("uint8")


def morphing(src_img, dst_img, src_points, src_tris, dst_points, ratio):
    H1, W1, _ = src_img.shape
    H2, W2, _ = dst_img.shape
    extend_lst(src_points, H1, W1)
    extend_lst(dst_points, H2, W2)
    out = np.zeros((round((1 - ratio) * H1 + ratio * H2), round((1 - ratio) * W1 + ratio * W2), 3))
    for i in tqdm(range(len(src_tris))):
        tri = src_tris[i]
        s0, s1, s2 = src_points[tri[0]], src_points[tri[1]], src_points[tri[2]]
        d0, d1, d2 = dst_points[tri[0]], dst_points[tri[1]], dst_points[tri[2]]
        fill_triangle(src_img, dst_img, s0, s1, s2, d0, d1, d2, out, ratio)
    return out


def pad_img(src, H, W):
    assert H >= src.shape[0] and W >= src.shape[1]
    return np.pad(src, ((0, H - src.shape[0]), (0, W - src.shape[1]), (0, 0)), "constant")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate morphing sequence using preprocessed data")
    parser.add_argument("--src", type=str, help="Source image", required=True)
    parser.add_argument("--target", type=str, help="Target image", required=True)
    parser.add_argument("--out", type=str, help="Output file", required=True)
    parser.add_argument("--ratio", type=float, help="Morphing ratio, -1 for video(.avi)", required=True)

    args = parser.parse_args()
    src_img = cv2.imread(args.src)
    target_img = cv2.imread(args.target)
    with open(args.src + ".json", "r") as f:
        src_points = json.load(f)
    with open(args.target + ".json", "r") as f:
        dst_points = json.load(f)
    with open(args.src + ".tris.json", "r") as f:
        src_tris = json.load(f)
    assert len(src_points) == len(dst_points)
    if args.ratio >= 0:
        cv2.imwrite(args.out, morphing(src_img, target_img, src_points, src_tris, dst_points, args.ratio))
    else:
        H = max(src_img.shape[0], target_img.shape[0])
        W = max(src_img.shape[1], target_img.shape[1])
        fps = 24
        seconds = 5
        videowriter = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc("X", "V", "I", "D"), fps, (W, H))
        for ratio in np.arange(0, 1, 1 / (fps * seconds)):
            print("Ratio:", ratio)
            videowriter.write(pad_img(morphing(src_img, target_img, src_points, src_tris, dst_points, ratio), H, W).astype("uint8"))
