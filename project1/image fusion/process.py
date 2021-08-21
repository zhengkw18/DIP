import numpy as np
import cv2
import argparse
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

THRESHOLD = 200


def process_inner_border(src_img, mask):
    H, W, _ = src_img.shape
    inner_lst, border_lst = [], []
    flag_arr, inner_arr, border_arr = np.zeros((H, W)), np.zeros((H, W)), np.zeros((H, W))
    inner_cnt, border_cnt = 0, 0
    out = np.zeros_like(src_img)
    for h in range(1, H - 1):
        for w in range(1, W - 1):
            if mask[h, w] <= THRESHOLD:
                if mask[h - 1, w] > THRESHOLD or mask[h + 1, w] > THRESHOLD or mask[h, w - 1] > THRESHOLD or mask[h, w + 1] > THRESHOLD:
                    flag_arr[h, w] = 1
                    border_lst.append((h, w))
                    border_arr[h, w] = border_cnt
                    border_cnt += 1
                    out[h, w, :] = [255, 255, 0]
            else:
                flag_arr[h, w] = 2
                inner_lst.append((h, w))
                inner_arr[h, w] = inner_cnt
                inner_cnt += 1
                out[h, w, :] = [255, 0, 255]
    cv2.imwrite("processed_mask.png", out)
    return inner_lst, border_lst, flag_arr, inner_arr, border_arr, border_cnt, inner_cnt


def fusion(src_img, target_img, dh, dw, inner_lst, flag_arr, inner_arr, inner_cnt, alpha):
    def _neighbors(h, w):
        return (h - 1, w), (h + 1, w), (h, w - 1), (h, w + 1)

    def _confine(x):
        return 0 if x < 0 else 255 if x > 255 else x

    out = target_img.copy()
    raw = target_img.copy()

    row = []
    col = []
    data = []
    b = []

    for i, (h, w) in enumerate(inner_lst):
        row.append(i)
        col.append(i)
        data.append(4)
        b1 = 0.0
        for h1, w1 in _neighbors(h, w):
            src_grad = int(src_img[h, w]) - int(src_img[h1, w1])
            target_grad = int(target_img[dh + h, dw + w]) - int(target_img[dh + h1, dw + w1])
            if alpha < 0:
                b1 += src_grad if abs(src_grad) > abs(target_grad) else target_grad
            else:
                b1 += alpha * src_grad + (1 - alpha) * target_grad
            flag = flag_arr[h1, w1]
            if flag == 1:
                b1 += target_img[dh + h1, dw + w1]
            elif flag == 2:
                row.append(i)
                col.append(inner_arr[h1, w1])
                data.append(-1)
        b.append([b1])
    A = csc_matrix((data, (row, col)), shape=(inner_cnt, inner_cnt), dtype=float)
    b = csc_matrix(b, dtype=float)
    x = spsolve(A, b)
    for i, (h, w) in enumerate(inner_lst):
        out[dh + h, dw + w] = _confine(x[i])
        raw[dh + h, dw + w] = src_img[h, w]
    return out, raw


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Image fusion")
    parser.add_argument("--src", help="Source image", required=True)
    parser.add_argument("--mask", help="Mask of source image", required=True)
    parser.add_argument("--target", help="Target image", required=True)
    parser.add_argument("--out", help="Output file", required=True)
    parser.add_argument("--dh", help="Offset of height", type=int, default=0)
    parser.add_argument("--dw", help="Offset of width", type=int, default=0)
    parser.add_argument("--alpha", help="Ratio of source gradient, -1 for adaptive", type=float, default=1.0)

    args = parser.parse_args()
    mask = cv2.cvtColor(cv2.imread(args.mask), cv2.COLOR_BGR2GRAY)
    src = cv2.imread(args.src)
    target = cv2.imread(args.target)
    out = np.zeros_like(target)
    raw = np.zeros_like(target)
    inner_lst, border_lst, flag_arr, inner_arr, border_arr, border_cnt, inner_cnt = process_inner_border(src, mask)
    print("{} points on border, {} points inner".format(border_cnt, inner_cnt))
    for i in range(3):
        print("Processing channel {}".format(i))
        out[:, :, i], raw[:, :, i] = fusion(src[:, :, i], target[:, :, i], args.dh, args.dw, inner_lst, flag_arr, inner_arr, inner_cnt, args.alpha)
    cv2.imwrite(args.out, out)
    cv2.imwrite("raw.jpg", raw)
