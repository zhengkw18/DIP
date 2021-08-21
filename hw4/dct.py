import numpy as np
import cv2

ratios = [1, 2, 4, 8]
blk_szs = [8, 32, 128, 512]

Q_jpeg = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 36, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ]
)

Q_canon = np.array(
    [
        [1, 1, 1, 2, 3, 6, 8, 10],
        [1, 1, 2, 3, 4, 8, 9, 8],
        [2, 2, 2, 3, 6, 8, 10, 8],
        [2, 2, 3, 4, 7, 12, 11, 9],
        [3, 3, 8, 11, 10, 16, 15, 11],
        [3, 5, 8, 10, 12, 15, 16, 13],
        [7, 10, 11, 12, 15, 17, 17, 14],
        [14, 13, 13, 15, 15, 14, 14, 14],
    ]
)

Q_nikon = np.array(
    [
        [2, 1, 1, 2, 3, 5, 6, 7],
        [1, 1, 2, 2, 3, 7, 7, 7],
        [2, 2, 2, 3, 5, 7, 8, 7],
        [2, 2, 3, 3, 6, 10, 10, 7],
        [2, 3, 4, 7, 8, 13, 12, 9],
        [3, 4, 7, 8, 10, 12, 14, 11],
        [6, 8, 9, 10, 12, 15, 14, 12],
        [9, 11, 11, 12, 13, 12, 12, 12],
    ]
)


def psnr(src, out):
    mse = ((out - src) ** 2).mean()
    return 10 * np.log10(255 * 255 / mse)


def dct(src, ratio, blk_sz):
    H, W = src.shape
    n_blk_h, n_blk_w = H // blk_sz, W // blk_sz
    out = np.zeros_like(src)
    A = np.zeros((blk_sz, blk_sz))
    A[0, :] = 1 * np.sqrt(1 / blk_sz)

    for i in range(1, blk_sz):
        for j in range(blk_sz):
            A[i, j] = np.cos(np.pi * i * (2 * j + 1) / (2 * blk_sz)) * np.sqrt(2 / blk_sz)
    for h in range(n_blk_h):
        for w in range(n_blk_w):
            middle = np.zeros((blk_sz, blk_sz))
            temp = A.dot(src[h * blk_sz : (h + 1) * blk_sz, w * blk_sz : (w + 1) * blk_sz]).dot(A.T)
            middle[: blk_sz // ratio, : blk_sz // ratio] = temp[: blk_sz // ratio, : blk_sz // ratio]
            out[h * blk_sz : (h + 1) * blk_sz, w * blk_sz : (w + 1) * blk_sz] = A.T.dot(middle).dot(A)
    print("psnr", psnr(src, out))
    cv2.imwrite("lena_{}_{}.png".format(ratio, blk_sz), (out + 128).astype(np.uint8))


if __name__ == "__main__":
    src = cv2.imread("lena.png")
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY).astype(np.float64)
    src -= 128
    for ratio in ratios:
        print("ratio", ratio)
        for blk_sz in blk_szs:
            dct(src, ratio, blk_sz)
    dic = {"jpeg": Q_jpeg, "canon": Q_canon, "nikon": Q_nikon}
    for name, Q_now in dic.items():
        H, W = src.shape
        blk_sz = 8
        n_blk_h, n_blk_w = H // blk_sz, W // blk_sz
        out = np.zeros_like(src)
        A = np.zeros((blk_sz, blk_sz))
        A[0, :] = 1 * np.sqrt(1 / blk_sz)
        for i in range(1, blk_sz):
            for j in range(blk_sz):
                A[i, j] = np.cos(np.pi * i * (2 * j + 1) / (2 * blk_sz)) * np.sqrt(2 / blk_sz)
        psnrs = []
        for i in range(1, 21):
            a = 0.1 * i
            Q = Q_now * a
            for h in range(n_blk_h):
                for w in range(n_blk_w):
                    middle = np.zeros((blk_sz, blk_sz))
                    temp = A.dot(src[h * blk_sz : (h + 1) * blk_sz, w * blk_sz : (w + 1) * blk_sz]).dot(A.T)
                    middle = np.round(temp / Q)
                    out[h * blk_sz : (h + 1) * blk_sz, w * blk_sz : (w + 1) * blk_sz] = A.T.dot(middle * Q).dot(A)
            psnrs.append(psnr(src, out))
            if i % 5 == 0:
                cv2.imwrite("lena_{}_{}.png".format(name, i), (out + 128).astype(np.uint8))
        print(name, psnrs)
