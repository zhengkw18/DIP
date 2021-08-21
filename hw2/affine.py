from scipy import interpolate
import cv2
import numpy as np

src_keypoints = []
target_keypoints = []

state = 0
finish = False


def on_click(event, x, y, flags, param):
    global state, finish
    if event == cv2.EVENT_LBUTTONUP:
        if state == 0:
            src_keypoints.append((x, y))
            cv2.drawMarker(src, (x, y), (0, 255, 255))
            if len(src_keypoints) >= 2:
                cv2.line(src, src_keypoints[-1], src_keypoints[-2], (0, 0, 255))
            cv2.imshow("image", src)
            if len(src_keypoints) == 4:
                state = 1
                cv2.line(src, src_keypoints[-1], src_keypoints[0], (0, 0, 255))
                cv2.imshow("image", src)
                print("Source selection okay")
                cv2.waitKey()
                cv2.imshow("image", target)
        elif state == 1:
            target_keypoints.append((x, y))
            cv2.drawMarker(target, (x, y), (0, 255, 255))
            if len(target_keypoints) >= 2:
                cv2.line(target, target_keypoints[-1], target_keypoints[-2], (0, 0, 255))
            cv2.imshow("image", target)
            if len(target_keypoints) == 4:
                state = 2
                cv2.line(target, target_keypoints[-1], target_keypoints[0], (0, 0, 255))
                cv2.imshow("image", target)
                print("Target selection okay")
                finish = True


def get_color(src, mat, x, y):
    res = mat.dot(np.array((x, y, 1)).T)[:2]
    x1 = int(np.floor(res[0]))
    x2 = int(np.ceil(res[0]))
    y1 = int(np.floor(res[1]))
    y2 = int(np.ceil(res[1]))
    x = res[0]
    y = res[1]
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


def get_color_no_interp(src, mat, x, y):
    res = mat.dot(np.array((x, y, 1)).T)[:2]
    return src[int(np.round(res[1])), int(np.round(res[0]))]


def bilinear_smooth(img, x, y):
    return np.round(0.25 * img[x + 1, y] + 0.25 * img[x, y + 1] + 0.25 * img[x - 1, y] + 0.25 * img[x, y - 1]).astype(np.uint8)


interp = None


if __name__ == "__main__":
    src = "source.jpg"
    target = "target.jpg"
    src = cv2.imread(src)
    target = cv2.imread(target)
    src_ = src.copy()
    target_ = target.copy()
    # cv2.namedWindow("image")
    # cv2.setMouseCallback("image", on_click)
    # while True:
    #     cv2.imshow("image", src)
    #     key = cv2.waitKey() & 0xFF
    #     if finish:
    #         break
    # print(src_keypoints)
    # print(target_keypoints)
    src_keypoints = [(157, 176), (157, 411), (575, 411), (575, 176)]
    target_keypoints = [(193, 194), (169, 316), (509, 388), (536, 264)]
    b = []
    for i in target_keypoints:
        b.extend(i)
    b = np.array(b).T
    A = []
    for i in src_keypoints:
        t1 = i + (1,) + (0,) * 3
        t2 = (0,) * 3 + i + (1,)
        A.append(t1)
        A.append(t2)
    A = np.array(A)
    # print(A, b)
    T = np.linalg.solve(A.T.dot(A), A.T.dot(b))
    T = np.vstack([T.reshape(2, 3), [0, 0, 1]])
    T_inv = np.linalg.inv(T)

    # fill color
    target_ = target_.astype(np.int32)
    for i in range(4):
        cv2.line(target_, target_keypoints[i % 4], target_keypoints[(i + 1) % 4], (-1, -1, -1))

    seed = np.round(np.mean(np.array(target_keypoints), axis=0)).astype(int)

    img = target_
    img_no_interp = target_.copy()
    borders = []
    for i in range(img.shape[0]):
        border = []
        for j in range(img.shape[1]):
            if (img[i, j] < 0).all():
                border.append(j)
        if len(border):
            for j in range(border[0], border[-1] + 1):
                img[i, j] = get_color(src_, T_inv, j, i)
                img_no_interp[i, j] = get_color_no_interp(src_, T_inv, j, i)
        borders.append(border)
    img_border_smooth = img.copy()
    for k in range(10):
        for i in range(img_border_smooth.shape[0]):
            border = borders[i]
            if len(border):
                j1 = border[0]
                j2 = border[-1]
                img_border_smooth[i, j1] = bilinear_smooth(img_border_smooth, i, j1)
                img_border_smooth[i - 1, j1] = bilinear_smooth(img_border_smooth, i - 1, j1)
                img_border_smooth[i, j1 - 1] = bilinear_smooth(img_border_smooth, i, j1 - 1)
                img_border_smooth[i + 1, j1] = bilinear_smooth(img_border_smooth, i + 1, j1)
                img_border_smooth[i, j1 + 1] = bilinear_smooth(img_border_smooth, i, j1 + 1)

                img_border_smooth[i, j2] = bilinear_smooth(img_border_smooth, i, j2)
                img_border_smooth[i - 1, j2] = bilinear_smooth(img_border_smooth, i - 1, j2)
                img_border_smooth[i, j2 - 1] = bilinear_smooth(img_border_smooth, i, j2 - 1)
                img_border_smooth[i + 1, j2] = bilinear_smooth(img_border_smooth, i + 1, j2)
                img_border_smooth[i, j2 + 1] = bilinear_smooth(img_border_smooth, i, j2 + 1)

    cv2.imwrite("interp.jpg", img)
    cv2.imwrite("interp_border_smooth.jpg", img_border_smooth)
    cv2.imwrite("no_interp.jpg", img_no_interp)
