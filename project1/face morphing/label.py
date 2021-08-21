import argparse
import json
import cv2
import numpy as np

cnt = 0
lst = []
ref = []
left = None
right = None
current_left = None
current_right = None


def concat_img(left, right):
    H = max(left.shape[0], right.shape[0])
    return np.concatenate((np.pad(left, ((0, H - left.shape[0]), (0, 0), (0, 0)), "constant"), np.pad(right, ((0, H - right.shape[0]), (0, 0), (0, 0)), "constant")), axis=1)


def callback(event, x, y, flags, param):
    global cnt, lst, ref, left, right, current_left, current_right
    # label
    if event == cv2.EVENT_LBUTTONDOWN and y < left.shape[0] and x < left.shape[1] and cnt < len(ref):
        lst.append((y, x))
        cnt += 1
        cv2.circle(current_left, (x, y), 1, (0, 0, 255))
        current_right = right.copy()
        if cnt < len(ref):
            cv2.circle(current_right, (ref[cnt][1], ref[cnt][0]), 1, (0, 0, 255))
        cv2.imshow("Label", concat_img(current_left, current_right))
    # cancel
    elif event == cv2.EVENT_RBUTTONDOWN and cnt > 0:
        lst.pop(-1)
        cnt -= 1
        current_left = left.copy()
        for y, x in lst:
            cv2.circle(current_left, (x, y), 1, (0, 0, 255))
        current_right = right.copy()
        cv2.circle(current_right, (ref[cnt][1], ref[cnt][0]), 1, (0, 0, 255))
        cv2.imshow("Label", concat_img(current_left, current_right))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Manually label for picture using a reference")
    parser.add_argument("--ref", type=str, help="Reference image", required=True)
    parser.add_argument("--src", type=str, help="Source image", required=True)

    args = parser.parse_args()
    with open(args.ref + ".json", "r") as f:
        ref = json.load(f)
    left = cv2.imread(args.src)
    right = cv2.imread(args.ref)
    cv2.namedWindow("Label")
    cv2.setMouseCallback("Label", callback)
    current_left = left.copy()
    current_right = right.copy()
    cv2.circle(current_right, (ref[cnt][1], ref[cnt][0]), 1, (0, 0, 255))
    cv2.imshow("Label", concat_img(current_left, current_right))
    while cnt < len(ref):
        cv2.waitKey()
    with open(args.src + ".json", "w") as f:
        json.dump(lst, f)
