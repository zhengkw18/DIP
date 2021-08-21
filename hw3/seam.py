import numpy as np
import cv2, argparse
from tqdm import tqdm


def rotate_image(img, cw):
    return np.rot90(img, 1 if cw else 3)


def fast_argmin_axis_0(a):
    matches = np.nonzero((a == np.min(a, axis=0)).ravel())[0]
    rows, cols = np.unravel_index(matches, a.shape)
    argmin_array = np.empty(a.shape[1], dtype=np.intp)
    argmin_array[cols] = rows
    return argmin_array


def energy_function(image):
    H, W, _ = image.shape
    out = np.zeros((H, W))

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    dx, dy = np.gradient(gray_image)
    out = np.abs(dx) + np.abs(dy)

    return out


def get_min_seam(img, remove_mask=None):
    h, w, _ = img.shape
    M = energy_function(img)
    if remove_mask is not None:
        M[np.where(remove_mask > 30)] = -1e3

    dp = np.zeros_like(M, dtype=np.int32)
    for i in range(1, h):
        for j in range(0, w):
            left = max(j - 1, 0)
            idx = np.argmin(M[i - 1, left : j + 2])
            dp[i, j] = idx + left
            M[i, j] += M[i - 1, idx + left]

    seam_idx = []
    j = np.argmin(M[-1])
    for i in range(h - 1, -1, -1):
        seam_idx.append(j)
        j = dp[i, j]

    seam_idx.reverse()
    return np.array(seam_idx)


def _remove_seam_impl(img, seam_idx):
    h, w = img.shape[:2]
    bin_mask = np.ones((h, w), dtype=np.int32)
    bin_mask[np.arange(h), seam_idx] = 0
    new_shape = list(img.shape)
    new_shape[1] -= 1
    if len(img.shape) == 3:
        # color img
        bin_mask = np.stack([bin_mask] * 3, axis=2)
    return img[bin_mask > 0].reshape(new_shape)


def _add_seam_impl(img, seam_idx):
    new_shape = list(img.shape)
    new_shape[1] += 1
    output = np.zeros(new_shape)
    h, w = img.shape[:2]
    for row in range(h):
        col = seam_idx[row]
        left = max(col - 1, 0)
        output[row, :col] = img[row, :col]
        output[row, col] = np.average(img[row, left : col + 2], axis=0)
        output[row, col + 1 :] = img[row, col:]
    return np.round(output).astype(np.uint8)


def manip_seam(img, delta):
    seams = []
    temp_img = img.copy()
    last_seam = None
    for _ in tqdm(range(abs(delta))):
        seam_idx = get_min_seam(temp_img)
        temp_img = _remove_seam_impl(temp_img, seam_idx)
        if last_seam is not None:
            seam_idx[seam_idx > last_seam] += 1
        last_seam = seam_idx
        seams.append(last_seam)

    if delta < 0:
        return temp_img, seams
    else:
        seams.reverse()
        for i in tqdm(range(delta)):
            seam = seams[i]
            img = _add_seam_impl(img, seam)
            for j in range(delta):
                seams[j][np.where(seams[j] > seam)] += 1
        return img, seams


def seam_carve(img, dx, dy):
    h, w, _ = img.shape
    assert w + dx > 0 and h + dy > 0 and dx < w and dy < h
    output = img.copy()
    seams_x = None
    seams_y = None
    if dx != 0:
        output_x, seams_x = manip_seam(output, dx)
    else:
        output_x = output
    if dy != 0:
        output = rotate_image(output_x, True)
        output, seams_y = manip_seam(output, dy)
        output = rotate_image(output, False)
    else:
        output = output_x
    return output


def object_removal(img, remove_mask, remove_horiz=False):
    h, w, _ = img.shape
    output = img
    if remove_horiz:
        output = rotate_image(output, True)
        remove_mask = rotate_image(remove_mask, True)

    last = -1
    while True:
        remain = len(np.where(remove_mask > 30)[0])
        if remain == last:
            break
        last = remain
        print("Remaining: {}".format(remain))
        if remain == 0:
            break
        seam_idx = get_min_seam(output, remove_mask)
        output = _remove_seam_impl(output, seam_idx)
        remove_mask = _remove_seam_impl(remove_mask, seam_idx)

    cv2.imwrite("partial.jpg", output)
    num_enlarge = (h if remove_horiz else w) - output.shape[1]
    print("Start to restore")
    output, _ = manip_seam(output, num_enlarge)
    print("Done")
    if remove_horiz:
        output = rotate_image(output, False)

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Seam carve")
    parser.add_argument("--src", help="Source image", required=True)
    parser.add_argument("--op", help="Operation (seam, remove)", default="seam")
    parser.add_argument("--rm_mask", help="Mask of the part to remove (only in REMOVE op)", default="")
    parser.add_argument("--out", help="Output file (optional)", default="")
    parser.add_argument("--dx", help="Number of cols to add/remove", type=int, default=0)
    parser.add_argument("--dy", help="Number of rows to add/remove", type=int, default=0)

    args = parser.parse_args()

    src = cv2.imread(args.src)
    assert src is not None

    if args.op == "seam":
        output = seam_carve(src, args.dx, args.dy)
    elif args.op == "remove":
        rm_mask = cv2.imread(args.rm_mask, 0)
        output = object_removal(src, rm_mask)
    else:
        raise NotImplementedError("Unsupported type of op: {}".format(args.op))
    if args.out:
        cv2.imwrite(args.out, output)