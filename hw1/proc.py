import cv2
import numpy as np


def gamma_correction(image, gamma=2.2):
    return np.array(np.power(image / 255.0, 1.0 / gamma) * 255).astype("uint8")


def scale(image, k=0.8):
    return np.array(image * k).astype("uint8")


def logistic_contrast(image):
    return np.array(255.0 / (1.0 + np.exp(-11.0 * (image / 255.0) + 5.5))).astype("uint8")


def reverse_logistic_contrast(image):
    return np.array(255.0 * (5.5 - np.log(np.ones_like(image) * 255.0 / np.maximum(image, 1))) / 11).astype("uint8")


def get_cdf(hist):
    cdf = hist.cumsum()
    return cdf / float(cdf.max())


def get_lut(src_cdf, ref_cdf):
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        lut[i] = np.searchsorted(ref_cdf, src_cdf[i])
    return lut


def match_channel_histogram(src, ref):
    h_src, _ = np.histogram(src, 256, [0, 256])
    h_ref, _ = np.histogram(ref, 256, [0, 256])
    s_cdf = get_cdf(h_src)
    r_cdf = get_cdf(h_ref)
    return cv2.LUT(src, get_lut(s_cdf, r_cdf))


def equalize_channel_histogram(channel):
    hist, _ = np.histogram(channel, 256, [0, 256])
    cdf = get_cdf(hist)
    return cv2.LUT(channel, np.round(255 * cdf).astype(np.uint8))


def match_image_histogram(src, ref):
    sb, sg, sr = cv2.split(src)
    rb, rg, rr = cv2.split(ref)

    b_res = match_channel_histogram(sb, rb)
    g_res = match_channel_histogram(sg, rg)
    r_res = match_channel_histogram(sr, rr)

    return cv2.convertScaleAbs(cv2.merge([b_res, g_res, r_res]))


def match_b_w_gr(src):
    sb, sg, sr = cv2.split(src)
    sb = match_channel_histogram(sb, (sg + sr) / 2)
    return cv2.convertScaleAbs(cv2.merge([sb, sg, sr]))


def match_g_w_br(src):
    sb, sg, sr = cv2.split(src)
    sg = match_channel_histogram(sg, (sb + sr) / 2)
    return cv2.convertScaleAbs(cv2.merge([sb, sg, sr]))


def match_r_w_bg(src):
    sb, sg, sr = cv2.split(src)
    sr = match_channel_histogram(sr, (sb + sg) / 2)
    return cv2.convertScaleAbs(cv2.merge([sb, sg, sr]))


def match_r_w_g(src):
    sb, sg, sr = cv2.split(src)
    sr = match_channel_histogram(sr, sg)
    return cv2.convertScaleAbs(cv2.merge([sb, sg, sr]))


def equalize_all_channels(src):
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    hist, _ = np.histogram(gray, 256, [0, 256])
    lut = np.round(255 * get_cdf(hist)).astype(np.uint8)
    res = cv2.LUT(src, lut)
    return cv2.convertScaleAbs(res)


def equalize_all_channels_separate(src):
    sb, sg, sr = cv2.split(src)
    hist_b, _ = np.histogram(sb, 256, [0, 256])
    hist_g, _ = np.histogram(sg, 256, [0, 256])
    hist_r, _ = np.histogram(sr, 256, [0, 256])
    lut_b = np.round(255 * get_cdf(hist_b)).astype(np.uint8)
    lut_g = np.round(255 * get_cdf(hist_g)).astype(np.uint8)
    lut_r = np.round(255 * get_cdf(hist_r)).astype(np.uint8)
    sb = cv2.LUT(sb, lut_b)
    sg = cv2.LUT(sg, lut_g)
    sr = cv2.LUT(sr, lut_r)
    return cv2.convertScaleAbs(cv2.merge([sb, sg, sr]))


if __name__ == "__main__":
    src = cv2.imread("03.png")
    # ref = cv2.imread("02.bmp")
    out = scale(src)
    out = logistic_contrast(out)
    out = gamma_correction(out, 1.4)
    cv2.imwrite("out-.png", out)
