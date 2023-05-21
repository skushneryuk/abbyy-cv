
import numpy as np
import cv2

from src.global_params import *


def add_paper(im, paper):
    im = im.copy()
    paper = cv2.resize(paper, im.shape[1::-1])

    if len(im.shape) != 3:
        im = im.reshape(*im.shape, 1)
        
    if len(paper.shape) != 3:
        paper = paper.reshape(*paper.shape, 1)

    return np.where(
        ((im == 255).min(axis=-1)).reshape(*im.shape[:2], 1),
        paper,
        im,
    ).squeeze().astype(np.uint8)


def to_grayscale(im):
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


def add_random_print_defects(im, n=5, max_width=0.01, mode='bw', white_p = 0.5):
    im_out = im.copy()
    colors = None
    if mode == 'bw':
        colors = 255 * np.random.binomial(1, white_p, n)
    elif mode == 'uniform':
        colors = 255 * np.random.uniform(0, 1, n)
    for color in colors:
        while True:
            start = np.random.uniform(0, im.shape[1])
            end = min(im.shape[1], start + np.random.uniform(0, im.shape[1] * max_width))
            start, end = int(start), int(end)
            if end == start:
                continue
            im_out[:, start:end] = color
            break
    return im_out


def add_low_contrast(im, alpha=0.75):
    return cv2.convertScaleAbs(im, alpha=alpha, beta=1)


def gauss_light(H, W, x0, y0, sigma=10000, gauss_scale=0.25):
    X, Y = np.mgrid[:H, :W]
    return np.exp(-((X - x0)**2 + (Y - y0)**2) / sigma) * gauss_scale


def add_uneven_lighting(im, n=3, sigma=0.2, gauss_scale=0.25, smooth_kernel=5):
    sigma = min(im.shape[0], im.shape[1]) * sigma
    sigma = sigma ** 2

    im_out = im / 255
    H, W = im.shape[:2]
    for _ in range(n):
        x0, y0 = np.random.uniform(0, 1, 2) * np.array([H, W])
        im_out = np.clip(im_out + gauss_light(H, W, x0, y0, sigma, gauss_scale), 0, 1)
    im_out = cv2.GaussianBlur(im_out, (smooth_kernel, smooth_kernel), 0, 0)
    return (im_out * 255).astype(np.uint8)


def band_limited_noise(min_freq, max_freq, samples=44100, step=10):
    t = np.linspace(0, 1, samples)
    freqs = np.arange(min_freq, max_freq+1, step)
    phases = np.random.rand(len(freqs))*2*np.pi
    signals = np.array([np.sin(2*np.pi*freq*t + phase) for freq, phase in zip(freqs,phases)])
    signal = signals.sum(axis=0)
    signal /= np.max(signal)
    return signal


def add_hf_noise(im, coef=0.05):
    im_out = im / 255 if im.dtype == np.uint8 else im

    noise_x = band_limited_noise(500, 2500, im_out.shape[1], 1)
    noise_y = band_limited_noise(500, 2500, im_out.shape[0], 1)

    noise = noise_y.reshape(-1, 1) * noise_x.reshape(1, -1)
    noise = (noise + 1) / 2

    if len(im_out.shape) == 3:
        noise = noise.reshape(im_out.shape[0], im_out.shape[1], 1)
    
    return (255 * (im_out * (1 - coef) + noise * coef)).astype(np.uint8)


def get_bayer_masks(n_rows, n_cols):
    result = np.zeros((n_rows, n_cols, 3))

    cols, rows = np.meshgrid(np.arange(n_cols), np.arange(n_rows))

    result[:, :, 0] = np.where((cols % 2 == 0) & (rows % 2 == 0), 1, 0)  # Red
    result[:, :, 1] = np.where((cols + rows + 1) % 2 == 0, 1, 0)         # Green
    result[:, :, 2] = np.where((cols % 2 == 1) & (rows % 2 == 1), 1, 0)  # Blue
    return result.astype(bool)


def get_colored_img(raw_img):
    return np.where(get_bayer_masks(*raw_img.shape), raw_img[..., np.newaxis], 0)


def mosaicing(img):
    return (img * get_bayer_masks(*img.shape[:2])).sum(axis=-1).astype(np.uint8)


def get_shift(img, shift_x, shift_y):
    result = np.zeros(img.shape, dtype=np.float32)
    from_x1, from_x2 = max(0, shift_x), img.shape[0] + min(0, shift_x)
    from_y1, from_y2 = max(0, shift_y), img.shape[1] + min(0, shift_y)

    to_x1, to_x2 = max(0, -shift_x), img.shape[0] + min(0, -shift_x)
    to_y1, to_y2 = max(0, -shift_y), img.shape[1] + min(0, -shift_y)
    result[to_x1:to_x2, to_y1:to_y2] += img[from_x1:from_x2, from_y1:from_y2]
    return result


def shift_conv(img, shifts_x=None, shifts_y=None, weights=None, normed=True):
    result = np.zeros(img.shape, dtype=np.float32)
    if shifts_x is None or shifts_y is None:
        shifts_x = np.array([ 1, 1, 1, 0, 0,  0, -1, -1, -1])
        shifts_y = np.array([-1, 0, 1, 1, 0, -1, -1,  0,  1])
    else:
        shifts_x = np.array(shifts_x)
        shifts_y = np.array(shifts_y)
    weights = np.ones_like(shifts_x) if weights is None else np.array(weights)

    for shift_x, shift_y, weight in zip(shifts_x, shifts_y, weights):
        result += get_shift(img, shift_x, shift_y) * weight

    if normed:
        result /= weights.sum()
    return result


def bilinear_interpolation(colored_img):
    bayer_mask = get_bayer_masks(*colored_img.shape[:-1])
    img_conv = shift_conv(colored_img)
    mask_conv = shift_conv(bayer_mask)
    inter = (img_conv / mask_conv).astype(dtype=np.uint8)

    # для известных значений ничего не интерполируем
    return colored_img + inter * (1 - bayer_mask)


def demosaicing_damage(img):
    return bilinear_interpolation(get_colored_img(mosaicing(img))).astype(np.uint8)


def bilateral_denoising(im, k=7):
    return cv2.bilateralFilter(im, k, 75, 75).astype(np.uint8)

