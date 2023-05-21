import numpy as np
from skimage.transform import ProjectiveTransform

import cv2

from src.global_params import *
from src.transforms import warp_image, get_flatten_transform, adjust_image


def corner_mouse_handler(event, x, y, flags, data):
    if event == cv2.EVENT_LBUTTONDOWN:
        data['points'].append([x, y])
        if len(data['points']) > 4:
            data['points'] = data['points'][-4:]

        data['im'] = data['im_default'].copy()
        for x, y in data['points']:
            cv2.circle(data['im'], (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Image", data['im'])


def get_four_points(im, win_h = STANDARD_WIH_H, win_w = STANDARD_WIH_W):
    im, k = adjust_image(im, win_h = win_h, win_w = win_w)

    data = {}
    data['im_default'] = im.copy()
    data['im'] = im.copy()
    data['points'] = []

    cv2.imshow("Image", im)
    cv2.setMouseCallback("Image", corner_mouse_handler, data)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 
    points = np.float32(data['points'])
    return points * k


def flatten_table(im_src, real_H, real_W, left_space=True, left_space_ratio = 0.05):
    '''
    params:
    im_src: original image of table
    real_H: real height of table in mm
    real_W: real width of table in mm

    return:
    dict with results
    '''
    # Show image and wait for 4 clicks.
    pts_src = get_four_points(im_src)
    output_H, output_W = real_H, real_W

    # Destination coordinates located in the center of the image
    pts_dst = np.float32([
        [0, 0],
        [real_W, 0],
        [real_W, real_H],
        [0, real_H],
    ])

    if left_space:
        pts_dst_min = pts_dst.min(axis=0)
        pts_dst_max = pts_dst.max(axis=0)
        pts_dst_center = (pts_dst_min + pts_dst_max) / 2

        pts_dst = (pts_dst - pts_dst_center) + pts_dst_center / (1 - left_space_ratio)
        output_H, output_W = int(output_H  / (1 - left_space_ratio)), int(output_W / (1 - left_space_ratio))

    # Calculate the homography
    transform = get_flatten_transform(pts_dst, pts_src)

    # Warp source image to destination
    im_out, _ = warp_image(im_src, transform, (output_H, output_W))
    im_out_adjusted, _ = adjust_image(im_out)

    # Show output
    cv2.imshow("Warped Source Image", im_out_adjusted)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return {
        "transform": transform,
        "im_src": im_src,
        "im_out": im_out,
        "pts_src": pts_src,
        "pts_dst": pts_dst,
    }