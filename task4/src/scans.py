import numpy as np
import cv2

from src.global_params import *
from src.transforms import scan_transform, adjust_image, project_image


def project_scan(scan, background, paper_size=(297, 210), angle=0, position=(0, 0), scale=None, adjust=True):
    if scale is None:
        scale = scan.shape[1] / paper_size[1]
    transform = scan_transform(angle=angle, position=position, scale=scale)

    im_out = project_image(scan, background, transform)
    if adjust:
        im_out = adjust_image(im_out)[0]
    return im_out


def position_mouse_handler(event, x, y, _, data):
    if event == cv2.EVENT_LBUTTONDOWN:
        data['position'] = (y * data['adjust_scale'], x * data['adjust_scale'])
        transform = scan_transform(
            angle=data['angle'],
            position=data['position'],
            scale=data['scale'],
            paper_size=data['paper_size'],
            adjust_scale=data['adjust_scale'],
        )

        im_out = project_image(data['scan'], data['background'], transform)
        im_out = adjust_image(im_out)[0]
        im_out = cv2.circle(im_out, (x, y), 5, (255, 0, 0), -1)
        cv2.imshow(data['name'], im_out)
