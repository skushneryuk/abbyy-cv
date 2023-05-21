import numpy as np
import cv2
import sys
sys.path.insert(0, './task4/src')

from src.global_params import *
from src.tables import flatten_table
from src.scans import position_mouse_handler
from src.transforms import scan_transform, project_image, adjust_image, inverse_transform


if __name__ == "__main__":
    table_img = cv2.imread("./task4/tables/table_5.jpg")
    scan_img = cv2.imread("./task4/scans/1.png")

    results = flatten_table(table_img, 800, 1200)

    scale = scan_img.shape[1] / 210
    angle = np.pi / 12

    data = {
        'background': results['im_out'],
        'scan': scan_img,
        'paper_size': (297, 210),
        'scale': scan_img.shape[1] / 210,
        'angle': np.pi / 12,
        'position': (0, 0),
        'name': "Positioning scan",
        'adjust_scale': adjust_image(results['im_out'])[1]
    }

    cv2.imshow(data['name'], adjust_image(data['background'])[0])
    cv2.setMouseCallback(data['name'], position_mouse_handler, data)
    while cv2.waitKey(0) != 13:
        continue
    cv2.destroyAllWindows()

    transform = scan_transform(angle=data['angle'], position=data['position'], scale=data['scale'])

    im_out = project_image(scan_img, results['im_out'], transform)
    im_out_adjusted, _ = adjust_image(im_out)

    cv2.imshow("Final: warped table + warped scan", im_out_adjusted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    transform = inverse_transform(results['transform']) + transform
    im_out = project_image(scan_img, table_img, transform)
    im_out_adjusted, _ = adjust_image(im_out)

    cv2.imshow("Original table + warped scan", im_out_adjusted)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
