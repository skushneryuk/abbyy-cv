import numpy as np
import cv2
import sys
import os
import json
from skimage.transform import ProjectiveTransform
from tqdm.auto import tqdm
sys.path.insert(0, './task4/src')

from src.global_params import *
from src.tables import flatten_table
from src.scans import position_mouse_handler
from src.transforms import scan_transform, project_image, adjust_image, inverse_transform
from src.damage import *


if __name__ == "__main__":
    with open("task4/params.json", 'r') as f:
        params = json.load(f)
    
    processed_tables = {table['name']:table for table in params['tables']}
    scans = dict()
    papers = list()

    for name in processed_tables:
        processed_tables[name]['src_img'] = cv2.imread(os.path.join("./task4/tables/", name))
    
    for name in os.listdir("./task4/scans/"):
        scans[name] = cv2.imread(os.path.join("./task4/scans/", name))
    
    for name in os.listdir("./task4/paper/"):
        papers.append(cv2.imread(os.path.join("./task4/paper/", name)))

    print('Total scans: ', len(scans))
    print('Total papers: ', len(papers))
    print('Total tables: ', len(processed_tables))

    paper_size = (297, 210)
    diag = np.sqrt(297 ** 2 + 210 ** 2)
    for _, (scan_name, scan) in tqdm(zip(range(10), scans.items())):
        for _, (table_name, table) in tqdm(zip(range(10), processed_tables.items())):
            # prepare scan
            for iter in range(2):
                curr = scan.copy()
                curr = add_paper(curr, papers[np.random.randint(0, len(papers))])
                curr = to_grayscale(curr)
                curr = add_random_print_defects(
                    curr,
                    np.random.randint(3, 7),
                    white_p=np.random.uniform(0.3, 0.7),
                )
                curr = add_low_contrast(
                    curr,
                    np.random.uniform(0.65, 0.85)
                )
                curr = add_uneven_lighting(
                    curr,
                    np.random.randint(2, 5),
                )

                position = (
                    np.random.uniform(diag, table['H'] - diag),
                    np.random.uniform(diag, table['W'] - diag),
                )
                angle = np.random.uniform(-np.pi, np.pi)
                scale = curr.shape[1] / 210
                transform = scan_transform(
                    angle=angle,
                    position=position,
                    scale=scale,
                )
                table_transform = ProjectiveTransform(
                    np.array(table['transform']),
                )
                transform = inverse_transform(table_transform) + transform

                im_out = project_image(curr, table['src_img'], transform)
                im_out = add_hf_noise(im_out)
                im_out = bilateral_denoising(im_out)
                # im_out = demosaicing_damage(im_out)

                cv2.imwrite("./task4/processed/{}_{}_{}.jpg".format(scan_name[:-4], table_name[:-4], iter + 1), im_out)
