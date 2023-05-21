import numpy as np
import cv2
import sys
import os
import json
sys.path.insert(0, './task4/src')

from src.global_params import *
from src.tables import flatten_table
from src.transforms import adjust_image


if __name__ == "__main__":
    with open("task4/params.json", 'r') as f:
        params = json.load(f)
    
    processed_tables = {table['name']:table for table in params['tables']}
    params['tables'] = []

    for name in os.listdir("task4/tables"):
        if name in processed_tables:
            ans = input(f"Reprocess table {name}? ")
            if ans.lower() not in ['y', 'yes']:
                params['tables'].append(processed_tables[name])
                continue
        table_img = cv2.imread(os.path.join("./task4/tables/", name))

        cv2.imshow("First look", adjust_image(table_img)[0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
            
        H, W = map(int, input("Print height and width of table in mm: ").split())

        results = flatten_table(table_img, H, W, left_space=False)
        results.pop('im_src')
        results.pop('im_out')
        results['name'] = name
        results['H'] = H
        results['W'] = W
        results['transform'] = results['transform'].params.tolist()
        results['pts_src'] = results['pts_src'].tolist()
        results['pts_dst'] = results['pts_dst'].tolist()

        params['tables'].append(results)

    with open("task4/params.json", 'w') as f:
        json.dump(params, f)
