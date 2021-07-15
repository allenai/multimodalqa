import argparse
import json
import os

import scipy.misc as m
from tqdm import tqdm


def create_json(base_json, img_root, split):
    img_paths = os.listdir(img_root)
    out_dict = json.load(open(base_json, 'r'))
    out_dict['images'] = []

    for img_path in tqdm(img_paths):
        img = m.imread(os.path.join(img_root, img_path))
        height, width = img.shape[0], img.shape[1]
        out_dict['images'].append({'file_name': img_path,
                                   'height': height,
                                   'width': width,
                                   'id': int(img_path[-16:-4])})

    with open('vqa_{}.json'.format(split), 'w') as fp:
        json.dump(out_dict, fp)

if __name__ = '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_json",
        default="datasets/vg/annotations/visual_genome_categories.json",
        metavar="FILE",
    )
    parser.add_argument(
        "--img_root",
        default="/datasets01/COCO/060817/val2014",
    )
    parser.add_argument(
        "--split",
        default="val2014",
    )
    args = parser.parse_args()

    create_json(args.base_json, args.img_root, args.split)
