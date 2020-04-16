"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved. ;)
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import argparse
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
from skimage.draw import polygon

parser = argparse.ArgumentParser()
parser.add_argument('--annotation_file', type=str, default="./annotations/instances_train2017.json",
                    help="Path to the annotation file.")
parser.add_argument('--input_image_dir', type=str, default="./train_img/",
                    help="Path to the directory containing img maps.")
parser.add_argument('--output_label_dir', type=str, default="./train_label/",
                    help="Path to the output directory of label maps")

opt = parser.parse_args()

print("annotation file at {}".format(opt.annotation_file))
print("input img maps at {}".format(opt.input_label_dir))
print("output dir at {}".format(opt.output_instance_dir))

# initialize COCO api for instance annotations
coco = COCO(opt.annotation_file)


# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
imgIds = coco.getImgIds(catIds=coco.getCatIds(cats))
for ix, id in enumerate(imgIds):
    if ix % 50 == 0:
        print("{} / {}".format(ix, len(imgIds)))
    img_dict = coco.loadImgs(id)[0]
    filename = img_dict["file_name"].replace("jpg", "png")
    label_name = os.path.join(opt.input_label_dir, filename)
    inst_name = os.path.join(opt.output_instance_dir, filename)
    img = io.imread(label_name, as_gray=True)
    img[:,:]=255
    annIds = coco.getAnnIds(imgIds=id, catIds=[], iscrowd=None)
    anns = coco.loadAnns(annIds)
    count = 0
    for ann in anns:
        if type(ann["segmentation"]) == list:
            if "segmentation" in ann:
                for seg in ann["segmentation"]:
                    poly = np.array(seg).reshape((int(len(seg) / 2), 2))
                    rr, cc = polygon(poly[:, 1] - 1, poly[:, 0] - 1)
                    img[rr, cc] = ann["category_id"]-1
                count += 1
    
    io.imsave(inst_name, img)
