import numpy as np
import torch

from pycocotools.coco import COCO
import clip

DATA_DIR = 'coco'
DATA_SPLITS = ['val2017', 'train2017']
ANN_FILE_FMT = '{}/annotations/captions_{}.json'    # .format(DATA_DIR, split)

for split in DATA_SPLITS:
  caption_file = ANN_FILE_FMT.format(DATA_DIR, split)
  coco_caps = COCO(caption_file)
  img_ids = coco_caps.getImgIds()
  print(len(img_ids))
  ann_ids = coco_caps.getAnnIds(imgIds=img_ids)
  print(len(ann_ids))
