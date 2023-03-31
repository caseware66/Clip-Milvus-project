from form import textForm
import numpy as np
import torch
import sys
from PIL import Image
from pycocotools.coco import COCO
import clip
import model

DATA_DIR = '/home/vec_sim/coco'
CLIP_MODEL = 'ViT-L/14@336px'

DATA_SPLIT = 'train2017'
ANN_FILE_FMT = '{}/annotations/captions_{}.json'
IMAGE_FILE_FMT = '{}.jpg' 
#IMAGE_FILE_FMT = '{}/images/{}/{}.jpg'
COLLECTION_NAME = "coco_imgcaptions"
#connections.connect("default", host="localhost", port="19530")
#coco_val = Collection(COLLECTION_NAME)
CAPTIONS_FILE = ANN_FILE_FMT.format(DATA_DIR, DATA_SPLIT)



def text2text(caption,milvus_db):
    
    
       
       # device = "cuda" if torch.cuda.is_available() else "cpu"
        #clip_model, clip_preprocess = clip.load(CLIP_MODEL, device)
    #token=clip.tokenize(caption)
    print(caption)
    text_features=model.encode_text(caption).squeeze()
    print(text_features.shape)
    search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10},
        }
        #result = coco_val.search([text_features], "embedding", search_params, limit=10, expr='type == "caption"', output_fields=["id"])
    result_text=milvus_db.search([text_features],'embedding',search_params,limit=10,expr='type == "caption"', output_fields=["id"])
    result_img=milvus_db.search([text_features],'embedding',search_params,limit=10,expr='type == "image"', output_fields=["id"])
    cap_results = []
    img_results=[]
    img_filenames = None
    for hits in result_text:
        for hit in hits:
            print(f"hit in captions: {hit}")
            cap_results.append(hit.entity.get('id'))
    if(result_img):
        for hits in result_img:
            for hit in hits:
                print(f"hit in images: {hit}")
                img_results.append(hit.entity.get('id'))
    img_filenames = [IMAGE_FILE_FMT.format(str(x).zfill(12)) for x in img_results]

    coco_caps = COCO(CAPTIONS_FILE)
        #caption_file = ANN_FILE_FMT.format(DATA_DIR, DATA_SPLIT)
        
    caption_jsons = coco_caps.loadAnns(cap_results)
    
    
    return img_filenames, caption_jsons



