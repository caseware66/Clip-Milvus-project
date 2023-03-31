from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)
import numpy as np
import torch
import sys

from PIL import Image
from pycocotools.coco import COCO
import clip

QUERY_IMG_ID = 37777

CLIP_MODEL = 'ViT-L/14@336px'
DATA_DIR = 'coco'
DATA_SPLIT = 'train2017'
ANN_FILE_FMT = '{}/annotations/captions_{}.json'    # .format(DATA_DIR, split) 
IMAGE_FILE_FMT = '{}/images/{}/{}.jpg'              # .format(DATA_DIR, split, photo id zero-filled to width 12)
COLLECTION_NAME = "coco_imgcaptions"


connections.connect("default", host="localhost", port="19530")


coco_val = Collection(COLLECTION_NAME)

# print("start loading collection, may take a long time")
# coco_val.load()
# print("load complete, begin queries")

# ============ CHECK IMAGE IS IN COLLECTION =========
# find_image_expr = "id == {}".format(QUERY_IMG_ID)
# find_result = coco_val.query(expr=find_image_expr, output_fields=["id", "type", "embedding"])
# if not find_result:
#   print("entry not found")
#   sys.exit()


# # ============ CHECK EMBEDDING IS EQUAL =========
# emb = find_result[0]['embedding']
# emb = np.array(emb)


device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load(CLIP_MODEL, device)
img_filename = "gu.jpg"
with Image.open(img_filename) as img:
  image_input = clip_preprocess(img).unsqueeze(0).to(device)
  with torch.no_grad():
    image_features = clip_model.encode_image(image_input).numpy()
image_features = image_features.flatten()

# is_equal = np.allclose(emb, image_features, atol = 0.001)
# print(f"Image embeddings are similar: {is_equal}")

# ========== TOP 10 MOST SIMILAR CAPTIONS + COMPARISON ==========

search_params = {
    "metric_type": "L2",
    "params": {"nprobe": 10},
}
result = coco_val.search([image_features], "embedding", search_params, limit=10, expr='type == "caption"', output_fields=["id"])

cap_results = []
i = 0
for hits in result:
    for hit in hits:
        print(f"{i+1}. L2 distance {hit.distance}")
        cap_results.append(hit.entity.get('id'))

# Print out captions of result IDs
caption_file = ANN_FILE_FMT.format(DATA_DIR, DATA_SPLIT)
coco_caps = COCO(caption_file)

captions = []
caption_jsons = coco_caps.loadAnns(cap_results)
for json in caption_jsons:
  print(f"ID {json['id']}; Caption: {json['caption']}")
  captions.append(json['caption'])

i = 0
for hits in result:
  for hit in hits:
    print(f"{i+1}. {hit.distance:.3f}; {captions[i]}")
    cap_results.append(hit.entity.get('id'))
    i += 1




# caption_ids = coco_caps.getAnnIds(imgIds=[QUERY_IMG_ID])
# print("True captions associated with image:")
# print(caption_ids)
# caption_jsons = coco_caps.loadAnns(caption_ids)
# for json in caption_jsons:
#   print(f"ID {json['id']}; Caption: {json['caption']}")

# ========== TOP 10 MOST SIMILAR IMAGES ==========
result = coco_val.search([image_features], "embedding", search_params, limit=3, expr='type == "image"')

for hits in result:
    for hit in hits:
        print(f"hit in images: {hit}")


# print("release collection from memory")
# coco_val.release()
# Don't release, as you'll need to load it in again


print("done")