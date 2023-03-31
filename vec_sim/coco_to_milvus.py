import numpy as np
import torch
import sys
import time

from PIL import Image
from pycocotools.coco import COCO
import clip

from pymilvus import (
  connections,
  utility,
  FieldSchema, CollectionSchema, DataType,
  Collection,
)

COLLECTION_NAME = "coco_imgcaptions"
COLLECTION_DESCRIPTION = "store different types of data and their embeddings, keyed by data id"

BATCH_SIZE = 100 # changed from batch size of 50
STARTING_INDEX = 26900 # = (192400-30014)/6 = around 27k images already inserted

DATA_DIR = 'coco'
DATA_SPLITS = ['train2017']
ANN_FILE_FMT = '{}/annotations/captions_{}.json'    # .format(DATA_DIR, split) 
IMAGE_FILE_FMT = '{}/images/{}/{}.jpg'              # .format(DATA_DIR, split, photo id zero-filled to width 12) 

CLIP_MODEL = 'ViT-L/14@336px'
EMB_DIM = 768                                       # p.48 of https://arxiv.org/pdf/2103.00020.pdf
IMG_DIM = (3, 336, 336)
# Credit: https://stackoverflow.com/a/434328
def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def embed_coco_img_batch(coco_obj, batch_img_ids, data_split, clip_model, clip_preprocess, device, ret_img=False):
  """
  Embed the images with the given ids and the 5 captions associated with each image
  :param  coco_obj  (object initialized with COCO())  : to load COCO data
          batch_img_ids (int list)                    : embed imgs and captions for these img ids
          data_split (string)                         : which data split to use
          clip_model (model)                          : returned from clip.load()
          clip_preprocess (preprocess function)       : returned from clip.load()
          device (string)                             : device to do calculations on

  :return batch_img_ids (int list)                    : length BATCH_SIZE
          image_embeddings (np array)                 : (BATCH_SIZE, EMB_DIM)
          batch_cap_ids (int list)                    : length BATCH_SIZE * 5
          text_embeddings (np array)                  : (BATCH_SIZE * 5, EMB_DIM)
  """

  # Embed images
  N = len(batch_img_ids)
  batch_image_filenames = [IMAGE_FILE_FMT.format(DATA_DIR, data_split, str(id).zfill(12)) for id in batch_img_ids]
  batch_preprocess = []
  batch_img = np.zeros((N,)+ IMG_DIM)
  i = 0 
  for img_filename in batch_image_filenames:
    with Image.open(img_filename) as img:
      cimg = clip_preprocess(img)
      batch_preprocess.append(cimg.unsqueeze(0))
      batch_img[i,...] = cimg
      i += 1 
  with torch.no_grad():
    image_embeddings = clip_model.encode_image(torch.cat(batch_preprocess).to(device)).cpu().numpy()

  # OLD CODE:
  # image_embeddings = np.zeros((len(batch_img_ids), EMB_DIM))
  # for i, img_filename in enumerate(batch_image_filenames):
  #   with Image.open(img_filename) as img:
  #     image_input = clip_preprocess(img).unsqueeze(0).to(device)
  #     with torch.no_grad():
  #       image_features = clip_model.encode_image(image_input).numpy()
  #       image_embeddings[i] = image_features

  # Embed captions
  batch_cap_ids = coco_obj.getAnnIds(imgIds=batch_img_ids)
  cap_dicts = coco_obj.loadAnns(batch_cap_ids)
  image_inputs = torch.cat([clip.tokenize(json['caption']) for json in cap_dicts]).to(device)
  
  with torch.no_grad():
    text_embeddings = clip_model.encode_text(image_inputs).cpu().numpy()
  if(ret_img):
    return batch_img, batch_img_ids, image_embeddings, batch_cap_ids, text_embeddings

  return batch_img_ids, image_embeddings, batch_cap_ids, text_embeddings

def create_collection():
   # Define collection schema
  fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=20),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMB_DIM)
  ]
  schema = CollectionSchema(fields, COLLECTION_DESCRIPTION)

  # CREATE COLLECTION
  my_collection = Collection(COLLECTION_NAME, schema, consistency_level="Strong")

  print("collection created")
  return my_collection

def create_index(my_collection):
  # CREATE INDEX ON THE COLLECTION
  
  # TODO: Adjust parameters for Milvus indexing
  print('creating index on collection')
  index_params = {
    "index_type": "FLAT",
    "metric_type": "L2",
    "params": {},
  }
  my_collection.create_index("embedding", index_params)

def main():
  ################################ MILVUS SETUP ###############################
  # Connect to Milvus
  connections.connect("default", host="localhost", port="19530")

  # Check that the collection does not exist. If it does exist, stop executing
  # If you would like to set up the collection again, drop this collection
  exists = utility.has_collection(COLLECTION_NAME)
  my_collection = None
  if not exists:
    my_collection = create_collection()
    create_index(my_collection)
  else:
    my_collection = Collection(COLLECTION_NAME)


  ################################ SEND DATA TO MILVUS ###############################

  print("start sending data")
  print(f"using batch size {BATCH_SIZE}")

  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f'using device {device}')
  model, preprocess = clip.load(CLIP_MODEL, device)

  for split in DATA_SPLITS:
    caption_file = ANN_FILE_FMT.format(DATA_DIR, split)
    coco_caps = COCO(caption_file)

    all_img_ids_capfile = coco_caps.getImgIds() # integer list
    all_img_ids_capfile.sort()
    
    # LOOP THROUGH BATCHES OF IMAGES:
    starting_index = 0
    print(f'{len(all_img_ids_capfile)} images to encode')
    for batch in chunker(all_img_ids_capfile, BATCH_SIZE):
      start_time = time.time()
      img_ids, img_emb, cap_ids, cap_emb = embed_coco_img_batch(coco_caps, batch, split, model, preprocess, device)
      print(f'encoding time {time.time() - start_time}')
      img_entries = [
        img_ids,
        ["image" for i in range(0, len(img_ids))],
        img_emb,
      ]
      cap_entries = [
        cap_ids,
        ["caption" for i in range(0, len(cap_ids))],
        cap_emb,
      ]
      start_time = time.time()
      insert_img_ret = my_collection.insert(img_entries)
      insert_cap_ret = my_collection.insert(cap_entries)

      my_collection.flush() # apparently sends the data to disk
      print(f'insert time {time.time() - start_time}')
      print(f'sent {starting_index}-{starting_index+BATCH_SIZE-1}')
      starting_index += BATCH_SIZE



  

  
  print("done")
  # Disconnect from Milvus
  connections.disconnect("default")



if __name__ == "__main__":
    main()