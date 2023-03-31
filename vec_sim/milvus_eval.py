from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)
import numpy as np
import torch
import sys
import string
import itertools
import time
import random

from PIL import Image
from pycocotools.coco import COCO
import clip

from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# CLIP_MODEL = 'ViT-L/14@336px'
SEED = 260
DATA_DIR = 'coco'
DATA_SPLIT = 'train2017'
ANN_FILE_FMT = '{}/annotations/captions_{}.json'    # .format(DATA_DIR, split) 
IMAGE_FILE_FMT = '{}/images/{}/{}.jpg'              # .format(DATA_DIR, split, photo id zero-filled to width 12)
COLLECTION_NAME = "coco_imgcaptions"


def jaccard_sim(list1, list2):
  strip1 = [(s.translate(str.maketrans('', '', string.punctuation))).lower() for s in list1]
  strip2 = [(s.translate(str.maketrans('', '', string.punctuation))).lower() for s in list2]
  tokens1 = [[w for w in s.split()] for s in strip1]
  tokens2 = [[w for w in s.split()] for s in strip2]
  words1 = list(itertools.chain.from_iterable(tokens1))
  words2 = list(itertools.chain.from_iterable(tokens2))
  a = set(words1)
  b = set(words2)
  c = a.intersection(b)
  return float(len(c)) / (len(a) + len(b) - len(c))
    
def cos_sim(list1, list2):
  corpus = list1 + list2
  vectorizer = CountVectorizer()
  vectorizer.fit(corpus)
  text1 = [" ".join(list1)]
  text2 = [" ".join(list2)]
  vec1 = vectorizer.transform(text1).toarray()
  vec2 = vectorizer.transform(text2).toarray()
  return cosine_similarity(vec1, vec2)[0][0]


def precision_and_recall_at_k(retrieved_items, true_items):
  # retrieved_items are of length k
  c = sum(el in retrieved_items for el in true_items)
  k = len(retrieved_items)
  precision = float(c/k)
  recall = float(c/len(true_items))
  return precision, recall

def query_image_emb_batch (milvus, img_ids):
  ### GET STORED EMBEDDINGS
  get_embeddings_expr = 'id in {} and type == "image"'.format(str(img_ids))

  start_time = time.time()
  emb_ret = milvus.query(expr=get_embeddings_expr, output_fields=["id", "type", "embedding"])
  end_time = time.time()
  print(f"query latency = {end_time-start_time:.5f}s")
  # i = 0
  # for result in emb_ret:
  #   print("i: {} id: {} type: {}".format(i, result["id"], result["type"]))
  #   i += 1

  embeddings = [result['embedding'] for result in emb_ret]
  return embeddings


def search_eval_topk(milvus, coco_caps, img_ids, embeddings, k):
  ### VECTOR SEARCH ON TEXT WITH IMAGE EMBEDDINGS
  search_params = {
      "metric_type": "L2",
      "params": {"nprobe": 10},
  }

  start_time = time.time()
  search_hits = milvus.search(embeddings, "embedding", search_params, limit=k, expr='type == "caption"', output_fields=["id"])
  end_time = time.time()
  print(f"search latency = {end_time-start_time:.5f}s")

  # 2D array of caption IDs, 1 row for each image
  search_caption_ids_agg = [[hit.entity.get('id') for hit in results_per_image] for results_per_image in search_hits]

  avg_jacc = 0.0
  avg_cos = 0.0
  avg_precision = 0.0
  avg_recall = 0.0
  for i, img_id in enumerate(img_ids):
    search_caption_ids = search_caption_ids_agg[i]
    true_caption_ids = coco_caps.getAnnIds(imgIds=img_id)
    prec, rec = precision_and_recall_at_k(search_caption_ids, true_caption_ids)

    
    true_caption_jsons = coco_caps.loadAnns(true_caption_ids)
    search_caption_jsons = coco_caps.loadAnns(search_caption_ids)
    true_captions = [json['caption'] for json in true_caption_jsons]
    search_captions = [json['caption'] for json in search_caption_jsons]

    jacc = jaccard_sim(true_captions, search_captions)
    cos = cos_sim(true_captions, search_captions)

    # print("jaccard similarity: {:.4f}".format(jacc))
    # print("cosine similarity: {:.4f}".format(cos))

    # print(img_id)
    # print("precision: {:.4f}".format(prec))
    # print("recall: {:.4f}".format(rec))

    # print("\ntrue")
    # for json in true_caption_jsons:
    #   print(f"ID {json['id']}; Caption: {json['caption']}")

    # print("\nsearch")
    # for json in search_caption_jsons:
    #   print(f"ID {json['id']}; Caption: {json['caption']}")

    avg_precision += prec
    avg_recall += rec
    avg_jacc += jacc
    avg_cos += cos
  avg_precision /= len(img_ids)
  avg_recall /= len(img_ids)
  avg_cos /= len(img_ids)
  avg_jacc /= len(img_ids)
  return avg_precision, avg_recall, avg_jacc, avg_cos



def main():
  random.seed(SEED)
  # Load all image IDs
  caption_file = ANN_FILE_FMT.format(DATA_DIR, DATA_SPLIT)
  coco_caps = COCO(caption_file)
  img_ids = coco_caps.getImgIds()

  # TODO: testing, ONLY QUERY SMALL AMT, AS 100K QUERIES MAY TAKE A LONG TIME
  # TODO: CHANGE THIS AS NECESSARY TO GET MORE RESULTS
  img_ids = random.sample(img_ids, 5000)
  # print(img_ids)


  img_ids = sorted(img_ids) # IMPORTANT! needs to be sorted, as milvus returns sorted results
  connections.connect("default", host="localhost", port="19530")


  milvus = Collection(COLLECTION_NAME)

  embeddings = query_image_emb_batch(milvus, img_ids)

  for k in [5, 7, 10]:
    prec, rec, jacc, cos = search_eval_topk(milvus, coco_caps, img_ids, embeddings, k)
    print()
    print("avg precision@{}: {:.4f}".format(k, prec))
    print("avg recall@{}: {:.4f}".format(k, rec))
    print("avg jaccard similarity: {:.4f}".format(jacc))
    print("avg cosine similarity: {:.4f}".format(cos))
    print()

  print("done")
  connections.disconnect("default")

if __name__ == "__main__":
    main()