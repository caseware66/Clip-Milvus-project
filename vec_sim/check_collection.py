from pymilvus import (
  connections,
  utility,
  Collection,
)
import sys

COLLECTION_NAME = "coco_imgcaptions"

connections.connect("default", host="localhost", port="19530")

exists = utility.has_collection(COLLECTION_NAME)
if not exists:
  sys.exit()

my_collection = Collection(COLLECTION_NAME)
print(my_collection.num_entities)

connections.disconnect("default")
