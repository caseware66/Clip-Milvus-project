from pymilvus import (
  connections, utility
)

COLLECTION_NAME = "TEST"

connections.connect("default", host="localhost", port="19530")

exists = utility.has_collection(COLLECTION_NAME)
if exists:
  print("removed")
  utility.drop_collection(COLLECTION_NAME)

connections.disconnect("default")