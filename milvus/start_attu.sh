#!/bin/bash
docker run -p 8000:3000  -e MILVUS_URL=35.215.80.6:19530 zilliz/attu:latest