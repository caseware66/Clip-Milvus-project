import model 
from pycocotools.coco import COCO
from flask import flash, render_template

DATA_DIR = '/home/vec_sim/coco'
DATA_SPLIT = 'train2017'
ANN_FILE_FMT = '{}/annotations/captions_{}.json'    # .format(DATA_DIR, split) 
CAPTIONS_FILE = ANN_FILE_FMT.format(DATA_DIR, DATA_SPLIT)
IMAGE_FILE_FMT = '{}.jpg'              # .format(DATA_DIR, split, photo id zero-filled to width 12)
QUERY_IMG_FILE_FMT = 'static/uploads/{}'
COLLECTION_NAME = "coco_imgcaptions"
SEARCH_PARAMS = {
			"metric_type": "L2",
			"params": {"nprobe": 10},
}

def extract_result(result):
	ids = []
	for hits in result:
		for hit in hits:
			ids.append(hit.entity.get('id'))
	return ids


def img_search(img_filename, milvus_db):

	'''
	imageid=form.imgid.data
	flash(imageid)
	QUERY_IMG_ID=imageid
	find_image_expr = "id == {}".format(QUERY_IMG_ID)
	find_result = coco_val.query(expr=find_image_expr, output_fields=["id", "type", "embedding"])
	if not find_result:
		print("entry not found")
		sys.exit()
	emb = find_result[0]['embedding']
	emb = np.array(emb)
	img_filename = IMAGE_FILE_FMT.format(DATA_DIR, DATA_SPLIT, str(QUERY_IMG_ID).zfill(12))
	image_features = model.encode_image(img_filename)
	'''
	image_features = model.encode_image(QUERY_IMG_FILE_FMT.format(img_filename))
	result_text = milvus_db.search([image_features], "embedding", SEARCH_PARAMS, limit=10, expr='type == "caption"', output_fields=["id"])
	result_img =  milvus_db.search([image_features], "embedding", SEARCH_PARAMS, limit=10, expr='type == "image"', output_fields=["id"])
	cap_results = extract_result(result_text)
	img_results = extract_result(result_img)
	
	print(cap_results)
	print(img_results)

	coco_caps = COCO(CAPTIONS_FILE)
	caption_jsons = coco_caps.loadAnns(cap_results)
	for json in caption_jsons:
		print(f"ID {json['id']}; Caption: {json['caption']}")

	img_filenames = [IMAGE_FILE_FMT.format(str(x).zfill(12)) for x in img_results]
	print(img_filenames)
	return img_filenames, caption_jsons