from flask import Flask


from form import imgIdForm
import os
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename





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


QUERY_IMG_ID =None

CLIP_MODEL = 'ViT-L/14@336px'
DATA_DIR = 'coco'
DATA_SPLIT = 'val2017'
ANN_FILE_FMT = '{}/annotations/captions_{}.json'    # .format(DATA_DIR, split) 
IMAGE_FILE_FMT = '{}/images/{}/{}.jpg'              # .format(DATA_DIR, split, photo id zero-filled to width 12)
COLLECTION_NAME = "coco_imgcaptions_valonly"

connections.connect("default", host="localhost", port="19530")


coco_val = Collection(COLLECTION_NAME)

print("start loading collection, may take a long time")
coco_val.load()
print("load complete, begin queries")


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
#app = Flask(__name__)
def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
	return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		#print('upload_image filename: ' + filename)
		flash('Image successfully uploaded and displayed below')
		return render_template('upload.html', filename=filename)
	else:
		flash('Allowed image types are -> png, jpg, jpeg, gif')
		return redirect(request.url)
@app.route('/search', methods=['POST','GET'])
def search():
	form=imgIdForm()
	if form.validate_on_submit():
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
  		device = "cuda" if torch.cuda.is_available() else "cpu"
  		clip_model, clip_preprocess = clip.load(CLIP_MODEL, device)
  		img_filename = IMAGE_FILE_FMT.format(DATA_DIR, DATA_SPLIT, str(QUERY_IMG_ID).zfill(12))
  		with Image.open(img_filename) as img:
  			image_input = clip_preprocess(img).unsqueeze(0).to(device)
  			with torch.no_grad():
  				image_features = clip_model.encode_image(image_input).numpy()
  		image_features = image_features.flatten()
  		s_equal = np.allclose(emb, image_features, atol = 0.001)
  		print(f"Image embeddings are similar: {is_equal}")
  		search_params = {
    		"metric_type": "L2",
    		"params": {"nprobe": 10},
				}
		result = coco_val.search([image_features], "embedding", search_params, limit=10, expr='type == "caption"', output_fields=["id"])
		cap_results = []
		for hits in result:
			for hit in hits:
				print(f"hit in captions: {hit}")
				cap_results.append(hit.entity.get('id'))
		caption_file = ANN_FILE_FMT.format(DATA_DIR, DATA_SPLIT)
		coco_caps = COCO(caption_file)
		caption_jsons = coco_caps.loadAnns(cap_results)
		for json in caption_jsons:
			print(f"ID {json['id']}; Caption: {json['caption']}")
		caption_ids = coco_caps.getAnnIds(imgIds=[QUERY_IMG_ID])
		caption_jsons = coco_caps.loadAnns(caption_ids)
		for json in caption_jsons:
			flash(f"ID {json['id']}; Caption: {json['caption']}")




	return render_template('search.html',form=form)


@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)