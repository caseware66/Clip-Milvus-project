from flask import Flask


from form import imgIdForm,textForm
import os
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename

import model
from img_routes import img_search
from text_routes import text2text




from pymilvus import (
	connections,
	utility,
	FieldSchema, CollectionSchema, DataType,
	Collection,
)
import numpy as np
import sys



QUERY_IMG_ID =None
DATA_DIR = '/home/vec_sim/coco'
DATA_SPLIT = 'train2017'
ANN_FILE_FMT = '{}/annotations/captions_{}.json'    # .format(DATA_DIR, split) 
CAPTIONS_FILE = ANN_FILE_FMT.format(DATA_DIR, DATA_SPLIT)
IMAGE_DIR_FMT = '{}/images/{}/'              # .format(DATA_DIR, split, photo id zero-filled to width 12)
COLLECTION_NAME = "coco_imgcaptions"
SEARCH_PARAMS = {
			"metric_type": "L2",
			"params": {"nprobe": 10}
}

connections.connect("default", host="localhost", port="19530")

coco_val = Collection(COLLECTION_NAME)

print("start loading collection, may take a long time")
coco_val.load()
print("load complete, begin queries")


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
#app = Flask(__name__)
def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/img_search')
def upload_form():
	redirect('/img_search')
	return render_template('upload.html')
@app.route('/img_search', methods=['POST'])
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
		img_filenames, captions_json = img_search(filename, coco_val)
		captions = [x['caption'] for x in captions_json]
		print(captions)
		return render_template('upload.html', filename=filename, target_imgs = img_filenames, captions=captions)
	else:
		flash('Allowed image types are -> png, jpg, jpeg, gif')
		return redirect(request.url)

@app.route('/')
def index():
	return redirect('/img_search')

@app.route('/display/<filename>',methods=['POST', 'GET'])
def display_image(filename):
	#print('display_image filename: + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/display_local/<filename>')
def display_local(filename):
    return send_from_directory(IMAGE_DIR_FMT.format(DATA_DIR, DATA_SPLIT), filename, as_attachment=True)
@app.route('/text_search',methods=['POST','GET'])
def text():
	form=textForm()
	if form.validate_on_submit():
		text=form.textfield.data
		img_filenames, captions_json = text2text(text, coco_val)
		captions = [x['caption'] for x in captions_json]
		return render_template('text.html', target_imgs = img_filenames, captions=captions,form=form)






	return render_template('text.html',form=form)