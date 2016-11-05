import numpy as np
from matplotlib import pyplot as plt
import json
from scipy import misc
import os
import sys
from matplotlib.patches import Polygon
import sys
sys.path.append('/home/drew/Documents/coco_images/coco/PythonAPI/pycocotools')
from tqdm import tqdm

home_dir = '/home/drew/Documents/coco_images'
annotation_dir = home_dir + '/annotations'
im_dir = home_dir + '/train2014'
json_file = annotation_dir + '/instances_train2014.json'
output_images = home_dir + '/processed_output'

if not os.path.exists(output_images):
    os.makedirs(output_images)

def get_ids(data):
	ids = []
	for i in data:
		ids.append(i['id'])
	return ids

with open(json_file,'rb') as jsonfile:
	data = json.load(jsonfile)


categories = data['categories']
annotations = data['annotations']
images = data['images']

category_ids = get_ids(categories)
annotation_ids = get_ids(annotations)
image_ids = get_ids(images)

training_image_category_id = []
training_content_category_id = []
training_image_category_name = []
training_content_category_name = []
training_image_map = []

for i in tqdm(range(0,len(annotations))): #do everything relative to the annotation number
	annotation_number = i
	annotation_image_id = annotations[annotation_number]['image_id']
	annotation_seg = annotations[annotation_number]['segmentation']
	annotation_ii_to_image_ii = image_ids.index(annotation_image_id)
	im_id = images[annotation_ii_to_image_ii]['id']
	im_h = images[annotation_ii_to_image_ii]['height']
	im_w = images[annotation_ii_to_image_ii]['width']
	im_name = im_dir + '/' + images[annotation_ii_to_image_ii]['file_name']
	im = misc.imread(im_name)
	im_category_id = annotations[annotation_number]['category_id']
	annotation_ii_to_category_ii = category_ids.index(im_category_id)
	im_categories = categories[annotation_ii_to_category_ii]
	misc.imsave(output_images + '/image_' + str(i) + '_' + str(im_categories['id']) + '.jpg',im)
        training_image_map.append('/image_' + str(i) + '_999.jpg')
	training_image_category_id.append(im_category_id)
	training_image_category_name.append((im_categories['supercategory']))
	temp_content_ids = []
	temp_content_names = []
	num_annotations = len(annotation_seg)
	if num_annotations > 0:
		for s in range(0,num_annotations):
	#		poly = np.array(s).reshape((len(s)/2, 2))
	#		polygons.append(Polygon(poly))
	#                if type(s['counts']) == list:
	#                else:
	#                    rle = [s]
	#		if len(im.shape) == 2:
	#			im = im[:,:,None]
	#			im = np.tile(im,[1,1,3])
	#		im = misc.imresize(im,[200,200])
	#		misc.imsave(output_images + '/image_' + str(training_image_category_id[0]) + '_' + str(i) + '.jpg',im)
			temp_content_ids.append(im_categories['id'])
		training_content_category_id.append(temp_content_ids)
np.savez(home_dir + '/coco_full_im_processed_labels',training_image_category_id=training_image_category_id,training_content_category_id=training_content_category_id,training_image_category_name=training_image_category_name,training_image_map=training_image_map)
