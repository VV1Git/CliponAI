import json
import urllib.request
import os


# Load the train_annotations.json file
with open('test_annotations.json', 'r') as file:
    train_annotations = json.load(file)

"""with open('annotations.json', 'r') as file:
    annotations = json.load(file)"""

# Extract all the IDs from the annotations
"""annotation_ids = list(set([annotation['image_id'] for annotation in train_annotations['annotations']]))
n = 0
for image in annotations['images']:
    n += 1
    path = "/Users/vedant/Documents/Python/Clipon/images/" + image['file_name']
    print(image['flickr_url'], path)
    if not os.path.isdir(os.path.split(path)[0]):
        print("created a new path")
        os.mkdir(os.path.split(path)[0])
    print(n)
    urllib.request.urlretrieve(image['flickr_url'], path)"""
for k, annotation in enumerate(train_annotations['annotations']):
    train_annotations['annotations'][k] = {'id': annotation['id'], 'image_id': annotation['image_id'], 'category_id': annotation['category_id'], 'segmentation': annotation['segmentation'], 'area': annotation['area'], 'bbox': annotation['bbox'], 'iscrowd': annotation['iscrowd']}

with open('test_annotations.json', 'w') as file:
    json.dump(train_annotations, file, indent=4)