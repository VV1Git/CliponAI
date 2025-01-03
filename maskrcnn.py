import os
import sys
import json
import datetime
import numpy as np
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn import model as modellib, utils
from PIL import Image, ImageDraw


class CocoLikeDataset(utils.Dataset):
    def load_data(self, annotation_json, images_dir):
        with open(annotation_json, 'r') as json_file:
            coco_json = json.load(json_file)

        source_name = "coco_like"
        for category in coco_json['categories']:
            class_id = category['id']
            class_name = category['name']
            if class_id < 1:
                print('Error: Class id for "{}" cannot be less than one. (0 is reserved for the background)'.format(
                    class_name))
                return
            self.add_class(source_name, class_id, class_name)
        # get every category id and name
        annotations = {}
        for annotation in coco_json['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations:  # if the image id is not in the annotations dictionary already add it
                annotations[image_id] = []
            annotations[image_id].append(annotation)

        seen_images = {}
        for image in coco_json['images']:
            image_id = image['id']
            seen_images[image_id] = image
            image_file_name = image['file_name']
            #input all the images into the dataset
            self.add_image(
                source=source_name,
                image_id=image['id'],
                path=os.path.abspath(os.path.join(images_dir, image_file_name)),
                width=image['width'],
                height=image['height'],
                annotations=annotations[image_id]
            )

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        annotations = image_info['annotations']
        instance_masks = []
        class_ids = []
        for annotation in annotations:
            class_id = annotation['category_id']
            mask = Image.new('1', (image_info['width'], image_info['height']))
            mask_draw = ImageDraw.ImageDraw(mask, '1')
            for segmentation in annotation['segmentation']:
                mask_draw.polygon(segmentation, fill=1)  #for each object in photo draw a polygon for it
                bool_array = np.array(mask) > 0  # make a np array
                instance_masks.append(bool_array)
                class_ids.append(class_id)
        if instance_masks:
            mask = np.dstack(instance_masks)
        else:
            mask = np.empty((image_info['height'], image_info['width'], 0), dtype=bool)
        class_ids = np.array(class_ids, dtype=np.int32)

        return mask, class_ids


dataset_train = CocoLikeDataset()
dataset_train.load_data('train_annotations.json', '')
dataset_train.prepare()

dataset_val = CocoLikeDataset()
dataset_val.load_data('test_annotations.json', '')
dataset_val.prepare()


class TrashConfig(Config):
    NAME = "test_config"
    NUM_CLASSES = 61
    STEPS_PER_EPOCH = 100
    IMAGES_PER_GPU = 3


config = TrashConfig()
config.display()

ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
print(ROOT_DIR, DEFAULT_LOGS_DIR, COCO_WEIGHTS_PATH)

model = MaskRCNN(mode='training', model_dir=DEFAULT_LOGS_DIR, config=config)
model.load_weights(COCO_WEIGHTS_PATH, by_name=True,
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
model.train(dataset_train, dataset_train, learning_rate=config.LEARNING_RATE, epochs=25, layers='heads')
