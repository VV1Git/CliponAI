from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
from mrcnn.utils import compute_ap
from numpy import expand_dims
from numpy import mean
from matplotlib.patches import Rectangle
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from mrcnn.visualize import display_instances, display_top_masks
from mrcnn.utils import extract_bboxes
from mrcnn.utils import Dataset
from matplotlib import pyplot as plt
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


class PredictionConfig(Config):
    NAME = "trash_cfg_coco"
    NUM_CLASSES = 69
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def evaluate_model(dataset, model, cfg):
    APs = list()
    for image_id in dataset.image_ids:
        image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
        scaled_image = mold_image(image, cfg)
        sample = expand_dims(scaled_image, 0)
        yhat = model.detect(sample, verbose=0)
        r = yhat[0]
        AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
        APs.append(AP)
    mAP = mean(APs)
    return mAP


cfg = PredictionConfig()
model = MaskRCNN(mode='inference', model_dir='logs', config=cfg)
model.load_weights('logs/mask_rcnn_trash_cfg_coco_0003.h5', by_name=True)
train_mAP = evaluate_model(dataset_train, model, cfg)
print("Train mAP: %.3f" % train_mAP)

marbles_img = skimage.io.imread("marble_dataset/val/test1.jpg")
plt.imshow(marbles_img)

detected = model.detect([marbles_img])
results = detected[0]
class_names = ['BG', 'Blue_Marble', 'Non_Blue_Marble']
display_instances(marbles_img, results['rois'], results['masks'], results['class_ids'], class_names, results['scores'])


def color_splash(img, mask):
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(img)) * 255
    if mask.shape[-1] > 0:
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, img, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    if image_path:
        img = skimage.io.imread(image_path)
        r = model.detect([img], verbose=1)[0]
        splash = color_splash(img, r['masks'])
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            success, img = vcapture.read()
            if success:
                img = img[..., ::-1]
                r = model.detect([img], verbose=0)[0]
                splash = color_splash(img, r['masks'])
                splash = splash[..., ::-1]
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


detect_and_color_splash(model, image_path="marble_dataset/val/test4.jpg")
