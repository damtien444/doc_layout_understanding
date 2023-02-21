import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image, ImageDraw

color_map = {4: (0, 0, 0), 5: (255, 0, 0), 6: (0, 255, 0), 21: (0, 0, 255),
             22: (255, 255, 0), 23: (0, 255, 255), 24:  (255, 0, 255), 25: (128, 128, 128),
             26: (192, 192, 192), 10:  (64, 64, 64)}
class CocoDataset(Dataset):
    def __init__(self, root_dir, annotation_file):
        self.root_dir = root_dir
        self.annotation_file = annotation_file

        self.coco = COCO(annotation_file)
        self.image_ids = list(sorted(self.coco.imgs.keys()))

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')

        annotations_ids = self.coco.getAnnIds(imgIds=img_info['id'])
        annotations = self.coco.loadAnns(annotations_ids)
        # target['boxes'] = torch.Tensor([ann['bbox'] for ann in annotations])
        # target['labels'] = torch.LongTensor([ann['category_id'] for ann in annotations])

        return image, annotations

    def draw_example(self, idx):

        image, annotations = self.__getitem__(idx)

        # Convert the image to a NumPy array and draw the annotations using cv2
        image_np = np.array(image)
        for ann in annotations:
            bbox = ann["bbox"]
            color = color_map.get(ann["category_id"])

            cv2.rectangle(image_np, (int(bbox[0]), int(bbox[1])),
                          (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                          color, thickness=1)
            label = self.coco.loadCats(ann["category_id"])[0]["name"]
            cv2.putText(image_np, label, (int(bbox[0]), int(bbox[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.25, color=(0,0,255), thickness=1)

        cv2.imshow("Image with Annotations", image_np)
        cv2.waitKey(0)

root = "/home/tiendq/Desktop/DocRec/2_data_preparation/2_selected_sample"
ann_file = "/home/tiendq/Downloads/temp_dataset/annotations/instances_default.json"
dataset = CocoDataset(root, ann_file)

print(len(dataset))
# print(dataset.draw_example(2))