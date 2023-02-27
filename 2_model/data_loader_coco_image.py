import os
import random
from enum import Enum

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image, ImageDraw

color_map = {4: (0, 0, 0), 5: (255, 0, 0), 6: (0, 255, 0), 21: (0, 0, 255),
             22: (255, 255, 0), 23: (0, 255, 255), 24: (255, 0, 255), 25: (128, 128, 128),
             26: (192, 192, 192), 10: (64, 64, 64)}

label_list = ['title', 'explanation', 'answer', 'super_title', 'header', 'footer', 'ending', 'heading', 'starting']

def normalize_bbox(bbox, width, height):
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]


def unnormalize_bbox(nbbox, width, height):
  return [
      int((nbbox[0] * width) / 1000),
      int((nbbox[1] * height) / 1000),
      int((nbbox[2] * width) / 1000),
      int((nbbox[3] * height) / 1000),
  ]


class DocumentLayoutAnalysisDataset(Dataset):
    def __init__(self, root_dir, annotation_file):
        self.root_dir = root_dir
        self.annotation_file = annotation_file

        self.coco = COCO(annotation_file)
        self.image_ids = list(sorted(self.coco.imgs.keys()))
        self.label_list = label_list
        self.label2CVATid = {self.coco.cats[_id]['name']: _id for _id in self.coco.cats.keys()}
        self.CVATid2label = {_id: self.coco.cats[_id]['name'] for _id in self.coco.cats.keys()}

        self.label2id = {label: idx for idx, label in enumerate(label_list)}
        self.id2label = {idx: label for idx, label in enumerate(label_list)}

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

        bboxes = []
        labels_id = []
        words = []
        for ann in annotations:

            coco2normalbbox = [ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3]]

            # skip instance that is other than the label list in this problem
            try:
                box = normalize_bbox(coco2normalbbox, image.width, image.height)
                label_id = self.label2id[self.CVATid2label[ann['category_id']]]
                word = ann['attributes']['value']
            except KeyError:
                continue

            bboxes.append(box)

            labels_id.append(label_id)

            words.append(word)

        # return image, annotations
        # print(type(words), words)
        # encoding = processor(
        #     np.array(image),
        #     words,
        #     boxes=torch.tensor(bboxes),
        #     word_labels=torch.tensor(labels_id),
        #     max_length=512,
        #     truncation=True,
        #     padding="max_length",
        #     # pad_to_multiple_of=8,
        #     return_tensors="pt"
        # )

        # return encoding
        # return dict(
        #     input_ids=encoding['input_ids'].flatten(),
        #     attention_mask=encoding['attention_mask'].flatten(),
        #     bbox=encoding['bbox'].flatten(end_dim=1),
        #     image=encoding['image'].flatten(end_dim=1),
        #     labels=encoding['labels'].flatten()
        # )

        # print(img_info)
        return {'words':words, 'boxes':bboxes, 'labels_id': labels_id, 'id': img_info['id'], 'width': img_info['width'], 'height': img_info['height'], 'image_path': self.root_dir + os.sep+ img_info['file_name']}

    def draw_example(self, boxes, labels_id, width, height, file_name, **kwargs):
        # Convert the image to a NumPy array and draw the annotations using cv2
        # self.root_dir
        image = Image.open(self.root_dir+os.sep+file_name).convert('RGB')
        image = np.array(image)
        for i in range(len(boxes)):
            bbox = boxes[i]
            bbox = unnormalize_bbox(bbox, width, height)

            cvatId = self.label2CVATid[self.id2label[labels_id[i]]]
            color = color_map.get(cvatId)

            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])),
                          (int(bbox[2]), int(bbox[3])),
                          color, thickness=1)
            label = self.coco.loadCats(cvatId)[0]["name"]
            cv2.putText(image, label, (int(bbox[0]), int(bbox[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.25, color=(0, 0, 255), thickness=1)

        return image
        # cv2.imshow("Image with Annotations", image_np)
        # cv2.waitKey(0)


if __name__ == "__main__":
    root = "/content/2_selected_sample"
    ann_file = "/content/1000DataForOCR_fineLabel_dataset_coco_1.json"
    torch_dataset = DocumentLayoutAnalysisDataset(root, ann_file)
# dataset[0]
# print(len(dataset))
# cv2_imshow(dataset.draw_example(**dataset[57]))
# print(dataset.draw_example(2))
