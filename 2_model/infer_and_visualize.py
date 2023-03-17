import json
import os
import textwrap

from transformers import LayoutXLMProcessor, LayoutLMv2ForTokenClassification
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from data_loader_coco_image import DocumentLayoutAnalysisDataset, unnormalize_bbox, color_map, label_list
import torch
from termcolor import colored
from collections import Counter

os.environ['TOKENIZERS_PARALLELISM'] = "false"

indicator_list = ['indicator_answer', 'indicator_title', 'indicator_super_title']
replacement_indicator = ['answer', 'title', 'super_title']
label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {idx: label for idx, label in enumerate(label_list)}
indicator_ids = [label2id[label] for label in indicator_list]
replacement_ids = [label2id[label] for label in replacement_indicator]
indicatorid2replacementid = {indicator_id: replacement_id for indicator_id, replacement_id in zip(indicator_ids, replacement_ids)}

def majority_voting_label(token_list, skip_indicator=True):
    # def majority_vote(l):
    vote_counts = {}
    for token in token_list:
        if token[1] in indicator_ids and skip_indicator:
            vote = indicatorid2replacementid[token[1]]
            # continue
        else:
            vote = token[1]
        if vote in vote_counts.keys():
            vote_counts[vote] += 1
        else:
            vote_counts[vote] = 1

    # winners = []
    max_count = max(vote_counts.values())
    for vote, count in vote_counts.items():
        if count == max_count:
            # winners.append(vote)

            return vote



def containment(box, point):
    # print(box)
    # print(point)
    # print((point[0] > box[0]) & (point[0] < box[2]))
    # print((point[1] > box[1]) & (point[1] < box[3]))
    if ((point[0] > box[0]) & (point[0] < box[2])) and ((point[1] > box[1]) & (point[1] < box[3])):
        return True

    return False


# instance_dict={}
def search_box(x, y, instance_dict):
    for box, list_token in instance_dict.items():
        if containment(box, (x, y)):
            return box, list_token

    return None, None

layout_lmv3 = False

pretrained_model_path = "/home/tiendq/Desktop/DocRec/3_model_checkpoint/LayoutXLM/3_w. indicator 5 fold_separate title expression" \
     if not layout_lmv3 else "/home/tiendq/Desktop/DocRec/3_model_checkpoint/LayoutLMv3/0_pilot_training_kaggle"

processor = (
    LayoutXLMProcessor if not layout_lmv3 else LayoutLMv3Processor
).from_pretrained(
    "microsoft/layoutxlm-base" if not layout_lmv3 else "microsoft/layoutlmv3-base",
    apply_ocr=False,
    only_label_first_subword=False,
    is_split_into_words=True,
)

anno_file = '/home/tiendq/Desktop/DocRec/2_data_preparation/dcu_layout_layoutxlm_output/coco_annotations_v5.0.0.json'
image_root_folder = '/home/tiendq/Desktop/DocRec/2_data_preparation/4_test_data/images'
torch_dataset = DocumentLayoutAnalysisDataset(image_root_folder, anno_file, has_label=False)

model_class = LayoutLMv2ForTokenClassification if not layout_lmv3 else LayoutLMv3ForTokenClassification
model = model_class.from_pretrained(
    pretrained_model_path,
    num_labels=len(torch_dataset.label_list),
    id2label=torch_dataset.id2label,
    label2id=torch_dataset.label2id,
)

# model = torch.compile(model)

model_root_ques_dir = "/home/tiendq/Desktop/DocRec/2_data_preparation/layoutxlm_indicator/questions"
no_model_root_ques_dir = "/home/tiendq/Desktop/DocRec/2_data_preparation/dcu_layout_no_model_output/questions"

def mouse_click(event, x, y, flags, *param):
    # to check if left mouse
    # button was clicked
    instance_dict, width, height = param[0]
    if event == cv2.EVENT_LBUTTONDOWN:
        # font for left click event
        # print(type(param))
        x = int((x / width) * 1000)
        y = int((y / height) * 1000)
        box, token_list = search_box(x, y, instance_dict)

        if box is not None:
            label = [torch_dataset.id2label[label] for token, label in token_list]
            words = [processor.tokenizer.decode([token]) for token, label in token_list]

            for k in range(len(label)):
                color = color_map[label[k]]
                print(colored(words[k], color[1]), end=" ")

            print()
            # print(x, y)
            # print(box)

            pass
        # print a line of token with corressponding colors


from PIL import Image, ImageFont, ImageDraw

image_column_name = "image"
text_column_name = "words"
boxes_column_name = "boxes"
label_column_name = "labels_id"

key = 0
k = 0
while True:
    examples = torch_dataset[k]

    images = Image.open(examples['image_path']).convert("RGB")
    words = examples[text_column_name]
    boxes = examples[boxes_column_name]
    if torch_dataset.has_label:
        word_labels = examples[label_column_name]

        # preprocess data

        encoding = processor(images, words, boxes=boxes, word_labels=word_labels, truncation=True, stride=128,
                             padding="max_length", max_length=512, return_overflowing_tokens=True,
                             return_offsets_mapping=True,
                             return_tensors='pt')

    else:
        encoding = processor(images, words, boxes=boxes, truncation=True, stride=128,
                             padding="max_length", max_length=512, return_overflowing_tokens=True,
                             return_offsets_mapping=True,
                             return_tensors='pt')

    offset_mapping = encoding.pop('offset_mapping')

    overflow_to_sample_mapping = encoding.pop('overflow_to_sample_mapping')
    if torch_dataset.has_label:
        del encoding['labels']

    if layout_lmv3:
        x = torch.stack(encoding['pixel_values'])
    else:
        x = torch.stack(encoding['image'])

    encoding['pixel_values' if layout_lmv3 else 'image'] = x



    # INFER
    import time

    start = time.time()
    with torch.no_grad():
        outputs = model(**encoding)
    infer_time = time.time() - start
    print("Infer time:", infer_time)

    # The model outputs logits of shape (batch_size, seq_len, num_labels).
    logits = outputs.logits
    # We take the highest score for each token, using argmax. This serves as the predicted label for each token.
    predictions = logits.argmax(-1).squeeze().tolist()
    token_boxes = encoding.bbox.squeeze().tolist()

    if (len(token_boxes) == 512):
        predictions = [predictions]
        token_boxes = [token_boxes]

    instance_dict = {}

    for i in range(0, len(token_boxes)):
        for j in range(0, len(token_boxes[i])):
            box = tuple(token_boxes[i][j])
            if box == (0, 0, 0, 0):
                continue
            if box not in instance_dict:
                instance_dict[box] = [(encoding["input_ids"][i][j], predictions[i][j])]
            else:
                instance_dict[box].append((encoding["input_ids"][i][j], predictions[i][j]))

    import numpy as np
    import cv2

    image = images
    width = image.width
    height = image.height

    image = np.array(image)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2RGB)
    for bbox, token_list in instance_dict.items():
        # done:
        #   xác định major label of each box
        #   parted color field area for each box
        #   return its original

        major_label_id = majority_voting_label(token_list)

        # bbox = boxes[i]
        bbox = unnormalize_bbox(bbox, width, height)
        #
        label = torch_dataset.id2label[major_label_id]
        color = color_map.get(label)[0]
        # print(label, color)
        #
        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])),
                      (int(bbox[2]), int(bbox[3])),
                      color, thickness=2)

        label = torch_dataset.id2label[major_label_id]
        cv2.putText(image, label, (int(bbox[0]), int(bbox[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.25, color=(0, 0, 255), thickness=1)

    cv2.imshow("Image with Annotations", image)
    cv2.setMouseCallback("Image with Annotations", mouse_click, (instance_dict, width, height))

    file_name = examples['image_path'].split('/')[-1] + ".json"

    try:
        with open(model_root_ques_dir + os.sep + file_name) as f:
            data = json.load(f)

        with open('model_temp.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        with open(no_model_root_ques_dir + os.sep + file_name) as f:
            data = json.load(f)

        with open('no_model_temp.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except:
        pass

    print(file_name)


    key = cv2.waitKey(0)

    if key == ord('d'):
        k += 1
    elif key == ord('a'):
        k -= 1
    elif key == 27:  # ESC key
        cv2.destroyAllWindows()
        break

    # cv2.waitKey(0)
