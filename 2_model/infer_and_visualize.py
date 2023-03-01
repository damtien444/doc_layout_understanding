from transformers import LayoutXLMProcessor, LayoutLMv2ForTokenClassification
from data_loader_coco_image import DocumentLayoutAnalysisDataset, unnormalize_bbox, color_map
import torch
from termcolor import colored


def majority_voting_label(token_list):
    # def majority_vote(l):
    vote_counts = {}
    for token in token_list:
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


# load_model and data
processor = LayoutXLMProcessor.from_pretrained(
    "microsoft/layoutxlm-base",
    apply_ocr=False,
    only_label_first_subword=False,
    is_split_into_words=True)

# anno_file = "/home/tiendq/PycharmProjects/DeepLearningDocReconstruction/0_data_repository/1000DataForOCR_fineLabel_dataset_coco.json"
# image_root_folder = "/home/tiendq/Desktop/DocRec/2_data_preparation/2_selected_sample"
# torch_dataset = DocumentLayoutAnalysisDataset(image_root_folder, anno_file)

anno_file = "/home/tiendq/Desktop/DocRec/2_data_preparation/4_test_data/coco_annotations_v5.0.0.json"
image_root_folder = "/home/tiendq/Desktop/DocRec/2_data_preparation/4_test_data/images"
torch_dataset = DocumentLayoutAnalysisDataset(image_root_folder, anno_file, has_label=False)

model = LayoutLMv2ForTokenClassification.from_pretrained(
    '/home/tiendq/Desktop/DocRec/3_model_checkpoint/0_model_repository',
    num_labels=len(torch_dataset.label_list),
    id2label=torch_dataset.id2label,
    label2id=torch_dataset.label2id)


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


from PIL import Image

image_column_name = "image"
text_column_name = "words"
boxes_column_name = "boxes"
label_column_name = "labels_id"

for i in range(len(torch_dataset)):
    examples = torch_dataset[i]

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

    x = []
    for i in range(0, len(encoding['image'])):
        x.append(encoding['image'][i])
    x = torch.stack(x)
    encoding['image'] = x

    # INFER
    import time

    start = time.time()
    with torch.no_grad():
        outputs = model(**encoding)
    infer_time = time.time()-start
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
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for bbox, token_list in instance_dict.items():
        # todo:
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

    cv2.waitKey(0)
