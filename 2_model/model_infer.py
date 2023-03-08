import torch
from PIL import Image
from transformers import LayoutLMv2ForTokenClassification, LayoutXLMProcessor
from data_loader_coco_image import DocumentLayoutAnalysisDataset

anno_file = "/home/tiendq/Desktop/DocRec/2_data_preparation/4_test_data/coco_annotations_v5.0.0.json"
image_root_folder = "/home/tiendq/Desktop/DocRec/2_data_preparation/4_test_data/images"
torch_dataset = DocumentLayoutAnalysisDataset(image_root_folder, anno_file, has_label=False)

model = LayoutLMv2ForTokenClassification.from_pretrained(
    '/home/tiendq/Desktop/DocRec/3_model_checkpoint/0_model_repository',
    num_labels=len(torch_dataset.label_list),
    id2label=torch_dataset.id2label,
    label2id=torch_dataset.label2id)

processor = LayoutXLMProcessor.from_pretrained(
    "microsoft/layoutxlm-base",
    apply_ocr=False,
    only_label_first_subword=False,
    is_split_into_words=True)

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
def infer(image_path, words, boxes):
    images = Image.open(image_path).convert("RGB")
    encoding = processor(images, words, boxes=boxes, truncation=True, stride=128,
                         padding="max_length", max_length=512, return_overflowing_tokens=True,
                         return_offsets_mapping=True, return_tensors='pt')

    offset_mapping = encoding.pop('offset_mapping')
    overflow_to_sample_mapping = encoding.pop('overflow_to_sample_mapping')
    with torch.no_grad():
        outputs = model(**encoding)
    logits = outputs.logits
    predictions = logits.argmax(-1).squeeze().tolist()
    token_boxes = encoding.bbox.squeeze().tolist()
    # original_words = encoding.word_ids()
    if (len(token_boxes) == 512):
        predictions = [predictions]
        token_boxes = [token_boxes]

    instance_dict = {}
    # iterate batch sample
    for i in range(0, len(token_boxes)):
        # iterate token in batch
        for j in range(0, len(token_boxes[i])):
            box = tuple(token_boxes[i][j])
            if box == (0, 0, 0, 0) or box == (1000, 1000, 1000, 1000):
                continue
            if box not in instance_dict:
                instance_dict[box] = {
                    "token_list": [(processor.decode(encoding["input_ids"][i][j]), predictions[i][j])],
                    "original_string": words[encoding.token_to_word(i, j)]}
            else:
                instance_dict[box]['token_list'].append(
                    (processor.decode(encoding["input_ids"][i][j]), predictions[i][j]))

    for box, box_info in instance_dict.items():
        box_info['box_label'] = majority_voting_label(box_info['token_list'])

    return instance_dict

