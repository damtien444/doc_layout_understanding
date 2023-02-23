from enum import Enum

from transformers import LayoutLMv2ForTokenClassification, LayoutXLMProcessor
from data_loader_coco_image import CocoDataset
import copy
import torch
import numpy as np




root = "/home/tiendq/Desktop/DocRec/2_data_preparation/2_selected_sample"
ann_file = "/home/tiendq/PycharmProjects/DeepLearningDocReconstruction/1_data_preparation/artifact/1000DataForOCR_fineLabel_dataset_coco.json"
dataset = CocoDataset(root, ann_file)

model = LayoutLMv2ForTokenClassification.from_pretrained('microsoft/layoutxlm-base', num_labels=len(dataset.label_list), id2label=dataset.id2label, label2id=dataset.label2id)

processor = LayoutXLMProcessor.from_pretrained("microsoft/layoutxlm-base", apply_ocr=False, only_label_first_subword=False, is_split_into_words=True)

image, words, boxes, word_labels = dataset[0]

exp_encoding = processor(
    np.array(image),
    words,
    boxes=torch.tensor(boxes),
    word_labels=torch.tensor(word_labels),
    max_length=512,
    truncation=True,
    padding=True,
    pad_to_multiple_of=8,
    return_tensors="pt")


tokens = processor.tokenizer.convert_ids_to_tokens(exp_encoding["input_ids"][0])

inf_example = {k: v[0] for k,v in exp_encoding.items()}

# inf_example['input_ids'] = torch.reshape(inf_example['input_ids'], (1,512))
# inf_example['attention_mask'] = torch.reshape(inf_example['attention_mask'], (1,512))
# inf_example['labels'] = torch.reshape(inf_example['labels'], (1,512))


inf_keep_keys = ['input_ids', 'bbox', 'image', 'attention_mask']
new_inf_example = {k: copy.deepcopy(inf_example[k]) for k in inf_keep_keys}
new_inf_example.keys()

debug = False
device = 'cpu'

example_encoded = exp_encoding

print(f'Fields of Encodeed Input Data:')
for k, v in example_encoded.items():
    print(f'\t{k}\t {v.shape}')

if debug:
    list_of_token = [processor.tokenizer.decode(example_encoded['input_ids'][0][i]) for i in range(len(example_encoded['input_ids'][0]))]
    for i in range(100):
        print(list_of_token[i])
        print(example_encoded['bbox'][0][i])
        print(example_encoded['labels'][0][i])
        print("------")
    bbox = example_encoded['bbox'][0].unsqueeze(0)
    print(bbox[:,:, 3] - bbox[:,:, 1])

with torch.no_grad():
    # del exp_encoding['labels']
    for k, v in new_inf_example.items():
        # if hasattr(v, "to") and hasattr(v, "device"):
        new_inf_example[k] = v.unsqueeze(0)

    # forward pass
    outputs = model(**new_inf_example)



# evaluation
predictions, labels = outputs.logits, inf_example['labels'].reshape(1,512)
predictions = np.argmax(predictions, axis=2)

import evaluate
seqeval = evaluate.load("seqeval")

true_predictions = [
        [dataset.label_list[p.item()] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

true_labels = [
        [dataset.label_list[l.item()] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

results = seqeval.compute(predictions=true_predictions, references=true_labels)
re = {
    "precision": results["overall_precision"],
    "recall": results["overall_recall"],
    "f1": results["overall_f1"],
    "accuracy": results["overall_accuracy"],
}

print("inference completed")