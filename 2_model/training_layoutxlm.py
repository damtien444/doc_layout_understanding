import copy

from PIL import Image
from transformers import LayoutXLMProcessor
from datasets import Dataset
from data_loader_coco_image import DocumentLayoutAnalysisDataset, label_list
from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D
from transformers import LayoutLMv2ForTokenClassification
import numpy as np
import torch
from datasets import load_metric
import wandb
import os
import re

import warnings
from transformers import Trainer, TrainingArguments
from transformers.data.data_collator import default_data_collator
from datasets import concatenate_datasets

label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {idx: label for idx, label in enumerate(label_list)}

containing_indicator = [label2id[label] for label in ['title', 'answer', 'super_title']]
final_indicator = r"(^((Câu|CÂU)|(Bài)|(Ví dụ)) *?(0|[1-9][0-9]?[0-9]?|1000|(?=[MDCLXVI])M*(C[MD]|D?C*)(X[CL]|L?X*)(" \
                  r"I[XV]|V?I*))[\\.,:;]?( \\(\\d([\\.,]\\d*?)? *?(điểm|đ)\\))?[\\.,:;]?)|(^(Question|Ouestion) (0|[" \
                  r"1-9][0-9]?[0-9]?)[\\.,:;)]?)|(^([A-Z][\\.,)])|^([a-z][\\.,)])|^([1-9]+[\\.,)])|^(\\{?\\math(" \
                  r"rm|bf|bb)((\\{~?[A-Z][\\.,])~?\\}|(\\{~?[A-Z]~?\\})[\\.,])))"

# # project name send to wandb
# os.environ['WANDB_PROJECT']="layoutxlm"
# # set cuda device use for training
# os.environ['CUDA_VISIBLE_DEVICES']='1'
# # limit numpy thread for not over gaining CPU consumption
# os.environ['OMP_NUM_THREADS']='4'

os.environ['WANDB_PROJECT'] = "layoutxlm"
os.environ['WANDB_NOTEBOOK_NAME'] = "LayoutXLM for document layout analysis kaggle"
os.environ['WANDB_API_KEY'] = "660c57d70a4424e5eea4d022af08716e197d2c6a"

wandb.login()

processor = LayoutXLMProcessor.from_pretrained(
    "microsoft/layoutxlm-base",
    apply_ocr=False,
    only_label_first_subword=False,
    is_split_into_words=True,
    # todo
    # add_special_tokens={}
)


def to_dataset():
    for i in range(len(torch_dataset)):
        # for i in range(10):
        yield torch_dataset[i]


def highlight_indicator(prev_i, visited_boxes, box_id, final_indicator, overflow_to_sample_mapping,
                        offset_mapping, label2id, _labels, lock_visit):
    # todo: refactor this part to a separate function
    # find_match_and_update(visited_boxes, prev_box,)
    # tim indicator cua original_words cua prev_box

    # re.search(superquestion_expression['VN'], x)
    # nếu tồn tại indicator trong original_words, thì tìm các token tương ứng
    # sử dụng offset_mapping để map trở lại
    # https://huggingface.co/transformers/v4.2.2/custom_datasets.html#:~:text=Let%E2%80%99s%20write%20a%20function%20to%20do,%5BPAD%5D%20or%20%5BCLS%5D.

    # sau đó update _labels tuong ung
    if box_id in lock_visit:
        return _labels

    prev_original_words = visited_boxes[box_id]['original_words']

    matches = re.search(final_indicator, prev_original_words)
    if matches is None:
        return _labels

    match_span = matches.span()

    match_start_idx = match_span[0]
    match_end_idx = match_span[1]

    box_type = visited_boxes[box_id]['box_type']

    # if isinstance(match_start_idx, list):
    #     match_start_idx = match_start_idx[0]
    #     match_end_idx = match_end_idx[0]

    # tim token co range trong start and end
    batch_pos = overflow_to_sample_mapping[prev_i]
    offset_mapping_interested = offset_mapping[batch_pos][
                                visited_boxes[box_id]['start_idx']:visited_boxes[box_id]['end_idx']]
    # if prev_original_words.startswith('Câu'):
    #     print("hehe")
    for idx, token_span in enumerate(offset_mapping_interested):
        if (match_start_idx <= token_span[0]) and (token_span[1] <= match_end_idx):
            _labels[batch_pos][visited_boxes[box_id]['start_idx'] + idx] = label2id[
                'indicator_' + id2label[box_type]]
        # elif token_span[0] > match_end_idx:
        #     break

    # print(prev_original_words, _labels[batch_pos][visited_boxes[tup_prev_box]['start_idx']:visited_boxes[
    # tup_prev_box]['end_idx']], sep="\n", end="\n----------------\n")
    lock_visit.add(box_id)

    return _labels





def prepare_examples(examples):
    images = [Image.open(path).convert("RGB") for path in examples['image_path']]
    words = examples[text_column_name]
    boxes = examples[boxes_column_name]
    word_labels = examples[label_column_name]

    # encoding = processor(images, words, boxes=boxes, word_labels=word_labels, return_overflowing_tokens=True, return_offsets_mapping=True)
    encoding = processor.tokenizer(words, boxes=boxes, word_labels=word_labels, return_overflowing_tokens=True,
                                   return_offsets_mapping=True, clean_up_tokenization_spaces=True,
                                   skip_special_tokens=True)

    offset_mapping = encoding.pop('offset_mapping')
    overflow_to_sample_mapping = encoding.pop('overflow_to_sample_mapping')

    # _image = encoding['image']
    _token = encoding['input_ids']
    _words = copy.deepcopy(_token)

    for i in range(0, len(_token)):
        for j in range(0, len(_token[i])):
            _words[i][j] = processor.tokenizer.decode(_token[i][j], clean_up_tokenization_spaces=True,
                                                      skip_special_tokens=True)

    _bbox = encoding['bbox']
    _labels = encoding['labels']

    visited_boxes = {}
    lock_visit = set()

    prev_box = None
    prev_i = 0
    prev_j = 0
    prev_id = None

    # ids = []

    for i in range(0, len(_bbox)):

        for j in range(0, len(_bbox[i])):

            box = _bbox[i][j]

            # ids.append(str(i) + "_" + str(encoding.token_to_word(i, j)))
            _id = (i, encoding.token_to_word(i, j))

            if _id != prev_id and prev_id is not None:
                # tup_prev_box = tuple(prev_box)

                visited_boxes[prev_id]['end_idx'] = prev_j
                _labels = highlight_indicator(prev_i, visited_boxes, prev_id, final_indicator,
                                              overflow_to_sample_mapping, offset_mapping, label2id, _labels, lock_visit)

                # if prev_i != i:
                #     visited_boxes = {}
                #     lock_visit = set()

            if box == [0, 0, 0, 0] or box == [1000, 1000, 1000, 1000]:
                continue

            if _labels[i][j] not in containing_indicator:
                continue

            prev_box = box
            prev_id = _id
            prev_i = i
            prev_j = j

            if _id in visited_boxes:
                visited_boxes[_id]['token_list'].append(
                    (_token[i][j], _labels[i][j], i, j, processor.tokenizer.decode(_token[i][j])))
                continue

            visited_boxes[_id] = {
                'token_list': [(_token[i][j], _labels[i][j], i, j, processor.tokenizer.decode(_token[i][j]))],
                "original_words": words[overflow_to_sample_mapping[i]][encoding.token_to_word(i, j)],
                "start_idx": j,
                "box_type": _labels[i][j]
                }

    print("finish relabel!")

    encoding = processor(images, _words, boxes=_bbox, word_labels=_labels, truncation=True, stride=128,
                         padding="max_length", max_length=512, return_overflowing_tokens=True,
                         return_offsets_mapping=True)

    offset_mapping = encoding.pop('offset_mapping')
    overflow_to_sample_mapping = encoding.pop('overflow_to_sample_mapping')

    return encoding

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        ["I-" + label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        ["I-" + label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)

    no_flatten = ['overall_precision', "overall_recall", 'overall_f1', 'overall_accuracy']
    to_be_flatten = []
    for key, val in results.items():
        if key in no_flatten:
            continue
        to_be_flatten.append(key)

    for key in to_be_flatten:
        val = results[key]
        for metr, value in val.items():
            results[key + "_" + metr] = value
        del results[key]

    return results
    # return {
    #     "precision": results["overall_precision"],
    #     "recall": results["overall_recall"],
    #     "f1": results["overall_f1"],
    #     "accuracy": results["overall_accuracy"],
    # }

anno_file = "/home/tiendq/PycharmProjects/DeepLearningDocReconstruction/0_data_repository/1000DataForOCR_fineLabel_dataset_coco_v1.1_titleNsuptitle.json"
image_root_folder = "/home/tiendq/Desktop/DocRec/2_data_preparation/2_selected_sample"
# anno_file = "/kaggle/input/vietnamese-exam-doc-layout-annotations/1000DataForOCR_fineLabel_dataset_coco_v1.1_titleNsuptitle.json"
# image_root_folder = "/kaggle/input/vietnamese-exam-doc-layout/2_selected_sample"
torch_dataset = DocumentLayoutAnalysisDataset(image_root_folder, anno_file)

model = LayoutLMv2ForTokenClassification.from_pretrained(
    'microsoft/layoutxlm-base',
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id)

metric = load_metric("seqeval")

warnings.filterwarnings("ignore")

fold_num = 5

ds = Dataset.from_generator(to_dataset)
kfold = [ds.shard(fold_num, i, contiguous=True) for i in range(fold_num)]
# ds = ds.train_test_split(test_size=0.2, shuffle=True)

features = ds.features
column_names = ds.column_names
image_column_name = "image"
text_column_name = "words"
boxes_column_name = "boxes"
label_column_name = "labels_id"


# we need to define custom features for `set_format` (used later on) to work properly
features = Features({
    'image': Array3D(dtype="float32", shape=(3, 224, 224)),
    'input_ids': Sequence(feature=Value(dtype='int64')),
    'attention_mask': Sequence(Value(dtype='int64')),
    'bbox': Array2D(dtype="int64", shape=(512, 4)),
    'labels': Sequence(feature=Value(dtype='int64')),
})

for idx_fold in range(fold_num):

    fold_val = kfold[idx_fold]
    fold_train = concatenate_datasets([kfold[i] for i in range(5) if i != idx_fold])

    train_dataset = fold_train.map(
        prepare_examples,
        batched=True,
        remove_columns=column_names,
        features=features,
    )
    eval_dataset = fold_val.map(
        prepare_examples,
        batched=True,
        remove_columns=column_names,
        features=features,
    )

    train_dataset.set_format("torch")
    eval_dataset.set_format("torch")

    checkpoint_dir = "0_model_repository/1_update_titleandsupertitlemismatch"
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,  # output directory
        num_train_epochs=3,  # total number of training epochs
        per_device_train_batch_size=4,  # batch size per device during training
        per_device_eval_batch_size=4,  # batch size for evaluation
        warmup_steps=30,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,
        learning_rate=5e-5,
        evaluation_strategy="steps",
        eval_steps=500,  # strength of weight decay
        logging_dir=f'{checkpoint_dir}/logs',  # directory for storing logs
        logging_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="overall_f1",
        greater_is_better=True,
        save_total_limit=3,
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,  # evaluation dataset
    )

    trainer.train()
    trainer.save_model()
