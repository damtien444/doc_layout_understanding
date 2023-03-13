from PIL import Image
from transformers import LayoutXLMProcessor
from datasets import Dataset
from data_loader_coco_image import DocumentLayoutAnalysisDataset
from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D
from transformers import LayoutLMv2ForTokenClassification
import numpy as np
import torch
from datasets import load_metric
import wandb
import os
import re

label_list = ['title', 'explanation', 'answer', 'super_title', 'header', 'footer', 'ending', 'heading', 'starting', 'indicator']

label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {idx: label for idx, label in enumerate(label_list)}

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

os.environ['WANDB_PROJECT']="layoutxlm"
os.environ['WANDB_NOTEBOOK_NAME']="LayoutXLM for document layout analysis kaggle"
os.environ['WANDB_API_KEY']="660c57d70a4424e5eea4d022af08716e197d2c6a"

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


# anno_file = "0_data_repository/v1.1_title_and_supertitle_mis_define/instances_default.json"
# image_root_folder = "0_data_repository/2_selected_sample"
anno_file = "/kaggle/input/vietnamese-exam-doc-layout-annotations/1000DataForOCR_fineLabel_dataset_coco_v1.1_titleNsuptitle.json"
image_root_folder = "/kaggle/input/vietnamese-exam-doc-layout/2_selected_sample"
torch_dataset = DocumentLayoutAnalysisDataset(image_root_folder, anno_file)
ds = Dataset.from_generator(to_dataset)
ds = ds.train_test_split(test_size=0.2, shuffle=True)

features = ds["train"].features
column_names = ds["train"].column_names
image_column_name = "image"
text_column_name = "words"
boxes_column_name = "boxes"
label_column_name = "labels_id"


def highlight_indicator(prev_i, visited_boxes, tup_prev_box, final_indicator, overflow_to_sample_mapping,
                        offset_mapping, label2id, _labels):
    # done: refactor this part to a separate function
    # find_match_and_update(visited_boxes, prev_box,)
    # tim indicator cua original_words cua prev_box

    # re.search(superquestion_expression['VN'], x)
    # nếu tồn tại indicator trong original_words, thì tìm các token tương ứng
    # sử dụng offset_mapping để map trở lại
    # https://huggingface.co/transformers/v4.2.2/custom_datasets.html#:~:text=Let%E2%80%99s%20write%20a%20function%20to%20do,%5BPAD%5D%20or%20%5BCLS%5D.

    # sau đó update _labels tuong ung
    prev_original_words = visited_boxes[tup_prev_box]['original_words']

    matches = re.search(final_indicator, prev_original_words)
    if matches is None:
        return _labels

    match_start_idx = matches.start(0)
    match_end_idx = matches.end(0)

    if isinstance(match_start_idx, list):
        match_start_idx = match_start_idx[0]
        match_end_idx = match_end_idx[0]

    # tim token co range trong start and end
    batch_pos = overflow_to_sample_mapping[prev_i]
    offset_mapping_interested = offset_mapping[batch_pos][
                                visited_boxes[tup_prev_box]['start_idx']:visited_boxes[tup_prev_box]['end_idx']]

    for idx, token_span in enumerate(offset_mapping_interested):
        if match_start_idx <= token_span[0] and token_span[1] <= match_end_idx:
            _labels[batch_pos][visited_boxes[tup_prev_box]['start_idx'] + idx] = label2id['indicator']
        elif token_span[0] > match_end_idx:
            break

    return _labels


containing_indicator = [torch_dataset.label2id[label] for label in ['heading', 'title', 'answer', 'super_title']]


def prepare_examples(examples):
    images = [Image.open(path).convert("RGB") for path in examples['image_path']]
    words = examples[text_column_name]
    boxes = examples[boxes_column_name]
    word_labels = examples[label_column_name]

    encoding = processor(images, words, boxes=boxes, word_labels=word_labels, truncation=True, stride=128,
                         padding="max_length", max_length=512, return_overflowing_tokens=True,
                         return_offsets_mapping=True)

    offset_mapping = encoding.pop('offset_mapping')
    overflow_to_sample_mapping = encoding.pop('overflow_to_sample_mapping')

    # _image = encoding['image']
    _token = encoding['input_ids']
    _bbox = encoding['bbox']
    _labels = encoding['labels']

    visited_boxes = {}

    prev_box = None
    prev_i = 0
    prev_j = 0

    for i in range(0, len(_bbox)):
        for j in range(0, len(_bbox[i])):
            # for idx, box in enumerate(boxes):
            box = _bbox[i][j]
            if box == [0, 0, 0, 0] or box == [1000, 1000, 1000, 1000]:
                # print(True)
                continue

            if _labels[i][j] not in containing_indicator:
                continue

            if box != prev_box and prev_box is not None:
                tup_prev_box = tuple(prev_box)
                visited_boxes[tup_prev_box]['end_idx'] = prev_j
                _labels = highlight_indicator(prev_i, visited_boxes, tup_prev_box, final_indicator,
                                              overflow_to_sample_mapping, offset_mapping, label2id, _labels)

            prev_box = box
            prev_i = i
            prev_j = j

            if tuple(box) in visited_boxes:
                visited_boxes[tuple(box)]['token_list'].append((_token[i][j], _labels[i][j], i, j))
                continue

            visited_boxes[tuple(box)] = {'token_list': [(_token[i][j], _labels[i][j], i, j)],
                                         "original_words": words[overflow_to_sample_mapping[i]][
                                             encoding.token_to_word(i, j)],
                                         "start_idx": j
                                         }

    return encoding


# we need to define custom features for `set_format` (used later on) to work properly
features = Features({
    'image': Array3D(dtype="float32", shape=(3, 224, 224)),
    'input_ids': Sequence(feature=Value(dtype='int64')),
    'attention_mask': Sequence(Value(dtype='int64')),
    'bbox': Array2D(dtype="int64", shape=(512, 4)),
    'labels': Sequence(feature=Value(dtype='int64')),
})

train_dataset = ds["train"].map(
    prepare_examples,
    batched=True,
    remove_columns=column_names,
    features=features,
)
eval_dataset = ds["test"].map(
    prepare_examples,
    batched=True,
    remove_columns=column_names,
    features=features,
)


train_dataset.set_format("torch")
eval_dataset.set_format("torch")


model = LayoutLMv2ForTokenClassification.from_pretrained(
    'microsoft/layoutxlm-base',
    num_labels=len(torch_dataset.label_list),
    id2label=torch_dataset.id2label,
    label2id=torch_dataset.label2id)


metric = load_metric("seqeval")


def compute_metrics(p, _torch_dataset=torch_dataset):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        ["I-" + _torch_dataset.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        ["I-" + _torch_dataset.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
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


import warnings
warnings.filterwarnings("ignore")


from transformers import Trainer, TrainingArguments
from transformers.data.data_collator import default_data_collator


checkpoint_dir="0_model_repository/1_update_titleandsupertitlemismatch"
training_args = TrainingArguments(
    output_dir=checkpoint_dir,          # output directory
    num_train_epochs=30,              # total number of training epochs
    per_device_train_batch_size=4,  # batch size per device during training
    per_device_eval_batch_size=4,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    # weight_decay=0.01,
    learning_rate=3e-5,
    evaluation_strategy="steps",
    eval_steps=500,              # strength of weight decay
    logging_dir=f'{checkpoint_dir}/logs',            # directory for storing logs
    logging_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="overall_f1",
    resume_from_checkpoint=False,
    greater_is_better=True,
    save_total_limit=3,
    
)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=eval_dataset,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,             # evaluation dataset

)

trainer.train()
trainer.save_model()