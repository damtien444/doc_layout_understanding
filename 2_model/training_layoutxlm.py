from PIL.Image import Image
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

os.environ['WANDB_PROJECT']="layoutxlm"


wandb.login()

processor = LayoutXLMProcessor.from_pretrained(
    "microsoft/layoutxlm-base",
    apply_ocr=False,
    only_label_first_subword=False,
    is_split_into_words=True,
    # todo
    add_special_tokens={}
)


def to_dataset():
    for i in range(len(torch_dataset)):
    # for i in range(10):
        yield torch_dataset[i]


anno_file = "0_data_repository/1000DataForOCR_fineLabel_dataset_coco.json"
image_root_folder = "0_data_repository/2_selected_sample"
torch_dataset = DocumentLayoutAnalysisDataset(image_root_folder, anno_file)
ds = Dataset.from_generator(to_dataset)
ds = ds.train_test_split(test_size=0.1, shuffle=True)

features = ds["train"].features
column_names = ds["train"].column_names
image_column_name = "image"
text_column_name = "words"
boxes_column_name = "boxes"
label_column_name = "labels_id"


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

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [torch_dataset.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [torch_dataset.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


import warnings
warnings.filterwarnings("ignore")


from transformers import Trainer, TrainingArguments
from transformers.data.data_collator import default_data_collator


checkpoint_dir="/home/tiendq/Desktop/DocRec/3_model_checkpoint/0_model_repository"
training_args = TrainingArguments(
    output_dir=checkpoint_dir,          # output directory
    num_train_epochs=15,              # total number of training epochs
    per_device_train_batch_size=4,  # batch size per device during training
    per_device_eval_batch_size=4,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,
    learning_rate=1e-5,
    evaluation_strategy="steps",
    eval_steps=500,              # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    resume_from_checkpoint=True,
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