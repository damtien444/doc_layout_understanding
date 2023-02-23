from transformers import PreTrainedTokenizerBase, LayoutLMv2FeatureExtractor, LayoutXLMTokenizer, AutoTokenizer
from transformers.file_utils import PaddingStrategy

import torch
from torch import nn

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Union


feature_extractor = LayoutLMv2FeatureExtractor(apply_ocr=False)

@dataclass
class DataCollatorForKeyValueExtraction:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None

        has_image_input = "image" in features[0]
        has_bbox_input = "bbox" in features[0]
        if has_image_input:
            image = feature_extractor([torch.tensor(feature["image"]) for feature in features], return_tensors="pt")['pixel_values']
            # image = ImageList.from_tensors([torch.tensor(feature["image"]) for feature in features], 32)
            for feature in features:
                del feature["image"]
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )

        if labels is None:
            return batch

        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch["labels"] = [label + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels]
            if has_bbox_input:
                batch["bbox"] = [bbox + [[0, 0, 0, 0]] * (sequence_length - len(bbox)) for bbox in batch["bbox"]]
        else:
            batch["labels"] = [[self.label_pad_token_id] * (sequence_length - len(label)) + label for label in labels]
            if has_bbox_input:
                batch["bbox"] = [[[0, 0, 0, 0]] * (sequence_length - len(bbox)) + bbox for bbox in batch["bbox"]]

        batch = {k: torch.tensor(v, dtype=torch.int64) if isinstance(v[0], list) else v for k, v in batch.items()}
        if has_image_input:
            batch["image"] = image
        return batch

from datasets import load_dataset

dataset = load_dataset('R0bk/XFUN', 'xfun.zh')

tokenizer = AutoTokenizer.from_pretrained('microsoft/layoutxlm-base', pad_token='<pad>')

from torch.utils.data import DataLoader
data_collator = DataCollatorForKeyValueExtraction(
    tokenizer,
    pad_to_multiple_of=8,
    padding=True,
    max_length=512,
)

train_dataset = dataset['train']
test_dataset = dataset['validation']

train_batch_size = 1
test_batch_size = 1

train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, collate_fn=data_collator, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, collate_fn=data_collator)


from torchvision.transforms import ToPILImage
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
to_img = ToPILImage()


from transformers import LayoutLMv2Model
model = LayoutLMv2Model.from_pretrained('microsoft/layoutxlm-base')

batch = next(iter(train_dataloader))

cnt = 0
for k,v in batch.items():
  print('\n', k)
  if k == 'input_ids':
    print(v.shape)
    print(v[0].max())
    # print(v[0])
    print('text is:', tokenizer.decode(v[0]))
    if cnt == 10:
        breakpoint()
    cnt +=1

import copy
batch = next(iter(test_dataloader))

inf_example = {k: v[0] for k,v in batch.items()}

inf_keep_keys = ['input_ids', 'bbox', 'hd_image', 'image', 'attention_mask']
new_inf_example = {k: copy.deepcopy(inf_example[k]) for k in inf_keep_keys}

with torch.no_grad():
    del new_inf_example['hd_image']
    for k, v in new_inf_example.items():
        if hasattr(v, "to") and hasattr(v, "device"):
            new_inf_example[k] = v.unsqueeze(0)

    # forward pass
    outputs = model(**new_inf_example)