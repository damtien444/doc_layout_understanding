import json
import os
import zipfile

import pandas as pd
from enum import Enum


def read_merge_images(dir_path):
    full_list_instances = []
    cnt = 0
    for file in os.listdir(dir_path):
        if file.startswith("ann"):
            cnt += 1
            with open(dir_path + os.sep + file) as sep_ann_file:
                parsed_json = json.load(sep_ann_file)

            full_list_instances.extend(parsed_json)

    return full_list_instances


class CategoryId(Enum):
    ILLUSTRATION = 0
    TEXT = 1
    FORMULA = 2
    QUESTION = 3
    TITLE = 4
    EXPLANATION = 5
    ANSWER = 6
    SUB_QUESTION = 7
    SUPER_QUESTION = 20
    SUPER_TITLE = 21
    HEADER = 22
    FOOTER = 23
    HEADING = 24
    ENDING = 25
    STARTING = 26


def save_to_coco(input_json, output_name: str):
    ocr_result = input_json

    categories = []
    for cat in CategoryId:
        cat_name = cat.name.lower()
        categories.append({
            'id': cat.value,
            'name': cat_name,
            'supercategory': '',
        })

    coco_anns = []
    for ann in ocr_result['annotations']:
        x0, y0, x1, y1 = ann['bbox']
        segs = ann.get('segs', None)
        coco_anns.append({
            'id': ann['id'],
            'image_id': ann['image_id'],
            'category_id': ann['category_id'],
            'bbox': [x0, y0, x1 - x0, y1 - y0],
            'area': (x1 - x0) * (y1 - y0),
            'segmentation': segs if segs is not None else [],
            'iscrowd': 0,
            'attributes': {
                'value': ann['pred_values'],
                'score': ann['score'],
                'occluded': False,
            }
        })

    with open(output_name, 'w', encoding='utf-8') as g:
        json.dump({
            'categories': categories,
            'images': ocr_result['images'],
            'annotations': coco_anns,
        }, g, indent=4, ensure_ascii=False)

    return output_name


def data_preparing(question_output_path, input_with_ocr_anno, output_name='annotated_input_coco.json'):
    print("READ OCR INPUT")
    with open(input_with_ocr_anno) as pipe_input:
        pipeline_inp = json.load(pipe_input)

    print("READ DCU OUTPUT")
    full_list_instances = read_merge_images(question_output_path)

    # parse to pandas dataframe
    output_with_label = pd.DataFrame(full_list_instances)

    # exclude not useful category
    print("FILTER OUT UNINTERESTED LABEL")
    filter_category = ~output_with_label['category_label'].isin([0, 1, 2, 3, 20])
    output_with_label_filtered = output_with_label[filter_category]

    # drop duplicated rows
    print("DROP DUPLICATED ROWS")
    drop_dup_output = output_with_label_filtered.loc[output_with_label_filtered.drop_duplicates(subset='id').index]

    # replace category_id with category_label where category_label is not none
    drop_dup_output.loc[drop_dup_output['category_label'].notna(), 'category_id'] = drop_dup_output.loc[
        drop_dup_output['category_label'].notna(), 'category_label']

    drop_dup_output['id'] = drop_dup_output['id'].astype(int)

    result_df = drop_dup_output[['id', 'image_id', 'bbox', 'category_id', 'pred_values', 'score']]

    assert drop_dup_output['id'].nunique() == len(drop_dup_output['id'])

    cwd = os.getcwd()
    pipeline_inp['annotations'] = result_df.to_dict(orient='records')
    print(f"SAVE TO DCU-format: {cwd}/dcu_{output_name}.json")
    with open(f'dcu_{output_name}.json', 'w', encoding='utf-8') as g:
        json.dump(pipeline_inp, g, indent=4, ensure_ascii=False)
    print(f"SAVE TO COCO: {cwd}/{output_name}")
    save_to_coco(pipeline_inp, output_name)

    print(f"ZIP THE FILE {output_name}")
    with zipfile.ZipFile('artifact/pack_2_CVAT.zip', 'w') as myzip:
        # write the files to the ZIP archive
        myzip.write(output_name)


if __name__ == "__main__":
    question_folder = "/home/tiendq/Desktop/DocRec/2_data_preparation/3_annotated_samples/output_1/questions"
    ocr_pipeline_input = "/home/tiendq/Desktop/DocRec/2_data_preparation/3_annotated_samples/output_1/pipeline_annos_dcu_inf_v4.0.0.json"

    data_preparing(question_folder, ocr_pipeline_input)
