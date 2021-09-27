import json
import os
from PIL import Image
from .sub_mask import create_sub_masks
from .sub_mask_annotation import create_sub_mask_annotation
from glob import glob
from tqdm import tqdm
from config.config import IMAGES_PATH, DATASET_PATH


def annotate_dataset(test_or_train):
    def get_image_info(file, img_id):
        info = {}
        image = Image.open(file)
        info.update({
            "id": img_id,
            "file_name": str(file.split('\\')[-1]),
            "width": image.size[0],
            "height": image.size[1],
            "date_captured": "",
            "license": 1,
            "coco_url": "",
            "flickr_url": ""
        })
        return info

    test_train_path = DATASET_PATH + 'prepared_data/' + test_or_train
    working_path = test_train_path + '/masks/'
    mask_images = glob(working_path + "*.jpeg")

    # Define which colors match which categories in the test
    background, nodule = [1, 2]
    category_ids = {
        '0': background,
        '1': nodule,
    }

    is_crowd = 0

    # These ids will be automatically increased as we go
    annotation_id = 1
    image_id = 1

    # Create the annotations
    annotations = []
    images = []
    categories = [{'id': int(key)+1, "name": val, "supercategory": None} for key, val in category_ids.items()]

    for mask_image in tqdm(mask_images):
        sub_masks = create_sub_masks(Image.open(mask_image))
        images.append(get_image_info(mask_image, image_id))
        for color, sub_mask in sub_masks.items():
            category_id = category_ids.get(color, 1)
            annotation = create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id, is_crowd)
            if annotation:
                annotations.append(annotation)
                annotation_id += 1
        image_id += 1
        print(f"{image_id}/{len(mask_images)} is complete")

    return annotations, images, categories


def run():
    for t in ['train', 'test']:
        print(f"Beginning annotations for {t} set.")
        annotations, images, categories = annotate_dataset(t)
        coco_annotation = {}
        coco_annotation.update({
            "images": images,
            "annotations": annotations,
            "licenses": [
                {
                    "id": 1,
                    "name": "",
                    "url": ""
                }
            ],
            "info": {
                "description": "Nodules",
                "url": "",
                "version": "",
                "year": 2021,
                "contributor": "",
                "data_created": "2021-09-24"
            },
            "categories": categories,
        })
        if not os.path.exists(DATASET_PATH + 'prepared_data/annotations/'):
            os.makedirs(DATASET_PATH + 'prepared_data/annotations/')

        with open(DATASET_PATH + 'prepared_data/annotations/instances_' + t + '.json', 'w') as outfile:
            json.dump(coco_annotation, outfile)
        print(f"{outfile} saved.")
