from glob import glob

from .LUNA_mask_extraction import extract_masks
from .LUNA_segment_lung_ROI import segment_lungs
from .prepare_data_new import prepare_data
from config.config import IMAGES_PATH, DATASET_PATH


def run():
    extract_masks(glob(IMAGES_PATH+"/**/*.mhd"), DATASET_PATH)
    segment_lungs(DATASET_PATH)
    prepare_data(DATASET_PATH)
