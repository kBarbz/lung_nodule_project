import glob
import cv2
import numpy as np
import os

try:
    from tqdm import tqdm  # long waits are not fun
except:
    print('TQDM does make much nicer wait bars...')
    tqdm = lambda x: x


def prepare_data(dataset_path):
    train_img = np.load(dataset_path + "processed_images/trainImages.npy")
    train_mask = np.load(dataset_path + "processed_images/trainMasks.npy")

    PREPARED_DATA = dataset_path + "prepared_data/"
    TRAIN_FILE = PREPARED_DATA + 'train/'
    TEST_FILE = PREPARED_DATA + 'test/'
    TRAIN_IMAGES = TRAIN_FILE + 'images/'
    TRAIN_MASKS = TRAIN_FILE + 'masks/'
    TEST_IMAGES = TEST_FILE + 'images/'
    TEST_MASKS = TEST_FILE + 'masks/'
    if not os.path.exists(PREPARED_DATA):
        os.mkdir(PREPARED_DATA)
    if not os.path.exists(TRAIN_FILE):
        os.mkdir(TRAIN_FILE)
    if not os.path.exists(TEST_FILE):
        os.mkdir(TEST_FILE)
    if not os.path.exists(TRAIN_IMAGES):
        os.mkdir(TRAIN_IMAGES)
    if not os.path.exists(TRAIN_MASKS):
        os.mkdir(TRAIN_MASKS)
    if not os.path.exists(TEST_IMAGES):
        os.mkdir(TEST_IMAGES)
    if not os.path.exists(TEST_MASKS):
        os.mkdir(TEST_MASKS)

    count = 0
    for img, mask in tqdm(zip(train_img, train_mask)):
        img = img.reshape(512, 512)
        img = cv2.normalize(img, img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        mask = mask.reshape(512, 512)
        mask = cv2.normalize(mask, mask, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        mask[mask > 0] = 1

        cv2.imwrite(TRAIN_IMAGES + str(count) + ".jpeg", img)
        cv2.imwrite(TRAIN_MASKS + str(count) + ".jpeg", mask)
        count = count + 1

    test_img = np.load(dataset_path + "processed_images/testImages.npy")
    test_mask = np.load(dataset_path + "processed_images/testMasks.npy")

    count = 0
    for img, mask in tqdm(zip(test_img, test_mask)):
        img = img.reshape(512, 512)
        img = cv2.normalize(img, img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        mask = mask.reshape(512, 512)
        mask = cv2.normalize(mask, mask, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        mask[mask > 0] = 1

        cv2.imwrite(TEST_IMAGES + str(count) + ".jpeg", img)
        cv2.imwrite(TEST_MASKS + str(count) + ".jpeg", mask)
        count = count + 1
