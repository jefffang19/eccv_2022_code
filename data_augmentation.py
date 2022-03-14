import numpy as np
from cv2 import cv2
import albumentations as albu

RESIZE_PATCH_SIZE = 300

def get_training_augmentation():
    train_transform = [
        albu.CLAHE(clip_limit=1.0, p=0.3),
        albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.1, always_apply=False, p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.25, rotate_limit=0.25, shift_limit=0.1, p=0.7, border_mode=0),
        albu.geometric.resize.LongestMaxSize(RESIZE_PATCH_SIZE, always_apply=True),
        albu.PadIfNeeded(min_height=RESIZE_PATCH_SIZE, min_width=RESIZE_PATCH_SIZE, always_apply=True, border_mode=0),
    ]

    return albu.Compose(train_transform)


def get_validation_augmentation():
    test_transform = [
        albu.geometric.resize.LongestMaxSize(RESIZE_PATCH_SIZE, always_apply=True),
        albu.PadIfNeeded(RESIZE_PATCH_SIZE, RESIZE_PATCH_SIZE, always_apply=True, border_mode=cv2.BORDER_CONSTANT, mask_value=0),
    ]
    return albu.Compose(test_transform)
