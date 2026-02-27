import numpy as np
from cv2 import cv2

RESIZE_PATCH_SIZE = 300

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, **kwargs):
        res = kwargs.copy()
        for t in self.transforms:
            res = t(**res)
        return res

class CLAHE:
    def __init__(self, clip_limit=1.0, p=0.3):
        self.clip_limit = clip_limit
        self.p = p

    def __call__(self, **kwargs):
        if np.random.rand() < self.p:
            image = kwargs['image'].copy()
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(8, 8))
            if len(image.shape) == 2:
                image = clahe.apply(image)
            elif len(image.shape) == 3:
                if image.shape[2] == 1:
                    image_sq = np.squeeze(image, axis=-1)
                    image_sq = clahe.apply(image_sq)
                    image = np.expand_dims(image_sq, axis=-1)
                else:
                    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    cl = clahe.apply(l)
                    lab = cv2.merge((cl, a, b))
                    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            kwargs['image'] = image
        return kwargs

class RandomBrightnessContrast:
    def __init__(self, brightness_limit=0.2, contrast_limit=0.1, always_apply=False, p=0.5):
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.always_apply = always_apply
        self.p = p

    def __call__(self, **kwargs):
        if self.always_apply or np.random.rand() < self.p:
            image = kwargs['image'].copy()
            alpha = 1.0 + np.random.uniform(-self.contrast_limit, self.contrast_limit)
            beta = 0.0 + np.random.uniform(-self.brightness_limit, self.brightness_limit) * 255
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            kwargs['image'] = image
        return kwargs

class ShiftScaleRotate:
    def __init__(self, scale_limit=0.25, rotate_limit=0.25, shift_limit=0.1, p=0.7, border_mode=cv2.BORDER_CONSTANT):
        self.shift_limit = shift_limit
        self.scale_limit = scale_limit
        self.rotate_limit = rotate_limit
        self.p = p
        self.border_mode = border_mode

    def __call__(self, **kwargs):
        if np.random.rand() < self.p:
            image = kwargs['image']
            h, w = image.shape[:2]
            
            angle = np.random.uniform(-self.rotate_limit, self.rotate_limit)
            scale = 1.0 + np.random.uniform(-self.scale_limit, self.scale_limit)
            dx = np.random.uniform(-self.shift_limit, self.shift_limit) * w
            dy = np.random.uniform(-self.shift_limit, self.shift_limit) * h

            center = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D(center, angle, scale)
            M[0, 2] += dx
            M[1, 2] += dy

            image_rot = cv2.warpAffine(image, M, (w, h), borderMode=self.border_mode, borderValue=0)
            if len(image.shape) == 3 and len(image_rot.shape) == 2:
                image_rot = np.expand_dims(image_rot, axis=-1)
            kwargs['image'] = image_rot
        return kwargs

class LongestMaxSize:
    def __init__(self, max_size, always_apply=True):
        self.max_size = max_size

    def __call__(self, **kwargs):
        image = kwargs['image']
        h, w = image.shape[:2]
        max_dim = max(h, w)
        if max_dim != self.max_size and max_dim > 0:
            scale = self.max_size / max_dim
            new_w, new_h = int(w * scale), int(h * scale)
            image_res = cv2.resize(image, (max(1, new_w), max(1, new_h)), interpolation=cv2.INTER_LINEAR)
            if len(image.shape) == 3 and len(image_res.shape) == 2:
                image_res = np.expand_dims(image_res, axis=-1)
            kwargs['image'] = image_res
        return kwargs

class PadIfNeeded:
    def __init__(self, min_height, min_width, always_apply=True, border_mode=cv2.BORDER_CONSTANT, mask_value=0, value=0):
        self.min_height = min_height
        self.min_width = min_width
        self.border_mode = border_mode
        self.value = value

    def __call__(self, **kwargs):
        image = kwargs['image']
        h, w = image.shape[:2]
        pad_h = max(0, self.min_height - h)
        pad_w = max(0, self.min_width - w)
        if pad_h > 0 or pad_w > 0:
            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left
            if len(image.shape) == 2:
                image = cv2.copyMakeBorder(image, top, bottom, left, right, self.border_mode, value=self.value)
            elif len(image.shape) == 3:
                if isinstance(self.value, (int, float)):
                    val = [self.value] * image.shape[2]
                else:
                    val = self.value
                image = cv2.copyMakeBorder(image, top, bottom, left, right, self.border_mode, value=val)
            kwargs['image'] = image
        return kwargs

def get_training_augmentation():
    train_transform = [
        CLAHE(clip_limit=1.0, p=0.3),
        RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.1, always_apply=False, p=0.5),
        ShiftScaleRotate(scale_limit=0.25, rotate_limit=0.25, shift_limit=0.1, p=0.7, border_mode=0),
        LongestMaxSize(RESIZE_PATCH_SIZE, always_apply=True),
        PadIfNeeded(min_height=RESIZE_PATCH_SIZE, min_width=RESIZE_PATCH_SIZE, always_apply=True, border_mode=0),
    ]
    return Compose(train_transform)

def get_validation_augmentation():
    test_transform = [
        LongestMaxSize(RESIZE_PATCH_SIZE, always_apply=True),
        PadIfNeeded(RESIZE_PATCH_SIZE, RESIZE_PATCH_SIZE, always_apply=True, border_mode=cv2.BORDER_CONSTANT, mask_value=0),
    ]
    return Compose(test_transform)
