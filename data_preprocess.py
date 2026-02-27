from cv2 import cv2
import numpy as np

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, **kwargs):
        res = kwargs.copy()
        for t in self.transforms:
            res = t(**res)
        return res

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
    def __init__(self, min_height, min_width, always_apply=True, border_mode=cv2.BORDER_CONSTANT, value=0):
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

class Lambda:
    def __init__(self, image=None, mask=None):
        self.image_fn = image
        self.mask_fn = mask

    def __call__(self, **kwargs):
        if self.image_fn is not None and 'image' in kwargs:
            kwargs['image'] = self.image_fn(kwargs['image'])
        if self.mask_fn is not None and 'mask' in kwargs:
            kwargs['mask'] = self.mask_fn(kwargs['mask'])
        return kwargs

def pad_image(v):
    test_transform = [
        LongestMaxSize(v, always_apply=True),
        PadIfNeeded(v, v, always_apply=True, border_mode=cv2.BORDER_CONSTANT, value=0)
    ]
    return Compose(test_transform)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    '''
    apply imagenet pre-processing
    '''
    _transform = [
        Lambda(image=preprocessing_fn),
        Lambda(image=to_tensor, mask=to_tensor),
    ]
    return Compose(_transform)

def get_preprocessing_no_pretrain():
    _transform = [
        Lambda(image=to_tensor, mask=to_tensor),
    ]
    return Compose(_transform)