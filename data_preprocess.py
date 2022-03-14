from cv2 import cv2
import segmentation_models_pytorch as smp
import albumentations as albu


def pad_image(v):
    test_transform = [
        albu.geometric.resize.LongestMaxSize(v, always_apply=True),
        albu.PadIfNeeded(v, v, always_apply=True, border_mode=cv2.BORDER_CONSTANT, value=0)
    ]
    return albu.Compose(test_transform)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    '''
    apply imagenet pre-processing
    '''
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

def get_preprocessing_no_pretrain():
    _transform = [
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)