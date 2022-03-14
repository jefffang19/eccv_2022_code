import os
import glob

import numpy as np
from cv2 import cv2
import pandas as pd
import segmentation_models_pytorch as smp

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

from data_augmentation import get_training_augmentation, get_validation_augmentation
from data_preprocess import get_preprocessing, pad_image

RESIZE_PATCH_SIZE = 300
ENCODER = 'resnet18'
ENCODER_WEIGHTS = 'imagenet'
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

class Dataset_NIH(BaseDataset):
    """
    Args:
        path (list of str): label csvs
        train_test: "train", "valid" or "test"
        clahe (bool): do clahe or not
        do_resize (tuple of ints): resize the image and mask, if None, DO NOT do resize
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    def __init__(
            self,
            path,
            train_valid_test,
            clahe = False,
            do_resize = (512, 512),
            augmentation=None, 
            preprocessing=None,
    ):      
        y = []
    
        if train_valid_test == "train":
            y = [pd.read_csv(path[0].format(train_valid_test)), pd.read_csv(path[1].format(train_valid_test))]
            # undersample
            drop_choices = np.random.choice(y[1].index, 31963, replace=False)
            y[1] = y[1].drop(drop_choices)
        else:
            y = [pd.read_csv(path[0].format(train_valid_test)), pd.read_csv(path[1].format(train_valid_test))]
        
#         # debug
#         y = [pd.read_csv(path[0].format(train_valid_test)).head(100), pd.read_csv(path[1].format(train_valid_test)).head(100)]
        
        
        self.y = pd.concat(y)

        # check if do augmentation and preprocess
        self.augmentation = augmentation
        self.preprocessing = preprocessing

        self.do_resize = do_resize
        self.clahe = clahe
        
        # self define, in order to stacking batch
        self.IMG_SIZE = 1024
    
    def __getitem__(self, i):

        # read img
        image = cv2.imread(self.y.iloc[i]['path'], 0) # read  GRAY
        
        # read label
        label = int(self.y.iloc[i]['label'])
        
        # patch locations
        patch_loc = []
        
        # check if coord is None
        if self.y.iloc[i]["coor_0"] != "None":
            for p_loc in range(16):
                patch_loc.append(self.y.iloc[i]["coor_{}".format(p_loc)].split(' '))
        else:
            patch_loc = np.zeros((16, 4))
        
        patch_loc = np.array(patch_loc).astype(np.long)

        if self.clahe:
            # create a CLAHE object (Arguments are optional).
            clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
            image = clahe.apply(image)

        # padding image and mask
        sample = pad_image(self.IMG_SIZE)(image=image)
        image = sample['image']
        
        # patching
        patches_imgs = [image]
        
        PATCH_NUM = 17
        if patch_loc.max() != 0:
            for p in range(16):
                _patch = image[patch_loc[p][0]:patch_loc[p][1], patch_loc[p][2]:patch_loc[p][3]]
                # check if shape is invalid (any shape is 0)
                if not np.all(_patch.shape):
                    patches_imgs.append(image)
                else:  
                    patches_imgs.append(_patch)
                
        else:
            PATCH_NUM = 1
            # pad empty patches
            for p in range(16):
                patches_imgs.append(np.zeros((3, RESIZE_PATCH_SIZE, RESIZE_PATCH_SIZE), dtype=np.uint8))

        
        # apply augmentations
        if self.augmentation:
            # if patches data invalid, don't apply augmentation
            for p in range(PATCH_NUM):
                sample = self.augmentation(image=patches_imgs[p])
                patches_imgs[p] = sample['image']
                patches_imgs[p] = np.expand_dims(patches_imgs[p], axis=-1)
                
        # apply preprocessing
        if self.preprocessing:
            for p in range(PATCH_NUM):
                sample = self.preprocessing(image=patches_imgs[p])
                patches_imgs[p] = sample['image']
                
        patches_imgs = np.array(patches_imgs)
        
        '''
        return:
        patches_imgs: image and patches of images of shape (1 + n_patch, 3, p_size, p_size)
        label: int
        image: whole CXR image of shape (x, x)
        patch_loc: locations of patches of shape (n_patch, 4)  [y_start, y_end, x_start, x_end]
        '''
        return patches_imgs, label, image, patch_loc
        
    def __len__(self):
        return len(self.y)

def get_dataset_nih(nih_nodule_path, nih_normal_path, batch = 4):
    
    # set data path
    df_paths = [nih_nodule_path, nih_normal_path]
    
    # create dataset
    trainset = Dataset_NIH(df_paths, "train", augmentation=get_training_augmentation(), preprocessing=get_preprocessing(preprocessing_fn), do_resize=(RESIZE_PATCH_SIZE, RESIZE_PATCH_SIZE), clahe=False)
    validset = Dataset_NIH(df_paths, "valid", augmentation=get_validation_augmentation(), preprocessing=get_preprocessing(preprocessing_fn), do_resize=(RESIZE_PATCH_SIZE, RESIZE_PATCH_SIZE), clahe=False)
    testset = Dataset_NIH(df_paths, "test", augmentation=get_validation_augmentation(), preprocessing=get_preprocessing(preprocessing_fn), do_resize=(RESIZE_PATCH_SIZE, RESIZE_PATCH_SIZE), clahe=False)
    
    # dataloader
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch, shuffle=True, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=batch, shuffle=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch, shuffle=False, num_workers=4)

    return train_loader, valid_loader, test_loader

def renew_train_nih(nih_nodule_path, nih_normal_path, batch = 4):
    
    # set data path
    df_paths = [nih_nodule_path, nih_normal_path]
    
    # create dataset
    trainset = Dataset_NIH(df_paths, "train", augmentation=get_training_augmentation(), preprocessing=get_preprocessing(preprocessing_fn), do_resize=(RESIZE_PATCH_SIZE, RESIZE_PATCH_SIZE), clahe=False)
    
    # dataloader
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch, shuffle=True, num_workers=4)

    return train_loader