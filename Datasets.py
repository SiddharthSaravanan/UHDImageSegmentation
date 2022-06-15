"""
Module for downloading and loading the BIG dataset

Make sure you download the data from 

https://github.com/hkchengrex/CascadePSP/blob/master/docs/dataset.md

Datasets:

1 - BIG Dataset, Directory Structure is as follows:

./data
- BIG
- - test
- - - **files**
- - val
- - - **files**

2 - Relabeled PASCAL VOC 2012.

./data
- relabeled_pascal
- - **files** 
"""

import numpy as np
import os
import re


def generate_BIG_dataset():
    """
    Process the dataset to get the following output:
    [img_name1, img_name2, class, gt/im, fname, fname_gt ]

    This is further used to read and load the images.
    """
    list_test_names = os.listdir("./data/BIG/test")
    BIG_test = []
    for fname in list_test_names:
        if fname.endswith(".png") or fname.endswith(".jpg"):
            tmp = re.split('\W+|_', fname)
            if tmp[4] == 'im' or tmp[3]=='im':
                fname_gt = fname.replace('im', 'gt')
                fname_gt = fname_gt.replace('jpg', 'png')
                if tmp[3]=='im':
                    BIG_test.append([tmp[0], tmp[1], tmp[2], tmp[3], "./data/BIG/test/"+fname, "./data/BIG/test/"+fname_gt])
                else:
                    BIG_test.append([tmp[0], tmp[1], tmp[3], tmp[4], "./data/BIG/test/"+fname, "./data/BIG/test/"+fname_gt])
    BIG_test = np.array(BIG_test)

    list_val_names = os.listdir("./data/BIG/val")
    BIG_val = []
    for fname in list_val_names:
        if fname.endswith(".png") or fname.endswith(".jpg"):
            tmp = re.split('\W+|_', fname)
            if tmp[4] == 'im':
                fname_gt = fname.replace('im', 'gt')
                fname_gt = fname_gt.replace('jpg', 'png')
                BIG_val.append([tmp[0], tmp[1], tmp[3], tmp[4], "./data/BIG/val/"+fname, "./data/BIG/val/"+fname_gt])
    BIG_val = np.array(BIG_val)

    return BIG_test, BIG_val


def generate_PASCAL_dataset():
    """
    Return processed data in the form
    [year, img_name, class, gt/im, fname]
    """
    list_names = os.listdir("./data/pascal/val")
    PASCAL_data = []
    for fname in list_names:
        if fname.endswith(".png") or fname.endswith(".jpg"):
            tmp = re.split('\W+|_', fname)
            if tmp[3] == 'im':
                fname_gt = fname.replace('im', 'gt')
                fname_gt = fname_gt.replace('jpg', 'png')
                PASCAL_data.append([tmp[0], tmp[1], tmp[2], tmp[3], "./data/pascal/val/"+fname, "./data/pascal/val/"+fname_gt])
    PASCAL_data = np.array(PASCAL_data)
    return PASCAL_data
