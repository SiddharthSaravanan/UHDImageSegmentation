"""
Here we compute the IoU scores between two images.
"""

import numpy as np

def iou_score(img, pred):
    """
    """
    intersection = np.sum(np.logical_and(img >= 1, pred >= 1))
    union = np.sum(np.logical_or(img >= 1, pred >= 1))
    return intersection/union


def overall_iou_score(list_img, list_pred):
    """
    """
    intersection, union = 0.0, 0.0
    for img, pred in zip(list_img, list_pred):
        intersection += np.sum(np.logical_and(img >= 1, pred >= 1))
        union += np.sum(np.logical_or(img >= 1, pred >= 1))
    return intersection/union
