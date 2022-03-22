"""
    Broadly, the idea is to use the boundary detector on UHD image,
    seeds from downscaled-UHD image, combine these two using random-walker.
"""
import numpy as np
import pdb

import scipy as sp
import cv2

# from Gradient import get_dollar_gradient
from Segmentation import get_seeds_UHD
from RandomWalk import RW
from Graph import construct_graph


def get_UHD_segmentation(img_fname, class_no, **param):
    """
    Here we obtain the segmentation for the class_no within
    the image.
    """
    img_boundary = get_dollar_gradient(img_fname)

    # img_boundary = cv2.imread("./grad/"+img_fname.split('/')[4],0)
    
    img_seeds = get_seeds_UHD(img_fname, class_no, **param)

    # cv2.imwrite('./seeds/'+(img_fname.split('/')[4]).split('.')[0]+'.png', img_seeds)

    # Construct the graph from the img_boundary
    edges, weights = construct_graph(img_boundary,**param)

    # Random Walk
    pred = RW(edges, weights , img_seeds, **param)

    pred = (pred-1)*255
    pred = cv2.medianBlur(pred,7)
    _,pred = cv2.threshold(pred,200,255,cv2.THRESH_BINARY)

    return pred
