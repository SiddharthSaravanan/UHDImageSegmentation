import numpy as np
import cv2
import skimage.segmentation as seg
from skimage.segmentation import flood, flood_fill
import random
import math
import os,sys

no=0
im_path = 'D:/downsamples/'+str(max_dim)+'/im/'
gt_path = 'D:/downsamples/'+str(max_dim)+'/gt/'

for max_dim in [512.0]:
	os.makedirs(im_path)
	os.makedirs(gt_path)

	for fname in os.listdir('./Datasets/BIG/val_im/'):
		file = os.path.join('./Datasets/BIG/val_im/',fname)

		no+=1
		img = cv2.imread(file,cv2.IMREAD_COLOR)
		print(img.shape)
		print(no)
		final_h=0
		final_w=0
		h,w,_ = img.shape
		final_dim=512
		if w>h:
			scale_factor = float(w)/(final_dim)
		else:
			scale_factor = float(h)/(final_dim)

		final_w = float(w)/scale_factor
		final_h = float(h)/scale_factor
		
		scale_factor=0
		img_resized = cv2.resize(img, dsize = (int(final_w),int(final_h)) , interpolation=cv2.INTER_CUBIC)
		img_label = np.zeros(img_resized.shape)

		cv2.imwrite('./downsamples/'+str(max_dim)+'/im/'+fname.split('.')[0]+'.png',img_resized)
		cv2.imwrite('./downsamples/'+str(max_dim)+'/gt/'+fname.split('.')[0]+'.png',img_label)
