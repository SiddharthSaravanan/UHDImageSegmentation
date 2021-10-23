import numpy as np
import cv2
import skimage.segmentation as seg
from skimage.segmentation import flood, flood_fill
import random
import math
import os,sys


# for fname in os.listdir('./Datasets/BIG/val_im/'):
# 	file = os.path.join('./Datasets/BIG/val_im/',fname)
# 	print(fname.split('.')[0])

# sys.exit()
no=0
# smallest=10000000000000000
# large_axis = 0

# for fname in os.listdir('./Datasets/BIG/test_im/'):
# 	file = os.path.join('./Datasets/BIG/test_im/',fname)

# 	img = cv2.imread(file,cv2.IMREAD_COLOR)
# 	print(img.shape)

# 	if img.shape[0]>img.shape[1]:
# 		large_axis = img.shape[0]
# 	else:
# 		large_axis = img.shape[1]

# 	if large_axis<smallest:
# 		smallest = large_axis

# print(smallest)
# sys.exit(0)
# for max_dim in [30.0,90.0,150.0,210.0,270.0,330.0,390.0,450.0]:
for max_dim in [512.0]:
	os.makedirs('D:/UG/IP SOP/downsamples/'+str(max_dim)+'/im/')
	os.makedirs('D:/UG/IP SOP/downsamples/'+str(max_dim)+'/gt/')

	for fname in os.listdir('./Datasets/BIG/val_im/'):
		file = os.path.join('./Datasets/BIG/val_im/',fname)

		# file1=open("test.txt","a")
		# file1.write(fname.split('.')[0]+'\n')
		# file1.close()

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
		# print(h)
		# print(w)
		# max_dim=512
		
		scale_factor=0
		img_resized = cv2.resize(img, dsize = (int(final_w),int(final_h)) , interpolation=cv2.INTER_CUBIC)


		img_label = np.zeros(img_resized.shape)
		# print('./downsamples/'+str(max_dim)+'/im/'+fname.split('.')[0]+'.png')
		cv2.imwrite('./downsamples/'+str(max_dim)+'/im/'+fname.split('.')[0]+'.png',img_resized)
		cv2.imwrite('./downsamples/'+str(max_dim)+'/gt/'+fname.split('.')[0]+'.png',img_label)

