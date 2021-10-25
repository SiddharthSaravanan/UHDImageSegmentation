import numpy as np
import cv2
import scipy.special as sp
import random
import math
import os

def find_extra(diffs,img,argmax_second,class_no,prob):
	not_sure = np.copy(diffs)
	perc=0
	num=0

	img2 = np.copy(argmax_second)
	img2[img2==class_no]=255
	img2[img2!=255]=0

	res = np.copy(img)

	for prob_diff in [prob]:
		if (np.count_nonzero(img))==0:
			break
		not_sure = np.copy(diffs)
		not_sure[not_sure<=prob_diff]=255
		not_sure[not_sure!=255]=0
		not_sure = not_sure.astype('uint8')

		extra_seeds = np.bitwise_and(img2,not_sure)
		res = np.bitwise_or(extra_seeds,img)

		perc = np.count_nonzero(extra_seeds)/(np.count_nonzero(res))
		break

	return (res-img)

def find_not_sure(diffs,img,argmax,class_no,prob):
	not_sure = np.copy(diffs)
	perc=0
	num=0

	for prob_diff in [prob,0]:
		if (np.count_nonzero(img))==0:
			break
		not_sure = np.copy(diffs)
		not_sure[not_sure<=prob_diff]=255
		not_sure[not_sure!=255]=0
		not_sure = not_sure.astype('uint8')

		not_sure = np.bitwise_and(not_sure,img)

		perc = np.count_nonzero(not_sure)/(np.count_nonzero(argmax==class_no))
		if perc<0.1:
			break

	return not_sure

def upscale(dataset,network,prob,upscale,ns):
	if dataset == 'BIG':
		upscale = True
	if dataset == 'pascal':
		upscale = False

	probs_folder = './datasets/'+dataset+'/segmentation_results/'
	output_folder = './upscale/upscaled_'+dataset+'/'

	for fname in os.listdir(probs_folder):
		file = os.path.join(probs_folder,fname)

		print(fname)

		if dataset == 'pascal':
			class_no = int(fname.split('_')[2])

		if dataset == 'BIG':
			obj = (((fname.split('.')[0]).split('_'))[3])

			# get object in image.
			class_no=0

			if obj=="aeroplane":
				class_no=1
			elif obj=="bicycle":
				class_no=2
			elif obj=="bird":
				class_no=3
			elif obj=="boat":
				class_no=4
			elif obj=="bottle":
				class_no=5
			elif obj=="bus":
				class_no=6
			elif obj=="car":
				class_no=7
			elif obj=="cat":
				class_no=8
			elif obj=="chair":
				class_no=9
			elif obj=="cow":
				class_no=10
			elif obj=="diningtable":
				class_no=11
			elif obj=="dog":
				class_no=12
			elif obj=="horse":
				class_no=13
			elif obj=="motorbike":
				class_no=14
			elif obj=="person":
				class_no=15
			elif obj=="pottedplant":
				class_no=16
			elif obj=="sheep":
				class_no=17
			elif obj=="sofa":
				class_no=18
			elif obj=="train":
				class_no=19
			elif obj=="tv" or obj=="tvmonitor":
				class_no=20
			else:
				class_no=0

		arr = np.load(file)

		if upscale:
			uhd_img = cv2.imread('./datasets/'+dataset+'/val_images/'+fname.split('.')[0]+'.jpg',0)
			arr = cv2.resize(arr, dsize = (uhd_img.shape[1],uhd_img.shape[0]) , interpolation=cv2.INTER_CUBIC)
		
		argmax = np.argmax(arr, axis=2)

		if ns:
			arr_max = np.max(arr, axis=2)

			arr[np.arange(arr.shape[0])[:,None],np.arange(arr.shape[1]),argmax] = 0

			argmax_second = np.argmax(arr,axis=2)
			arr_max_second = np.max(arr,axis=2)

			diffs = arr_max - arr_max_second

		img = np.copy(argmax)
		img[img==class_no]=255
		img[img!=255]=0

		if ns:
			if dataset == 'BIG':
				not_sure = find_not_sure(diffs,img,argmax,class_no,prob)
				extra = find_extra(diffs,img,argmax_second,class_no,prob)
			elif dataset == 'pascal':
				not_sure = find_not_sure(diffs,img,argmax,class_no,prob)
				extra = find_extra(diffs,img,argmax_second,class_no,prob)

			img[not_sure==255]=127
			img[extra==255]=127

		cv2.imwrite(output_folder+(fname.split('.')[0])+".png",img)
