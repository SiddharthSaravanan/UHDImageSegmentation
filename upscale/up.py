import numpy as np
import cv2
import scipy.special as sp
import random
import math
import os


def find_extra(diffs,img,argmax_second,class_no):
	not_sure = np.copy(diffs)
	perc=0
	num=0

	im = np.copy(img)

	img2 = np.copy(argmax_second)
	img2[img2==class_no]=255
	img2[img2!=255]=0

	res = np.copy(im)

	for prob_diff in [0.005]:
		if (np.count_nonzero(im))==0:
			break
		not_sure = np.copy(diffs)
		not_sure[not_sure<=prob_diff]=255
		not_sure[not_sure!=255]=0
		not_sure = not_sure.astype('uint8')

		extra_seeds = np.bitwise_and(img2,not_sure)
		res = np.bitwise_or(extra_seeds,im)

		perc = np.count_nonzero(extra_seeds)/(np.count_nonzero(res))
		break
		# if perc<0.5:
		# 	break

	return (res-im)

def find_not_sure(diffs,img,argmax,class_no):
	not_sure = np.copy(diffs)
	perc=0
	num=0
	im = np.copy(img)

	for prob_diff in [0.01,0]:
		if (np.count_nonzero(im))==0:
			break
		not_sure = np.copy(diffs)
		not_sure[not_sure<=prob_diff]=255
		not_sure[not_sure!=255]=0
		not_sure = not_sure.astype('uint8')

		not_sure = np.bitwise_and(not_sure,im)

		perc = np.count_nonzero(not_sure)/(np.count_nonzero(argmax==class_no))
		if perc<0.1:
			break

	im[not_sure==255]=127

	return not_sure

def upscale(dataset,network,upscale,ns):
	no=0

	if dataset == 'BIG':
		upscale = True
	if dataset == 'pascal':
		upscale = False


	probs_folder = './datasets/'+dataset+'/segmentation_probs/'
	output_folder = './upscale/upscaled_'+dataset+'/'

	for fname in os.listdir(probs_folder):
		file = os.path.join(probs_folder,fname)

		no+=1
		print(fname)
		print(no)

		if dataset == 'pascal':
			class_no = int(fname.split('_')[2])

		if dataset == 'BIG':

			obj = (((fname.split('.')[0]).split('_'))[3])

			# print(obj)
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
			elif obj=="tv":
				class_no=20
			else:
				class_no=0

		arr = np.load(file)

		# print(arr.shape)
		if upscale:
			# print('./datasets/'+dataset+'/val_images/'+fname.split('.')[0]+'.jpg')
			uhd_img = cv2.imread('./datasets/'+dataset+'/val_images/'+fname.split('.')[0]+'.jpg',0)
			arr = cv2.resize(arr, dsize = (uhd_img.shape[1],uhd_img.shape[0]) , interpolation=cv2.INTER_CUBIC)
		
		argmax = np.argmax(arr, axis=2)

		if ns:
			arr_max = np.max(arr, axis=2)
			arr_copy = np.copy(arr)

			for i in range(arr_copy.shape[0]):
				for j in range(arr_copy.shape[1]):
					arr_copy[i][j][argmax[i][j]]=0


			argmax_second = np.argmax(arr_copy,axis=2)
			arr_max_second = np.max(arr_copy,axis=2)

			diffs = arr_max - arr_max_second

		img = np.copy(argmax)
		img[img==class_no]=255
		img[img!=255]=0

		if ns:
			not_sure = find_not_sure(diffs,img,argmax,class_no)
			extra = find_extra(diffs,img,argmax_second,class_no)

			img[not_sure==255]=127
			img[extra==255]=127

		# print(output_folder+(fname.split('.')[0])+".png")

		cv2.imwrite(output_folder+(fname.split('.')[0])+".png",img)
