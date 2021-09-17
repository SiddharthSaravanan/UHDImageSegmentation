import numpy as np
import cv2
import skimage.segmentation as seg
from skimage.segmentation import flood, flood_fill
import random
import math
import os,sys

folder = './results/'
no=0

for fname in os.listdir(folder):
	file = os.path.join(folder,fname)

	txt_filename = "./exp1,neg/"+fname.split('.')[0]+".txt"

	file1=open(txt_filename,"a")
	file1.write('distance,positives,true positives,percentage')
	file1.close()

	print(fname)
	print(no)

	img = cv2.imread(file,0)
	# img = 255-img
	arr = cv2.distanceTransform(src=img,distanceType = cv2.DIST_L2, maskSize = cv2.DIST_MASK_PRECISE)
	i=1
	arr = arr.astype('int')
	gt = cv2.imread("./Datasets/BIG/val_gt/"+fname.split('.')[0].rsplit('_',1)[0]+"_gt.png",0)
	while i<=np.max(arr):
		t = np.zeros(arr.shape,dtype='uint8')
		t[arr==i]=255
		pos = np.count_nonzero(t)
		tp = np.count_nonzero(cv2.bitwise_and(gt,t))
		if pos==0:
			file1=open(txt_filename,"a")
			file1.write('\n')
			file1.write(str(i))
			file1.write(',')
			file1.write(str(pos))
			file1.write(',')
			file1.write(str(tp))
			file1.write(',')
			file1.write("undef")
			file1.close()
			del t
			i+=1
			continue

		perc = float(tp)/float(pos)

		file1=open(txt_filename,"a")
		file1.write('\n')
		file1.write(str(i))
		file1.write(',')
		file1.write(str(pos))
		file1.write(',')
		file1.write(str(tp))
		file1.write(',')
		file1.write(str(perc))
		file1.close()

		if i>=5000:
			break

		cv2.imwrite("./dist/"+str(i)+".png",t)

		del t
		i+=1
