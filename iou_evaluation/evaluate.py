import numpy as np
import cv2
import random
import math
import os


def eval_iou(dataset):
	txt_filename = "./iou_evaluation/iou_"+dataset+".txt"
	
	no=0
	file1=open(txt_filename,"w")
	file1.write('No.,Im name,TP,FP,FN,IoU')
	file1.close()

	imgs_folder = './refinement_results/'+dataset+'/'

	for fname in os.listdir(imgs_folder):
		file = os.path.join(imgs_folder,fname)

		no+=1
		file1 = open(txt_filename, "a")  # append mode
		file1.write("\n")
		file1.write(str(no))
		file1.write(",")
		file1.write(fname.split('.')[0])
		file1.write(",")
		file1.close()

		gt_path = './datasets/'+dataset+'/gt_images/'+fname.rsplit('_',1)[0]+'_gt.png'
		print(gt_path)
		img = cv2.imread(file,0)
		gt = cv2.imread(gt_path,0)

		union = cv2.bitwise_or(img,gt)
		intersection = cv2.bitwise_and(img,gt)

		tp = cv2.countNonZero(intersection)

		false=union-intersection

		fp = cv2.countNonZero(cv2.bitwise_and(false,img))
		fn = cv2.countNonZero(cv2.bitwise_and(false,gt))

		file1=open(txt_filename,"a")
		print("iou = %f\n"%((tp/(tp+fp+fn))))
		file1.write(str(tp))
		file1.write(",")
		file1.write(str(fp))
		file1.write(",")
		file1.write(str(fn))
		file1.write(",")
		file1.write(str((tp/(tp+fp+fn))))
		file1.close()