import numpy as np
import cv2

from Main import get_UHD_segmentation
from Evaluate import iou_score
from Evaluate import overall_iou_score
from Datasets import generate_BIG_dataset
from Segmentation import map_label_to_class

param = {}
param['scale_larger_axis'] = 512.0
param['thin_iter'] = 35
param['prune_iter'] = 20
param['thresh_unsure'] = 0.03
param['beta'] = 11.30
param['eps'] = 1e-10
param['prob_tol'] = 1e-3
param['dataset'] = 'BIG'

# param['scale_larger_axis'] = 512.0
# param['thin_iter'] = 85
# param['prune_iter'] = 10
# param['thresh_unsure'] = 0.4
# param['beta'] = 10.8
# param['eps'] = 1e-10
# param['prob_tol'] = 1e-3
# param['dataset'] = 'BIG'


BIG_test, BIG_val = generate_BIG_dataset()

list_img=[]
list_gt=[]
n=0
for idx in range(BIG_val.shape[0]):

	print(idx)
	img_code_tmp, label, gt, img_fname, img_fname_gt = BIG_val[idx, 1:]
	print(img_fname)
	class_no = map_label_to_class[label]
	img_pred = get_UHD_segmentation(img_fname, class_no, **param)

	cv2.imwrite('./results/'+img_fname.split('/')[-1], img_pred )

	print("iou = ",iou_score(img_pred,cv2.imread(img_fname_gt,0)))

	list_img.append(img_pred)
	list_gt.append(cv2.imread(img_fname_gt,0))


print("overall iou = ",overall_iou_score(list_img, list_gt))
print(param)
