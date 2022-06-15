import numpy as np
import cv2

from Main import get_UHD_segmentation
from Evaluate import iou_score
from Evaluate import overall_iou_score
from Datasets import generate_PASCAL_dataset
from Segmentation import map_label_to_class

param = {}
# param['scale_larger_axis'] = 512.0
# param['thin_iter'] = 0
# param['prune_iter'] = 0
# param['thresh_unsure'] = 0.40
# param['beta'] = 10.50
# param['eps'] = 1e-10
# param['prob_tol'] = 1e-3
# param['dataset'] = 'pascal'

param['scale_larger_axis'] = 512.0
param['thin_iter'] = 0
param['prune_iter'] = 0
param['thresh_unsure'] = 0.95
param['beta'] = 10.90
param['eps'] = 1e-10
param['prob_tol'] = 1e-3
param['dataset'] = 'pascal'


PASCAL_val = generate_PASCAL_dataset()

list_img=[]
list_gt=[]
print(param)

for idx in range(PASCAL_val.shape[0]):
	print(idx)

	_, class_no, gt, img_fname, img_fname_gt = PASCAL_val[idx, 1:]

	img_pred = get_UHD_segmentation(img_fname, class_no, **param)

	cv2.imwrite('./results/'+img_fname.split('/')[-1], img_pred )

	print("iou = ",iou_score(img_pred,cv2.imread(img_fname_gt,0)))

	list_img.append(img_pred)
	list_gt.append(cv2.imread(img_fname_gt,0))


print("overall iou = ",overall_iou_score(list_img, list_gt))
print(param)