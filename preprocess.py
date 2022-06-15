import numpy as np
import cv2
import os
import argparse


parser = argparse.ArgumentParser(description='Refine and evaluate')
parser.add_argument('--dataset', type=str, default='BIG', choices = ['BIG','pascal'], help='dataset to be used')
parser.add_argument('--network', type=str, default= 'deeplab', choices = ['deeplab','fcn'], help='network used to generate segmentation')

args = parser.parse_args()

dir_name = ''


if args.dataset == 'BIG':
	dir_name = './data/BIG/val/'
elif args.dataset == 'pascal':
	dir_name = './data/pascal/val/'


file1=open("./models/research/deeplab/datasets/pascal_voc_seg/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt","w")
file2=open("./models/research/deeplab/datasets/pascal_voc_seg/VOCdevkit/VOC2012/ImageSets/Segmentation/trainval.txt","w")
file3=open("./models/research/deeplab/datasets/pascal_voc_seg/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt","w")

file1.close()
file2.close()
file3.close()

file1=open("./models/research/deeplab/datasets/pascal_voc_seg/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt","a")
file2=open("./models/research/deeplab/datasets/pascal_voc_seg/VOCdevkit/VOC2012/ImageSets/Segmentation/trainval.txt","a")
file3=open("./models/research/deeplab/datasets/pascal_voc_seg/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt","a")

no=0

for f in os.listdir('./models/research/deeplab/datasets/im/'):
    os.remove(os.path.join('./models/research/deeplab/datasets/im/', f))
for f in os.listdir('./models/research/deeplab/datasets/gt/'):
    os.remove(os.path.join('./models/research/deeplab/datasets/gt/', f))
for f in os.listdir('./fcn/demo/'):
    os.remove(os.path.join('./fcn/demo/', f))
for f in os.listdir('./initial_segmentation_results/'):
    os.remove(os.path.join('./initial_segmentation_results/', f))
for f in os.listdir('./results/'):
    os.remove(os.path.join('./results/', f))

if args.network == 'deeplab': 
	for fname in os.listdir(dir_name):
		file = os.path.join(dir_name,fname)

		if args.dataset == 'BIG':
			if fname[-5] == 't':
				continue

			no+=1
			print('processing image ',no)

			file1.write(fname.split('.')[0])
			file1.write('\n')
			file2.write(fname.split('.')[0])
			file2.write('\n')
			file3.write(fname.split('.')[0])
			file3.write('\n')
			img = cv2.imread(file,cv2.IMREAD_COLOR)

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

			cv2.imwrite('./models/research/deeplab/datasets/im/'+fname.split('.')[0]+'.png',img_resized)
			cv2.imwrite('./models/research/deeplab/datasets/gt/'+fname.split('.')[0]+'.png',img_label)

		elif args.dataset == 'pascal':
			if fname[-5] == 't':
				continue
			no+=1
			print('processing image ',no)

			file1.write(fname.split('.')[0])
			file1.write('\n')
			file2.write(fname.split('.')[0])
			file2.write('\n')
			file3.write(fname.split('.')[0])
			file3.write('\n')
			img = cv2.imread(file,cv2.IMREAD_COLOR)

			img_label = np.zeros(img.shape)

			cv2.imwrite('./models/research/deeplab/datasets/im/'+fname,img)
			cv2.imwrite('./models/research/deeplab/datasets/gt/'+fname,img_label)


elif args.network == 'fcn':
	for fname in os.listdir(dir_name):
		file = os.path.join(dir_name,fname)

		if args.dataset == 'BIG':
			if fname[-5] == 't':
				continue

			no+=1
			print('processing image ',no)

			img = cv2.imread(file,cv2.IMREAD_COLOR)

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

			cv2.imwrite('./fcn/demo/'+fname.split('.')[0]+'.png',img_resized)

		elif args.dataset == 'pascal':
			if fname[-5] == 't':
				continue
			no+=1
			print('processing image ',no)

			img = cv2.imread(file,cv2.IMREAD_COLOR)

			cv2.imwrite('./fcn/demo/'+fname,img)



file1.close()
file2.close()
file3.close()