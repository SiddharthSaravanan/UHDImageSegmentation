
import numpy as np
import cv2
import scipy.special as sp
import skimage.segmentation as seg
from skimage.segmentation import flood, flood_fill
import random
import math
import os

folder = './segmentation_results/'
for fname in os.listdir(folder):
	file = os.path.join(folder,fname)
	
	print(file)
	arr = np.load(file)
	l = file.split('\'')
	f = l[1]
	print(f)

	np.save('./lig/'+f+'.npy',arr)
