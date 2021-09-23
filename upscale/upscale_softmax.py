import numpy as np

from scipy.special import softmax
import os

def gen_probs(dataset):

	input_folder = './datasets/'+dataset+'/segmentation_results/'
	output_folder = './datasets/'+dataset+'/segmentation_probs/'

	for filename in os.listdir(input_folder):
		file = os.path.join(input_folder,filename)
		print(file)

		arr = np.load(file)
		arr = softmax(arr,axis=2)
		np.save(output_folder+filename,arr)
