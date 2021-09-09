import numpy as np
import argparse
import os,sys

from gradients import gen_gradient
from upscale import upscale_softmax,up
import refine
from iou_evaluation import evaluate

parser = argparse.ArgumentParser(description='Refine and evaluate')
parser.add_argument('--dataset', type=str, default='BIG', choices = ['BIG','pascal','custom'], help='dataset to be used')
parser.add_argument('--network', type=str, default= 'deeplab', choices = ['deeplab','fcn'], help='network used to generate segmentation')
parser.add_argument('--output', type=str, default='./refinement_results/', help='path where refinement results are saved')
parser.add_argument('--not_sure', type=bool, default=True, help='set to True if you want to include not_sure labels when upscaling/converting network results to images')
parser.add_argument('--upscale', type=bool, default=True, help='set to True if you need to upscale images back to their original high resolution')
parser.add_argument('--x', type=int, default=28, help='Number of iterations to perform thinning')
parser.add_argument('--beta', type=int, default=111, help='Beta for Random Walker')
parser.add_argument('--prune', type=int, default=13, help='Number of iterations to perform pruning')
parser.add_argument('--evaluate', type=bool, default=True, help='set to True if you want to evaluate the results using gt_images')

args = parser.parse_args()

#------------------------------------------------

def main():

	#generate dollar gradient images
	# gen_gradient.gen_grad(args.dataset)

	#upscaling
	# upscale_softmax.gen_probs(args.dataset)
	# up.upscale(args.dataset,args.network,args.upscale,args.not_sure)

	#run random walker
	refine.refinement(args.dataset,args.x,args.prune,args.beta)

	#evaluate iou
	if args.evaluate:
		evaluate.eval_iou(args.dataset)



if __name__ == '__main__':
	main()
