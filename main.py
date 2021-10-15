import numpy as np
import argparse
import os,sys
import matplotlib.pyplot as plt

from gradients import gen_gradient
from upscale import upscale_softmax,up
import refine
from iou_evaluation import evaluate

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



parser = argparse.ArgumentParser(description='Refine and evaluate')
parser.add_argument('--dataset', type=str, default='BIG', choices = ['BIG','pascal','custom'], help='dataset to be used')
parser.add_argument('--network', type=str, default= 'deeplab', choices = ['deeplab','fcn'], help='network used to generate segmentation')
parser.add_argument('--not_sure', type=str2bool, default=True, help='set to True if you want to include not_sure labels when upscaling/converting network results to images')
parser.add_argument('--upscale', type=str2bool, default=True, help='set to True if you need to upscale images back to their original high resolution')
parser.add_argument('--prob', type=float, default=0.2, help='The number that defines the maximum probability difference required between the most probable and second most probable classes for a pixel to be classified as not_sure')
parser.add_argument('--thin', type=int, default=30, help='Number of iterations to perform thinning')
parser.add_argument('--beta', type=float, default=110, help='Beta for Random Walker')
parser.add_argument('--prune', type=int, default=19, help='Number of iterations to perform pruning')
parser.add_argument('--evaluate', type=str2bool, default=True, help='set to True if you want to evaluate the results using gt_images')
parser.add_argument('--ideal',type = str2bool, default = False,  help = 'set to true if you want to use ideal values of prob, thin, beta, and prune to refine your segmentations')

args = parser.parse_args()


#ideal hyperparameters for refining segmentations of certain Networks.
ideal_params = { 
'BIG':{
'deeplab': {'prob':0.035, 'thin':36 , 'beta':112 , 'prune':16},
'fcn': {'prob':1, 'thin':85 , 'beta':108 , 'prune':10}
},

'pascal':{
'deeplab': {'prob':0.285, 'thin':0 , 'beta':105 , 'prune':0},
'fcn': {'prob':1.7, 'thin':0 , 'beta':139 , 'prune':0}
}
}

#------------------------------------------------

def main():

	if args.ideal:
		args.prob = ideal_params[args.dataset][args.network]['prob']
		args.thin = ideal_params[args.dataset][args.network]['thin']
		args.beta = ideal_params[args.dataset][args.network]['beta']
		args.prune = ideal_params[args.dataset][args.network]['prune']

	print("chosen parameters:\n")
	print("prob = %f\nthin = %d\nbeta = %f\nprune = %d"%(args.prob,args.thin,args.beta,args.prune))

	#generate dollar gradient images, might take a while.
	gen_gradient.gen_grad(args.dataset)

	#upscaling
	up.upscale(args.dataset,args.network,args.prob,args.upscale,args.not_sure)
	
	#run random walker
	refine.refinement(args.dataset,args.thin,args.prune,args.beta)

	#evaluate iou
	if args.evaluate:
		evaluate.eval_iou(args.dataset,args.prob,args.thin,args.prune,args.beta)


if __name__ == '__main__':
	main()
