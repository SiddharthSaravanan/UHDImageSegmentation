import matlab.engine

def gen_grad(dataset):

	eng = matlab.engine.start_matlab()
	eng.addpath('./gradients/edges-master');

	if dataset == 'BIG':
		eng.grads_BIG(nargout=0)
	elif dataset == 'pascal':
		eng.grads_pascal(nargout=0)
	elif dataset == 'custom':
		eng.grads_custom(nargout=0)

	eng.quit()