import numpy as np
import cv2
import skimage.segmentation as seg
from skimage.segmentation import flood, flood_fill
import random
import math
import os

from numpy.core.multiarray import normalize_axis_index
from scipy import sparse, ndimage as ndi

from skimage._shared import utils
from skimage._shared.utils import warn

#------------------------------------------------

try:
	from scipy.sparse.linalg.dsolve.linsolve import umfpack
	old_del = umfpack.UmfpackContext.__del__

	def new_del(self):
		try:
			old_del(self)
		except AttributeError:
			pass
	umfpack.UmfpackContext.__del__ = new_del
	UmfpackContext = umfpack.UmfpackContext()
except ImportError:
	UmfpackContext = None

try:
	from pyamg import ruge_stuben_solver
	amg_loaded = True
except ImportError:
	amg_loaded = False

from skimage.util import img_as_float

from scipy.sparse.linalg import cg, spsolve
#-------------------------------------------------

img = np.zeros((1,1))

# array of SE used for thinning/
b = np.zeros((8,3,3))
b[0] = np.array((
		[[-1, -1, -1],
		 [0, 1, 0],
		 [1, 1, 1]]), dtype="int")
b[1] = np.array((
		[[0, -1, -1],
		 [1, 1, -1],
		 [1, 1, 0]]), dtype="int")
b[2] = np.array((
		[[1, 0, -1],
		 [1, 1, -1],
		 [1, 0, -1]]), dtype="int")
b[3] = np.array((
		[[1, 1, 0],
		 [1, 1, -1],
		 [0, -1, -1]]), dtype="int")
b[4] = np.array((
		[[1, 1, 1],
		 [0, 1, 0],
		 [-1, -1, -1]]), dtype="int")
b[5] = np.array((
		[[0, 1, 1],
		 [-1, 1, 1],
		 [-1, -1, 0]]), dtype="int")
b[6] = np.array((
		[[-1, 0, 1],
		 [-1, 1, 1],
		 [-1, 0, 1]]), dtype="int")
b[7] = np.array((
		[[-1, -1, 0],
		 [-1, 1, 1],
		 [0, 1, 1]]), dtype="int")


#--------------------------------------------------------------------
# function used to calculate edge weights
def discrete_sum(a, axis=-1):

	a = np.asanyarray(a)
	nd = a.ndim
	if nd == 0:
		raise ValueError("diff requires input that is at least one dimensional")
	axis = normalize_axis_index(axis, nd)

	combined = []
	combined.append(a)

	if len(combined) > 1:
		a = np.concatenate(combined, axis)

	slice1 = [slice(None)] * nd
	slice2 = [slice(None)] * nd
	slice1[axis] = slice(1, None)
	slice2[axis] = slice(None, -1)
	slice1 = tuple(slice1)
	slice2 = tuple(slice2)

	op = not_equal if a.dtype == np.bool_ else np.add
	for _ in range(1):
		a = op(a[slice1], a[slice2])

	return a


def _make_graph_edges_3d(n_x, n_y, n_z):
	"""Returns a list of edges for a 3D image.
	Parameters
	----------
	n_x : integer
		The size of the grid in the x direction.
	n_y : integer
		The size of the grid in the y direction
	n_z : integer
		The size of the grid in the z direction
	Returns
	-------
	edges : (2, N) ndarray
		with the total number of edges::
			N = n_x * n_y * (nz - 1) +
				n_x * (n_y - 1) * nz +
				(n_x - 1) * n_y * nz
		Graph edges with each column describing a node-id pair.
	"""
	vertices = np.arange(n_x * n_y * n_z).reshape((n_x, n_y, n_z))
	edges_deep = np.vstack((vertices[..., :-1].ravel(),
							vertices[..., 1:].ravel()))
	edges_right = np.vstack((vertices[:, :-1].ravel(),
							 vertices[:, 1:].ravel()))
	edges_down = np.vstack((vertices[:-1].ravel(), vertices[1:].ravel()))
	edges = np.hstack((edges_deep, edges_right, edges_down))
	return edges


def _compute_weights_3d(data, spacing, beta, eps, multichannel):
	# Weight calculation is main difference in multispectral version
	# Original gradient**2 replaced with sum of gradients ** 2 (a[...,channel][:-1]+a[...,channel][1:])
	# print(data.shape)
	gradients = np.concatenate(
		[discrete_sum(data[..., 0], axis=ax).ravel() / spacing[ax]
		 for ax in [2, 1, 0] if data.shape[ax] > 1], axis=0) ** 2
	for channel in range(1, data.shape[-1]):
		gradients += np.concatenate(
			[discrete_sum(data[..., channel], axis=ax).ravel() / spacing[ax]
			 for ax in [2, 1, 0] if data.shape[ax] > 1], axis=0) ** 2

	# print(gradients.shape)

	# All channels considered together in this standard deviation
	scale_factor = -beta / (10 * data.std())
	if multichannel:
		# New final term in beta to give == results in trivial case where
		# multiple identical spectra are passed.
		scale_factor /= np.sqrt(data.shape[-1])
	weights = np.exp(scale_factor * gradients)
	weights += eps
	return weights


def _build_laplacian(data, spacing, mask, beta, multichannel):
	l_x, l_y, l_z = data.shape[:3]
	edges = _make_graph_edges_3d(l_x, l_y, l_z)
	weights = _compute_weights_3d(data, spacing, beta=beta, eps=1.e-10,
								  multichannel=multichannel)
	if mask is not None:
		# Remove edges of the graph connected to masked nodes, as well
		# as corresponding weights of the edges.
		mask0 = np.hstack([mask[..., :-1].ravel(), mask[:, :-1].ravel(),
						   mask[:-1].ravel()])
		mask1 = np.hstack([mask[..., 1:].ravel(), mask[:, 1:].ravel(),
						   mask[1:].ravel()])
		ind_mask = np.logical_and(mask0, mask1)
		edges, weights = edges[:, ind_mask], weights[ind_mask]

		# Reassign edges labels to 0, 1, ... edges_number - 1
		_, inv_idx = np.unique(edges, return_inverse=True)
		edges = inv_idx.reshape(edges.shape)

	# Build the sparse linear system
	pixel_nb = l_x * l_y * l_z
	i_indices = edges.ravel()
	j_indices = edges[::-1].ravel()
	data = np.hstack((weights, weights))
	# print(data.shape)
	lap = sparse.coo_matrix((data, (i_indices, j_indices)),
							shape=(pixel_nb, pixel_nb))
	lap.setdiag(-np.ravel(lap.sum(axis=0)))
	return lap.tocsr()


def _build_linear_system(data, spacing, labels, nlabels, mask,
						 beta, multichannel):
	"""
	Build the matrix A and rhs B of the linear system to solve.
	A and B are two block of the laplacian of the image graph.
	"""
	if mask is None:
		labels = labels.ravel()
	else:
		labels = labels[mask]

	indices = np.arange(labels.size)
	seeds_mask = labels > 0
	unlabeled_indices = indices[~seeds_mask]
	seeds_indices = indices[seeds_mask]

	lap_sparse = _build_laplacian(data, spacing, mask=mask,
								  beta=beta, multichannel=multichannel)

	rows = lap_sparse[unlabeled_indices, :]
	lap_sparse = rows[:, unlabeled_indices]
	B = -rows[:, seeds_indices]

	seeds = labels[seeds_mask]
	seeds_mask = sparse.csc_matrix(np.hstack(
		[np.atleast_2d(seeds == lab).T for lab in range(1, nlabels + 1)]))
	rhs = B.dot(seeds_mask)

	return lap_sparse, rhs


def _solve_linear_system(lap_sparse, B, tol, mode):

	if mode is None:
		mode = 'cg_j'

	if mode == 'cg_mg' and not amg_loaded:
		warn('"cg_mg" not available, it requires pyamg to be installed. '
			 'The "cg_j" mode will be used instead.',
			 stacklevel=2)
		mode = 'cg_j'

	if mode == 'bf':
		X = spsolve(lap_sparse, B.toarray()).T
	else:
		maxiter = None
		if mode == 'cg':
			if UmfpackContext is None:
				warn('"cg" mode may be slow because UMFPACK is not available. '
					 'Consider building Scipy with UMFPACK or use a '
					 'preconditioned version of CG ("cg_j" or "cg_mg" modes).',
					 stacklevel=2)
			M = None
		elif mode == 'cg_j':
			M = sparse.diags(1.0 / lap_sparse.diagonal())
		else:
			# mode == 'cg_mg'
			lap_sparse = lap_sparse.tocsr()
			ml = ruge_stuben_solver(lap_sparse)
			M = ml.aspreconditioner(cycle='V')
			maxiter = 30
		cg_out = [
			cg(lap_sparse, B[:, i].toarray(),
			   tol=tol, atol=0, M=M, maxiter=maxiter)
			for i in range(B.shape[1])]
		if np.any([info > 0 for _, info in cg_out]):
			warn("Conjugate gradient convergence to tolerance not achieved. "
				 "Consider decreasing beta to improve system conditionning.",
				 stacklevel=2)
		X = np.asarray([x for x, _ in cg_out])

	return X


def _preprocess(labels):

	label_values, inv_idx = np.unique(labels, return_inverse=True)

	if not (label_values == 0).any():
		warn('Random walker only segments unlabeled areas, where '
			 'labels == 0. No zero valued areas in labels were '
			 'found. Returning provided labels.',
			 stacklevel=2)

		return labels, None, None, None, None

	# If some labeled pixels are isolated inside pruned zones, prune them
	# as well and keep the labels for the final output

	null_mask = labels == 0
	pos_mask = labels > 0
	mask = labels >= 0

	fill = ndi.binary_propagation(null_mask, mask=mask)
	isolated = np.logical_and(pos_mask, np.logical_not(fill))

	pos_mask[isolated] = False

	# If the array has pruned zones, be sure that no isolated pixels
	# exist between pruned zones (they could not be determined)
	if label_values[0] < 0 or np.any(isolated):
		isolated = np.logical_and(
			np.logical_not(ndi.binary_propagation(pos_mask, mask=mask)),
			null_mask)

		labels[isolated] = -1
		if np.all(isolated[null_mask]):
			warn('All unlabeled pixels are isolated, they could not be '
				 'determined by the random walker algorithm.',
				 stacklevel=2)
			return labels, None, None, None, None

		mask[isolated] = False
		mask = np.atleast_3d(mask)
	else:
		mask = None

	# Reorder label values to have consecutive integers (no gaps)
	zero_idx = np.searchsorted(label_values, 0)
	labels = np.atleast_3d(inv_idx.reshape(labels.shape) - zero_idx)

	nlabels = label_values[zero_idx + 1:].shape[0]

	inds_isolated_seeds = np.nonzero(isolated)
	isolated_values = labels[inds_isolated_seeds]

	return labels, nlabels, mask, inds_isolated_seeds, isolated_values


# @utils.channel_as_last_axis(multichannel_output=False)
# @utils.deprecate_multichannel_kwarg(multichannel_position=6)
def random_walk(data, labels, beta=130, mode='cg_j', tol=1.e-3, copy=True,
				  multichannel=False, return_full_prob=False, spacing=None,
				  *, prob_tol=1e-3, channel_axis=None):
	
	# Parse input data
	if mode not in ('cg_mg', 'cg', 'bf', 'cg_j', None):
		raise ValueError(
			"{mode} is not a valid mode. Valid modes are 'cg_mg',"
			" 'cg', 'cg_j', 'bf' and None".format(mode=mode))

	# Spacing kwarg checks
	if spacing is None:
		spacing = np.ones(3)
	elif len(spacing) == labels.ndim:
		if len(spacing) == 2:
			# Need a dummy spacing for singleton 3rd dim
			spacing = np.r_[spacing, 1.]
		spacing = np.asarray(spacing)
	else:
		raise ValueError('Input argument `spacing` incorrect, should be an '
						 'iterable with one number per spatial dimension.')

	# This algorithm expects 4-D arrays of floats, where the first three
	# dimensions are spatial and the final denotes channels. 2-D images have
	# a singleton placeholder dimension added for the third spatial dimension,
	# and single channel images likewise have a singleton added for channels.
	# The following block ensures valid input and coerces it to the correct
	# form.
	multichannel = channel_axis is not None
	if not multichannel:
		if data.ndim not in (2, 3):
			raise ValueError('For non-multichannel input, data must be of '
							 'dimension 2 or 3.')
		if data.shape != labels.shape:
			raise ValueError('Incompatible data and labels shapes.')
		data = np.atleast_3d(img_as_float(data))[..., np.newaxis]
	else:
		if data.ndim not in (3, 4):
			raise ValueError('For multichannel input, data must have 3 or 4 '
							 'dimensions.')
		if data.shape[:-1] != labels.shape:
			raise ValueError('Incompatible data and labels shapes.')
		data = img_as_float(data)
		if data.ndim == 3:  # 2D multispectral, needs singleton in 3rd axis
			data = data[:, :, np.newaxis, :]

	labels_shape = labels.shape
	labels_dtype = labels.dtype

	if copy:
		labels = np.copy(labels)

	(labels, nlabels, mask,
	 inds_isolated_seeds, isolated_values) = _preprocess(labels)

	if isolated_values is None:
		# No non isolated zero valued areas in labels were
		# found. Returning provided labels.
		if return_full_prob:
			# Return the concatenation of the masks of each unique label
			return np.concatenate([np.atleast_3d(labels == lab)
								   for lab in np.unique(labels) if lab > 0],
								  axis=-1)
		return labels

	# Build the linear system (lap_sparse, B)
	lap_sparse, B = _build_linear_system(data, spacing, labels, nlabels, mask,
										 beta, multichannel)

	# Solve the linear system lap_sparse X = B
	# where X[i, j] is the probability that a marker of label i arrives
	# first at pixel j by anisotropic diffusion.
	X = _solve_linear_system(lap_sparse, B, tol, mode)

	if X.min() < -prob_tol or X.max() > 1 + prob_tol:
		warn('The probability range is outside [0, 1] given the tolerance '
			 '`prob_tol`. Consider decreasing `beta` and/or decreasing '
			 '`tol`.')

	# Build the output according to return_full_prob value
	# Put back labels of isolated seeds
	labels[inds_isolated_seeds] = isolated_values
	labels = labels.reshape(labels_shape)

	mask = labels == 0
	mask[inds_isolated_seeds] = False

	if return_full_prob:
		out = np.zeros((nlabels,) + labels_shape)
		for lab, (label_prob, prob) in enumerate(zip(out, X), start=1):
			label_prob[mask] = prob
			label_prob[labels == lab] = 1
	else:
		X = np.argmax(X, axis=0) + 1
		out = labels.astype(labels_dtype)
		out[mask] = X

	return out


#--------------------------------------------------------------------


def thinning(img1, iterations):
	# img = np.zeros((img1.shape[0]+2,img1.shape[1]+2),dtype='uint8')
	# img[1:img1.shape[0]+1,1:img1.shape[1]+1] = np.copy(img1)
	img = np.copy(img1)
	for i in range(iterations):
		for j in range(8):
			temp = np.copy(img)
			temp = cv2.morphologyEx(src=temp, op=cv2.MORPH_HITMISS, kernel=b[j],borderType=cv2.BORDER_REPLICATE)
			img = img - temp #center of the SE always has 1 (foreground)
		if np.count_nonzero(img) <=40000:
			print("thinning stopped at iteration %d"%(i))
			break
	return img

def gen_seed(thin,thick,x,prune):

	#thinning
	thin = thinning(thin,x)
	#thickening
	thick = thinning(thick,x)

	y=2
	kernel = np.ones((y,y),np.uint8)
	opening_thin = cv2.morphologyEx(thin, cv2.MORPH_OPEN, kernel)
	opening_thick = cv2.morphologyEx(thick, cv2.MORPH_OPEN, kernel)

	thin_prot = thin-opening_thin
	thick_prot = thick-opening_thick

	se = np.ones((13,13),np.uint8)
	dilation_thin_prot = cv2.dilate(thin_prot,se,iterations = 1)
	dilation_thick_prot = cv2.dilate(thick_prot,se,iterations = 1)

	union = cv2.bitwise_or(dilation_thick_prot,dilation_thin_prot)
	intersection = cv2.bitwise_and(dilation_thick_prot,dilation_thin_prot)

	# remove connected components
	xn=intersection
	xnp1=intersection
	se3 = np.ones((3,3),np.uint8)

	no_iter=0
	while True:
		no_iter+=1
		xn=xnp1
		xnp1 = cv2.dilate(xn,se3,iterations = 1)
		xnp1 = cv2.bitwise_and(xnp1,union)

		if np.array_equal(xn,xnp1) or no_iter>=prune:
			break

	union = union - xn

	thin_prot = cv2.bitwise_and(union,thin_prot)
	thick_prot = cv2.bitwise_and(union,thick_prot)

	thin = opening_thin + thin_prot
	thick = opening_thick + thick_prot

	return thin,thick


def refinement(dataset,thin_iter,prune,beta_param):

	imgs_folder = './upscale/upscaled_'+dataset+'/'
	grad_folder = './gradients/grad_images_'+dataset+'/'

	if dataset == 'BIG':
		form = '.jpg'
	if dataset == 'pascal':
		form = '.png'

	no=0
	for fname in os.listdir(imgs_folder):
		file = os.path.join(imgs_folder,fname)

		
	
		print(fname)
		
		img = cv2.imread(file,0)
		graph = cv2.imread(grad_folder+fname.split('.')[0]+form,0)

		if cv2.countNonZero(img)==0:
			result = np.copy(img)
		else:
			
			thin = np.copy(img)
			thick = np.copy(img)

			if np.count_nonzero(img == 255) == 0:
				thin[thin>100] = 255

			thin[thin<200] = 0
			thick[thick>100] = 255
			thick = 255-thick

			if dataset == 'BIG':
				thin,thick = gen_seed(thin,thick,thin_iter,prune)

			thick = np.clip(thick,0,127)
			seed = thin + thick
			
			#random walker
			result = random_walk(graph, seed, beta=beta_param, mode='bf',multichannel = False)

			result = (result-1)*255

			result = cv2.medianBlur(result,7)
			_,result = cv2.threshold(result,200,255,cv2.THRESH_BINARY)

		cv2.imwrite('./refinement_results/'+dataset+'/'+fname,result)
