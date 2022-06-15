"""
Recall there are two parts of the process -

1) Obtain bondaries for high-res image
2) Downscale the image and obtain low-res semantic information. This low-res-semantic information is
further upscaled to obtain seeds.

Then finally we use the random walk to combine the seeds and gradient. 
This is implemented here.
"""
import pdb
import numpy as np
from scipy.sparse.linalg import cg, spsolve

from scipy import sparse, ndimage as ndi

from skimage._shared import utils
from skimage._shared.utils import warn

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


def _build_laplacian(edges, weights, mask, pixel_nb):
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
    # pixel_nb = mask.shape[0]*mask.shape[1]
    i_indices = edges.ravel()
    j_indices = edges[::-1].ravel()
    data = np.hstack((weights, weights))
    # print(data.shape)
    lap = sparse.coo_matrix((data, (i_indices, j_indices)),
                            shape=(pixel_nb, pixel_nb))
    lap.setdiag(-np.ravel(lap.sum(axis=0)))
    return lap.tocsr()


def _build_linear_system(edges, weights, labels, nlabels, mask, pixel_nb):
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

    lap_sparse = _build_laplacian(edges, weights, mask, pixel_nb)

    rows = lap_sparse[unlabeled_indices, :]
    lap_sparse = rows[:, unlabeled_indices]
    B = -rows[:, seeds_indices]

    seeds = labels[seeds_mask]
    seeds_mask = sparse.csc_matrix(np.hstack(
        [np.atleast_2d(seeds == lab).T for lab in range(1, nlabels + 1)]))
    rhs = B.dot(seeds_mask)

    return lap_sparse, rhs

def RW(edges, weights, seeds, **param):
    """
    """
    if np.count_nonzero(seeds) == 0:
        return seeds

    labels_shape = seeds.shape
    labels_dtype = seeds.dtype
    pixel_nb = seeds.shape[0]*seeds.shape[1]
    
    (labels, nlabels, mask, inds_isolated_seeds, isolated_values) = _preprocess(seeds)

    if isolated_values is None: # No non isolated zero valued areas in labels were found. Returning provided labels.
        return labels

    lap_sparse, B = _build_linear_system(edges, weights, labels, nlabels, mask, pixel_nb)
    X = spsolve(lap_sparse, B.toarray()).T
    
    if X.min() < -param['prob_tol'] or X.max() > 1 + param['prob_tol']:
        warn('The probability range is outside [0, 1] given the tolerance `prob_tol`. Consider decreasing `beta` and/or decreasing `tol`.')

    # Build the output according to return_full_prob value
    # Put back labels of isolated seeds
    labels[inds_isolated_seeds] = isolated_values
    labels = labels.reshape(labels_shape)

    mask = labels == 0
    mask[inds_isolated_seeds] = False

    X = np.argmax(X, axis=0) + 1
    out = labels.astype(labels_dtype)
    out[mask] = X

    return out
