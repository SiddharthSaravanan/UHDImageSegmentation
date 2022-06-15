import numpy as np
import cv2
import skimage.segmentation as seg
import math
import os

from numpy.core.multiarray import normalize_axis_index
from skimage.util import img_as_float

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


def _compute_weights_3d(data, **param):
    # Weight calculation is main difference in multispectral version
    # Original gradient**2 replaced with sum of gradients ** 2 (a[...,channel][:-1]+a[...,channel][1:])
    spacing = np.ones(3)
    gradients = np.concatenate(
        [discrete_sum(data[..., 0], axis=ax).ravel() / spacing[ax]
         for ax in [2, 1, 0] if data.shape[ax] > 1], axis=0) ** 2
    for channel in range(1, data.shape[-1]):
        gradients += np.concatenate(
            [discrete_sum(data[..., channel], axis=ax).ravel() / spacing[ax]
             for ax in [2, 1, 0] if data.shape[ax] > 1], axis=0) ** 2

    # All channels considered together in this standard deviation
    scale_factor = -param['beta'] / (data.std())
    weights = np.exp(scale_factor * gradients)
    weights += param['eps']
    return weights



def construct_graph(data,**param):

    data = np.atleast_3d(img_as_float(data))[..., np.newaxis]
    l_x, l_y, l_z = data.shape[:3]

    edges = _make_graph_edges_3d(l_x, l_y, l_z)
    weights = _compute_weights_3d(data, **param)

    return edges, weights