import numpy as np
import xarray as xr
import pandas as pd
import skimage.measure
import scipy
import os
from segmentation.utils import spatial_analysis_utils as sau
from scipy.spatial.distance import cdist
from skimage.measure import label


# Constants for random centroid matrix generation
A_CENTROID_FACTOR = 0.5
B_CENTROID_FACTOR = 0.9
C_CENTROID_FACTOR = 0.4

A_CENTROID_COV = [[1, 0], [0, 1]]
B_CENTROID_COV = [[1, 0], [0, 1]]
C_CENTROID_COV = [[1, 0], [0, 1]]


def direct_init_dist_matrix(num_A=100, num_B=100, num_C=100,
    ab_dist_mean=100, ab_dist_var=1, ac_dist_mean=20, ac_dist_var=1,
    distr_AB={'mean': 100, 'var': 1}, distr_AC={'mean': 20, 'var': 1},
    seed=None):
    """
    This function will return a random dist matrix such that the distance between cells
    of types A and B are overall larger than the distance between cells of types A and C

    Each row and column representing a cell.
    We generate the points using Gaussian distributions
    Ideally, the parameters for A to B distances will be set such that they produce a lower range of values
    than A to C distances.

    Will return a random distance matrix as well as the dictionary of associated cell: label IDs
    The above is generated from the generate_labels function

    Args:
        num_A: the number of A cells we wish to generate. Default 100
        num_B: the number of B cells we wish to generate. Default 100
        num_C: the number of C cells we wish to generate. Default 100
        distr_AB: if specified, will be a dict listing the mean and variance of the Gaussian distribution
            we wish to generate numbers from. Default mean=100 and var=1
        distr_AC: similar to dist_AB. Default mean=20 and var=1
        seed: whether to fix the random seed or not. Useful for testing.
            Should be a specified integer value. Default 42.
    """

    # set the mean and variance of the Gaussian distributions of both AB and AC distances

    mean_ab = distr_AB['mean']
    var_ab = distr_AB['var']

    mean_ac = ac_dist_mean
    var_ac = ac_dist_var

    # set random seed if set
    if seed:
        np.random.seed(seed)

    # we initialize the random distances across different types of points
    # note that we don't really care about aa, bb, bc, or cc, so we
    # initialize those to garbage. We do need them for a proper
    # distance matrix format, however.
    random_aa = np.random.normal(0, 1, (num_A, num_A))
    random_ab = np.random.normal(mean_ab, var_ab, (num_A, num_B))
    random_ac = np.random.normal(mean_ac, var_ac, (num_A, num_C))
    random_bb = np.random.normal(0, 1, (num_B, num_B))
    random_bc = np.random.normal(0, 1, (num_B, num_C))
    random_cc = np.random.normal(0, 1, (num_C, num_C))

    # create each row one-by-one first
    # we need to correct each aa, bb, and cc matrix to ensure symmetry
    first_row = np.concatenate(((random_aa + random_aa.T) / 2, random_ab, random_ac), axis=1)
    second_row = np.concatenate((random_ab.T, (random_bb + random_bb.T) / 2, random_bc), axis=1)
    third_row = np.concatenate((random_ac.T, random_bc.T, (random_cc + random_cc.T) / 2), axis=1)

    # then concatenate them together
    dist_mat = np.concatenate((first_row, second_row, third_row), axis=0)

    # assert that the created submatrix is symmetric
    assert np.allclose(dist_mat, dist_mat.T, rtol=1e-05, atol=1e-08)

    return dist_mat


def get_random_centroid_centers(size_img=(1024, 1024), num_A=100, num_B=100, num_C=100,
    distr_A={'centroid_factor': (0.5, 0.5), 'cov': [[1, 0], [0, 1]]},
    distr_B={'centroid_factor': (0.9, 0.9), 'cov': [[1, 0], [0, 1]]},
    distr_C={'centroid_factor': (0.4, 0.4), 'cov': [[1, 0], [0, 1]]},
    seed=None):
    """
    This function generates random centroid centers in the form of a label map
    such that those of type A will have centers closer on average to those of type B
    than those of type C

    We will use a multivariate Gaussian distribution for A, B, and C type cells to generate their respective centers.

    Returns the set of points associated with the centroids of cells of types A, B, and C.

    Args:
        size_img: a tuple indicating the size of the image. Default 1024 x 1024
        num_A: the number of A centroids to generate. Default 100.
        num_B: the number of B centroids to generate. Default 100.
        num_C: the number of C centroids to generate. Default 100.

        distr_A: a dict indicating the parameters of the multivariate normal distribution to generate A cell centroids.
            Params:
                centroid_factor: a tuple to determine which number to multiply the height and width by
                    to indicate the center (mean) of the distribution
                cov: in the format [[varXX, varXY], [varYX, varYY]]
        distr_B: similar to distr_A
        distr_C: similar to distr_C
        seed: whether to fix the random seed or not. Useful for testing.
            Should be a specified integer value. Default None.
    """

    # extract the height and width
    height = size_img[0]
    width = size_img[1]

    a_mean = (height * distr_A['centroid_factor'][0], width * distr_A['centroid_factor'][1])
    a_cov = distr_A['cov']

    b_mean = (height * distr_B['centroid_factor'][0], width * distr_B['centroid_factor'][1])
    b_cov = distr_B['cov']

    c_mean = (height * distr_C['centroid_factor'][0], width * distr_C['centroid_factor'][1])
    c_cov = distr_C['cov']

    b_mean = (height * B_CENTROID_FACTOR, width * B_CENTROID_FACTOR)
    b_cov = B_CENTROID_COV

    # if specified, set the random seed
    if seed:
        np.random.seed(seed)

    # use the multivariate_normal distribution to generate the points
    # because we're passing these into skimage.measure.label, it is important
    # that we convert these to integers beforehand
    # since label only takes a binary matrix
    a_points = np.random.multivariate_normal(a_mean, a_cov, num_A).astype(np.int16)
    b_points = np.random.multivariate_normal(b_mean, b_cov, num_B).astype(np.int16)
    c_points = np.random.multivariate_normal(c_mean, c_cov, num_C).astype(np.int16)

    # because we have converted to int, it is more likely that points may overap, especially between a and b
    # this check is just to remove any duplicate labelled points to prevent confusion
    intersect_points = np.concatenate(
        np.intersect1d(a_points, b_points), np.intersect1d(a_points, c_points), np.intersect1d(b_points, c_points))

    # for simplicity, we just remove duplicate coordinates from all of a_points, b_points, and c_points
    a_points = a_points[~a_points.isin(intersect_points)]
    b_points = b_points[~b_points.isin(intersect_points)]
    c_points = c_points[~c_points.isin(intersect_points)]

    # get the x and y coords to index binary mat for assigning the matrix
    a_rows, a_cols = zip(*a_points)
    b_rows, b_cols = zip(*b_points)
    c_rows, c_cols = zip(*c_points)

    # generate the binary matrix to pass into label_map
    binary_mat = np.zeros(size_img)
    binary_mat[a_rows, a_cols] = True
    binary_mat[b_rows, b_cols] = True
    binary_mat[c_rows, c_cols] = True

    # generate the label matrix for the image now
    label_mat = label(binary_mat)

    return label_mat


def point_init_dist_matrix(size_img=(1024, 1024), num_A=100, num_B=100, num_C=100, distr_A=None, distr_B=None, distr_C=None, seed=None):
    """
    This function generates random points using the get_random_centroid_centers function and from that
    generates a distance matrix.

    Each row and column of the matrix represents a specific cell, and the elements represent the distance
    from the respective cell centroids.

    The format of the matrix in terms of distances would look like:
        A-A   A-B   A-C
        B-A   B-B   B-C
        C-A   C-B   C-C

    Args:
        size_img: a tuple indicating the size of the image. Default 1024 x 1024.
        num_A: the number of A centroids to generate. Used by the get_random_centroid_centers function. Default 100.
        num_B: similar to num_A
        num_C: similar to num_C
        distr_A: a dict indicating the mean and covariance matrix of the multivariate distribution we pull A centroids from.
            Used by get_random_centroid_centers. Default None, and will use predefined parameters.
        distr_B: dimilar to distr_A
        distr_C: similar to distr_C
        seed: whether to fix the random seed or not. Used by get_random_centroid_centers. Useful for testing.
            Should be a specified integer value. Default None.
    """

    # generate points for cell types A, B, and C
    a_points, b_points, c_points = get_random_centroid_centers(size_img, num_A, num_B, num_C, distr_A, distr_B, distr_C, seed)

    # compute the distances between each point pair
    a_a_dist = cdist(a_points, a_points)
    a_b_dist = cdist(a_points, b_points)
    a_c_dist = cdist(a_points, c_points)
    b_b_dist = cdist(b_points, b_points)
    b_c_dist = cdist(b_points, c_points)
    c_c_dist = cdist(c_points, c_points)

    # create each matrix row
    # we need to correct aa, bb, and cc to ensure symmetry
    first_row = np.concatenate(((a_a_dist + a_a_dist.T) / 2, a_b_dist, a_c_dist), axis=1)
    second_row = np.concatenate((a_b_dist.T, (b_b_dist + b_b_dist.T) / 2, b_c_dist), axis=1)
    third_row = np.concatenate((a_c_dist.T, b_c_dist.T, (c_c_dist + c_c_dist) / 2), axis=1)

    # and then the entire matrix
    dist_mat = np.concatenate((first_row, second_row, third_row), axis=0)

    return dist_mat
