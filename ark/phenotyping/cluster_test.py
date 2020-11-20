import numpy as np
import pandas as pd

import ark.phenotyping.cluster as cluster


def test_decay_function():
    sample_param = 1
    sample_t = 5
    sample_num_iters = 10

    decay_param = cluster.decay_function(sample_param, sample_t, sample_num_iters)

    assert decay_param == 0.5


def test_winner():
    test_sample = np.array([0.01, 0.02])
    test_weights = np.array([[[0.001, 0.005],
                              [0.002, 0.004]],
                             [[0.003, 0.009],
                              [0.008, 0.027]]])

    winning_coords = cluster.winner(test_sample, test_weights)

    assert winning_coords == (1, 1)


def test_update():
    test_sample = np.array([1, 2]).astype(float)
    test_weights = np.array([[[1, 5],
                              [2, 4]],
                             [[3, 9],
                              [8, 27]]]).astype(float)
    test_winning_coords = (1, 1)
    test_sigma = 1.0
    test_learning_rate = 0.5
    test_x_mesh, test_y_mesh = np.meshgrid(np.arange(2), np.arange(2))

    # it's impractical to check exact values, so we'll round to make life easier
    weights = cluster.update(test_sample, test_weights, test_winning_coords,
                             test_sigma, test_learning_rate, test_x_mesh, test_y_mesh)
    weights = np.round(weights, decimals=8)

    result = np.array([[[1., 3.90893398],
                        [1.5735679, 3.1471358]],
                       [[2.1471358, 6.01497529],
                        [4.5, 14.5]]])

    # assert that we get the correct weights results
    assert np.all(weights == result)

    # get the difference in weights, and the magnitude of the change for each neuron
    diff = test_weights - weights
    diff_magnitude = np.apply_along_axis(np.linalg.norm, axis=2, arr=diff)

    # assert that the neuron associated with the winning coords changed the most
    assert np.all(diff_magnitude[test_winning_coords] >= diff)


def test_train_som():
    # create a som with just 1 weight for each channel per pixel
    test_pixel_mat = pd.DataFrame(np.random.rand(9, 4))
    test_x = 3
    test_y = 3
    test_num_passes = 1000

    # only to see if it runs to completion with default sigma, learning_rate, and randomization
    weights = cluster.train_som(test_pixel_mat, test_x, test_y, test_num_passes, random_seed=0)

    # find the winning coordinates associated with the neuron weights closest to each pixel
    cluster_coords = test_pixel_mat.apply(
        lambda row: cluster.winner(np.array(row.values), weights), axis=1)

    # now verify that the assigned weights are the ones that are the closest to each pixel
    for row, coord in enumerate(cluster_coords.values):
        pixel_data = test_pixel_mat.iloc[row, :].values
        diff = weights - pixel_data
        diff_magnitude = np.apply_along_axis(np.linalg.norm, axis=2, arr=diff)

        assert np.all(diff_magnitude[coord] <= diff_magnitude)


def test_cluster_som():
    test_pixel_mat = pd.DataFrame(np.reshape(np.arange(0.01, 0.28, 0.01), (-1, 3)))
    test_weights = np.array([[[0.001, 0.005, 0.010],
                              [0.002, 0.004, 0.007],
                              [0.003, 0.006, 0.009]],
                             [[0.003, 0.009, 0.027],
                              [0.008, 0.027, 0.064],
                              [0.011, 0.022, 0.033]],
                             [[0.025, 0.050, 0.075],
                              [0.033, 0.066, 0.099],
                              [0.016, 0.032, 0.064]]])

    cluster_labels = cluster.cluster_som(test_pixel_mat, test_weights)

    assert np.all(cluster_labels.values == np.array([0, 1, 2, 2, 2, 2, 2, 2, 2]))
