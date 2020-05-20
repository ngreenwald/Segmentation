import numpy as np
import pandas as pd
from segmentation.utils import spatialanalysis_utils
from segmentation.utils import spatialorgmarkerexpression_utils
import importlib
importlib.reload(spatialorgmarkerexpression_utils)


def test_calc_dist_matrix():
    test_mat = np.zeros((512, 512), dtype="int")
    test_mat[0, 20] = 1
    test_mat[4, 17] = 2

    dist_matrix = spatialanalysis_utils.calc_dist_matrix(test_mat)
    real_mat = np.array([[0, 5], [5, 0]])
    assert np.array_equal(dist_matrix, real_mat)


# def test_distmat():
#     testtiff = io.imread(
#         "/Users/jaiveersingh/Documents/MATLAB/SpatialAnalysis/newLmod.tiff")
#     distmat = np.asarray(pd.read_csv(
#         "/Users/jaiveersingh/Documents/MATLAB/distancesMat5.csv", header=None))
#     testmat = spatialanalysis_utils.calc_dist_matrix(testtiff)
#     assert np.allclose(distmat, testmat)


def make_distance_matrix(typeofenfrichment):
    if(typeofenfrichment == "none"):
        randmat = np.random.randint(0, 200, size=(60, 60))
        np.fill_diagonal(randmat, 0)
        return randmat
    elif(typeofenfrichment == "positive"):
        distmatp = np.zeros((80, 80))
        distmatp[10:20, :10] = 50
        distmatp[:10, 10:20] = 50
        distmatp[20:40, :20] = 200
        distmatp[:20, 20:40] = 200
        distmatp[40:80, :40] = 300
        distmatp[:40, 40:80] = 300
        return distmatp
    elif(typeofenfrichment == "negative"):
        distmatn = np.zeros((60, 60))
        distmatn[20:40, :20] = 300
        distmatn[:20, 20:40] = 300
        distmatn[40:50, :40] = 50
        distmatn[:40, 40:50] = 50
        distmatn[50:60, :50] = 200
        distmatn[:50, 50:60] = 200
        return distmatn


def make_expression_matrix(typeofencrichment):
    if(typeofencrichment == "none"):
        cellarray = pd.DataFrame(np.zeros((60, 31)))
        cellarray[30] = "Point8"
        cellarray[24] = np.arange(len(cellarray[1])) + 1
        cellarray.iloc[0:20, 2] = 1
        cellarray.iloc[20:40, 3] = 1
        cellarray = cellarray.rename({0: 'cell_size', 1: 'Background', 14: "HH3",
                                        23: "summed_channel", 24: "label", 25: "area", 26: "eccentricity",
                                        27: "major_axis_length", 28: "minor_axis_length", 29: "perimeter",
                                        30: "fov"}, axis=1)
        return cellarray
    elif(typeofencrichment == "positive"):
        cellarrayp = pd.DataFrame(np.zeros((80, 31)))
        cellarrayp[30] = "Point8"
        cellarrayp[24] = np.arange(len(cellarrayp[1])) + 1
        cellarrayp.iloc[0:8, 2] = 1
        cellarrayp.iloc[28:30, 2] = 1
        cellarrayp.iloc[38:40, 2] = 1
        cellarrayp.iloc[10:18, 3] = 1
        cellarrayp.iloc[27, 3] = 1
        cellarrayp.iloc[30, 3] = 1
        cellarrayp.iloc[36:38, 3] = 1
        cellarrayp = cellarrayp.rename({0: 'cell_size', 1: 'Background', 14: "HH3",
                                        23: "summed_channel", 24: "label", 25: "area", 26: "eccentricity",
                                        27: "major_axis_length", 28: "minor_axis_length", 29: "perimeter",
                                        30: "fov"}, axis=1)
        return cellarrayp
    elif(typeofencrichment == "negative"):
        cellarrayn = pd.DataFrame(np.zeros((60, 31)))
        cellarrayn[30] = "Point8"
        cellarrayn[24] = np.arange(len(cellarrayn[1])) + 1
        cellarrayn.iloc[0:20, 2] = 1
        cellarrayn.iloc[20:40, 3] = 1
        cellarrayn = cellarrayn.rename({0: 'cell_size', 1: 'Background', 14: "HH3",
                                        23: "summed_channel", 24: "label", 25: "area", 26: "eccentricity",
                                        27: "major_axis_length", 28: "minor_axis_length", 29: "perimeter",
                                        30: "fov"}, axis=1)
        return cellarrayn


def make_threshold_mat():
    #thresh = pd.DataFrame(np.zeros((37, 2)))
    thresh = pd.DataFrame(np.zeros((20, 2)))
    thresh.iloc[:, 1] = .5
    return thresh


def make_test_closenum():
    cellarray = pd.DataFrame(np.zeros((10, 31)))
    cellarray[30] = "Point8"
    cellarray[24] = np.arange(len(cellarray[1])) + 1
    cellarray.iloc[0:4, 2] = 1
    cellarray.iloc[0:4, 3] = 1
    cellarray.iloc[4:9, 5] = 1
    cellarray.iloc[4:9, 6] = 1
    cellarray.iloc[9, 7] = 1
    cellarray.iloc[9, 8] = 1

    distmat = np.zeros((10, 10))
    np.fill_diagonal(distmat, 0)
    distmat[1:4, 0] = 50
    distmat[0, 1:4] = 50
    distmat[4:9, 0] = 200
    distmat[0, 4:9] = 200
    distmat[9, 0] = 500
    distmat[0, 9] = 500
    distmat[2:4, 1] = 50
    distmat[1, 2:4] = 50
    distmat[4:9, 1] = 150
    distmat[1, 4:9] = 150
    distmat[9, 1:9] = 200
    distmat[1:9, 9] = 200
    distmat[3, 2] = 50
    distmat[2, 3] = 50
    distmat[4:9, 2] = 150
    distmat[2, 4:9] = 150
    distmat[4:9, 3] = 150
    distmat[3, 4:9] = 150
    distmat[5:9, 4] = 50
    distmat[4, 5:9] = 50
    distmat[6:9, 5] = 50
    distmat[5, 6:9] = 50
    distmat[7:9, 6] = 50
    distmat[6, 7:9] = 50
    distmat[8, 7] = 50
    distmat[7, 8] = 50

    return cellarray, distmat


def test_closenum():
    # test the closenum function
    test_cellarray, test_distmat = make_test_closenum()
    test_thresholds = make_threshold_mat()

    point = "Point8"
    dist_lim = 100

    # subsetting expression matrix
    data_all = test_cellarray
    # identifies columns with markers
    # subsets the expression matrix to only have marker columns
    data_markers = data_all.drop(data_all.columns[[0, 1, 14, 23, 24, 25, 26, 27, 28, 29, 30]], axis=1)
    # list of all markers
    marker_titles = data_markers.columns
    # length of marker list
    marker_num = len(marker_titles)

    # subsetting threshold matrix to only include column with threshold values
    thresh_vec = test_thresholds.iloc[0:20, 1]

    # subsetting expression matrix to only include patients with correct label
    patient_ids = data_all.iloc[:, 30] == point
    patient_data = data_all[patient_ids]
    # patients with correct label, and only columns of markers
    patient_data_markers = data_markers[patient_ids]

    test_closenum, marker1_num, marker2_num = spatialorgmarkerexpression_utils.helper_function_closenum(
        patient_data, patient_data_markers, thresh_vec, test_distmat, marker_num, dist_lim)
    assert (test_closenum[:2, :2] == 16).all()
    # assert (test_closenum[2:4, 2:4] == 25).all()
    # assert (test_closenum[4:6, 4:6] == 1).all()

    # Now test with Erin's output
    # cell_array = pd.read_csv(
    #     "/Users/jaiveersingh/Downloads/SpatialEnrichm"
    #     "ent/granA_cellpheno_CS-asinh-norm_revised.csv")
    # marker_thresholds = pd.read_csv(
    #     "/Users/jaiveersingh/Downloads/SpatialEnrichment/markerThresholds.csv")
    # dist_matrix = np.asarray(pd.read_csv(
    #     "/Users/jaiveersingh/Documents/MATLAB/distancesMat5.csv", header=None))
    # closenum, closenumRand, z, muhat, sigmahat, p, h, adj_p, markertitles = \
    #     spatialorgmarkerexpression_utils.calculate_channel_spatial_enrichment(
    #         dist_matrix, marker_thresholds, cell_array)
    # real_closenum = np.asarray(pd.read_csv(
    #     "/Users/jaiveersingh/Documents/MATLAB/SpatialAnalysis/closeNum.csv"))
    # assert np.array_equal(closenum, real_closenum)


def test_closenumrand():
    test_cellarray, test_distmat = make_test_closenum()
    test_thresholds = make_threshold_mat()

    point = "Point8"
    dist_lim = 100

    # subsetting expression matrix
    data_all = test_cellarray
    # identifies columns with markers
    # subsets the expression matrix to only have marker columns
    data_markers = data_all.drop(data_all.columns[[0, 1, 14, 23, 24, 25, 26, 27, 28, 29, 30]], axis=1)
    # list of all markers
    marker_titles = data_markers.columns
    # length of marker list
    marker_num = len(marker_titles)

    # subsetting threshold matrix to only include column with threshold values
    thresh_vec = test_thresholds.iloc[0:20, 1]

    # subsetting expression matrix to only include patients with correct label
    patient_ids = data_all.iloc[:, 30] == point
    patient_data = data_all[patient_ids]
    # patients with correct label, and only columns of markers
    patient_data_markers = data_markers[patient_ids]

    close_num, marker1_num, marker2_num = spatialorgmarkerexpression_utils.helper_function_closenum(
        patient_data, patient_data_markers, thresh_vec, test_distmat, marker_num, dist_lim)

    test_closenumrand = spatialorgmarkerexpression_utils.helper_function_closenumrand(
        marker1_num, marker2_num, patient_data, test_distmat, marker_num, dist_lim)
    assert test_closenumrand.shape == (20, 20, 100)


def test_spatial_analysis():
    # test z and p values
    marker_thresholds = make_threshold_mat()
    # positive enrichment
    cellarrayp = make_expression_matrix("positive")
    distmatp = make_distance_matrix("positive")
    closenum, closenumrand, z, muhat, sigmahat, p, h, adj_p, markertitles = \
        spatialorgmarkerexpression_utils.calculate_channel_spatial_enrichment(
            distmatp, marker_thresholds, cellarrayp)
    assert p[0, 1, 0] < .05
    assert p[0, 1, 1] > .05
    assert z[0, 1] > 0
    # negative enrichment
    cellarrayn = make_expression_matrix("negative")
    distmatn = make_distance_matrix("negative")
    closenum, closenumrand, z, muhatn, sigmahatn, p, h, adj_p, markertitles = \
        spatialorgmarkerexpression_utils.calculate_channel_spatial_enrichment(
            distmatn, marker_thresholds, cellarrayn)
    assert p[0, 1, 1] < .05
    assert p[0, 1, 0] > .05
    assert z[0, 1] < 0
    # no enrichment
    cellarray = make_expression_matrix("none")
    distmat = make_distance_matrix("none")
    closenum, closenumrand, z, muhat, sigmahat, p, h, adj_p, markertitles = \
        spatialorgmarkerexpression_utils.calculate_channel_spatial_enrichment(
            distmat, marker_thresholds, cellarray)
    assert p[0, 1, 0] > .05
    assert p[0, 1, 1] > .05
    assert abs(z[0, 1]) < 1
