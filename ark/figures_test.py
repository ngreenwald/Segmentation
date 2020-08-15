import numpy as np
import pandas as pd

from segmentation import figures
import importlib
importlib.reload(figures)


def test_plot_mod_ap():
    labels = ['alg1', 'alg2', 'alg3']
    thresholds = np.arange(0.5, 1, 0.1)
    mAP_array = [{'scores': [0.9, 0.8, 0.7, 0.4, 0.2]}, {'scores': [0.8, 0.7, 0.6, 0.3, 0.1]},
                 {'scores': [0.95, 0.85, 0.75, 0.45, 0.25]}]

    figures.plot_mod_ap(mAP_array, thresholds, labels)


def test_plot_error_types():
    stats_dict = {
        'n_pred': 200,
        'n_true': 200,
        'correct_detections': 140,
        'missed_detections': 40,
        'gained_detections': 30,
        'merge': 20,
        'split': 10,
        'catastrophe': 20
    }

    stats_dict1 = {
        'n_pred': 210,
        'n_true': 210,
        'correct_detections': 120,
        'missed_detections': 30,
        'gained_detections': 50,
        'merge': 50,
        'split': 30,
        'catastrophe': 50
    }

    stats_dict2 = {
        'n_pred': 10,
        'n_true': 20,
        'correct_detections': 10,
        'missed_detections': 70,
        'gained_detections': 50,
        'merge': 5,
        'split': 3,
        'catastrophe': 5
    }

    figures.plot_error_types([stats_dict, stats_dict1, stats_dict2], ['alg1', 'alg2', 'alg3'],
                                ['missed_detections', 'gained_detections', 'merge', 'split',
                                 'catastrophe'], ['red', 'violet', 'green', 'blue'])


def test_plot_f1_scores():
    f1_list = [20, 29, 45, 60, 69]
    method_list = ['Ilastik', 'PixelNet', 'MaskRCNN', 'FPN', 'Multiplexed Deepcell']
    colors = ['red', 'yellow', 'green', 'blue', 'purple']
    figures.plot_f1_scores(f1_list=f1_list, method_list=method_list, colors=colors)


def test_plot_annotator_agreement():
    data1 = np.random.rand(10)
    data2 = np.random.rand(7)
    data3 = np.random.rand(4)
    data = [data1, data2, data3]

    labels = ['cHL annotator', 'cHL model', 'other']

    figures.plot_annotator_agreement(f1_scores_list=data, labels=labels)


def test_nuclear_expansion_pixel():
    fake_labels = np.zeros((50, 50))
    fake_labels[5:15, 5:15] = 1
    fake_labels[21:31, 21:31] = 2

    expanded_labels = nuclear_expansion_pixel(fake_labels, 3)

    assert np.sum(expanded_labels > 0) > np.sum(fake_labels > 0)


def test_get_paired_regionprops():
    true_labels, pred_labels = np.zeros((30, 30)), np.zeros((30, 30))

    # first set of paired labels
    true_labels[:5, :5] = 1
    pred_labels[:7, :6] = 1

    # overlapping two labels
    true_labels[18:20, 18:20] = 2
    true_labels[10:18, 10:18] = 3

    pred_labels[10:20, 10:20] = 2

    # skipped indx 4
    true_labels[24:26] = 5
    pred_labels[24:26] = 5

    # no corresponding true label
    pred_labels[28:, 28:] = 6

    true_ids = [1, 2, 3, 5]
    true_ecc = [0.1, 0.2, 0.3, 0.4]

    pred_ids = [1, 2, 5, 6]
    pred_ecc = [0.1, 0.3, 0.4, 0]

    true_frame = pd.DataFrame({'label': true_ids, 'eccentricity': true_ecc})
    pred_frame = pd.DataFrame({'label': pred_ids, 'eccentricity': pred_ecc})

    match_ecc_true, match_ecc_pred = figures.get_paired_regionprops(true_labels=true_labels,
                                                                    pred_labels=pred_labels,
                                                                    true_props_table=true_frame,
                                                                    pred_props_table=pred_frame,
                                                                    field='eccentricity')
    assert match_ecc_true == match_ecc_pred