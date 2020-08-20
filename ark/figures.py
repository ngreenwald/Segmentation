import os
import copy

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import skimage.morphology as morph
import skimage.io as io
import networkx as nx

from skimage.transform import resize
from skimage.segmentation import find_boundaries
from skimage.future import graph

from skimage.exposure import rescale_intensity
from deepcell_toolbox.metrics import Metrics


from scipy.ndimage import gaussian_filter


def plot_mod_ap(mod_ap_list, thresholds, labels):
    df = pd.DataFrame({'iou': thresholds})

    for idx, label in enumerate(labels):
        df[label] = mod_ap_list[idx]['scores']

    fig, ax = plt.subplots()
    for label in labels:
        ax.plot('iou', label, data=df, linestyle='-', marker='o')

    ax.set_xlabel('IOU Threshold')
    ax.set_ylabel('mAP')
    ax.legend()
    fig.show()


def plot_error_types(error_dicts, method_labels, error_labels, colors):
    data_dict = pd.DataFrame(pd.Series(error_dicts[0])).transpose()

    # create single dict with all errors
    for i in range(1, len(method_labels)):
        data_dict = data_dict.append(error_dicts[i], ignore_index=True)

    data_dict['algos'] = method_labels

    fig, axes = plt.subplots(1, len(error_labels), figsize=(15, 4))
    for i in range(len(error_labels)):
        barchart_helper(ax=axes[i], values=data_dict[error_labels[i]], labels=method_labels,
                        title='{} Errors'.format(error_labels[i]), colors=colors, y_max=80)

    fig.show()
    #fig.tight_layout()


def plot_f1_scores(f1_list, method_list, colors):
    fig, ax = plt.subplots(figsize=(8, 8))
    barchart_helper(ax=ax, values=f1_list, labels=method_list, title='F1 scores', colors=colors,
                    y_max=80)


def barchart_helper(ax, values, labels, title, colors, y_max):

    # bars are evenly spaced based on number of categories
    positions = range(len(values))
    ax.bar(positions, values, color=colors)

    # x ticks
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)

    # y ticks
    y_positions = np.arange(0, y_max, 10)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_positions)

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    # title
    ax.set_title(title)


def plot_annotator_agreement(f1_scores_list, labels):

    fig, ax = plt.subplots()
    for i in range(len(labels)):
        current_data = f1_scores_list[i]
        x = [i] * len(current_data)
        ax.plot(x, current_data, marker='o', linestyle='none', color='blue')

    ax.set_xticks(list(range(len(labels))))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    #ax.yaxis.set_ticks_position('left')
    #ax.xaxis.set_ticks_position('bottom')


def nuclear_expansion_pixel(label_map, expansion_radius):
    expanded_map = morph.dilation(label_map, selem=morph.disk(expansion_radius))
    return expanded_map


def nuclear_expansion_watershed(label, membrane):
    new_labels = morph.watershed(membrane, markers=label, watershed_line=False)
    return new_labels


def get_paired_regionprops(true_labels, pred_labels, true_props_table, pred_props_table,
                        field='eccentricity'):

    true_field, pred_field = [], []

    for pred_idx, pred_cell in enumerate(pred_props_table['label']):
        pred_mask = pred_labels == pred_cell
        overlap_ids, overlap_counts = np.unique(true_labels[pred_mask], return_counts=True)

        # get ID of the true cell that overlaps with predicted cell most
        max_overlap = np.max(overlap_counts)
        max_idx = np.where(np.isin(overlap_counts, max_overlap))[0][0]
        max_id = overlap_ids[max_idx]

        # no matching cell
        if max_id == 0:
            pass
        else:
            true_row_idx = np.where(np.isin(true_props_table['label'], max_id))[0][0]
            true_field.append(true_props_table[field][true_row_idx])
            pred_field.append(pred_props_table[field][pred_idx])

    return true_field, pred_field


def preprocess_overlays(img_dir):
    DNA = io.imread(os.path.join(img_dir, 'DNA.tiff'))
    #DNA = gaussian_filter(DNA, 1)
    DNA = resize(DNA, [DNA.shape[0] * 2, DNA.shape[1] * 2], order=3)
    DNA = DNA.astype('float32')
    DNA = DNA / np.max(DNA)

    Membrane = io.imread(os.path.join(img_dir, 'Membrane.tiff'))
    Membrane = resize(Membrane, [Membrane.shape[0] * 2, Membrane.shape[1] * 2], order=3)
    Membrane = Membrane.astype('float32')
    Membrane = Membrane / np.max(Membrane)

    labels = io.imread(os.path.join(img_dir, 'labels.tiff'))
    labels = resize(labels, [labels.shape[0] * 2, labels.shape[1] * 2],
                    order=0, preserve_range=True)
    labels = labels.astype('int16')
    boundaries_sub = find_boundaries(labels.astype('int'), connectivity=1, mode='subpixel')

    # subpixel producing an image with len i * 2 - 1, need to pad with a single pixel
    boundaries = np.zeros_like(Membrane)
    boundaries[:-1, :-1] = boundaries_sub

    io.imsave(os.path.join(img_dir, 'DNA_resized.tiff'), DNA)
    io.imsave(os.path.join(img_dir, 'Membrane_resized.tiff'), Membrane)
    io.imsave(os.path.join(img_dir, 'boundaries_resized.tiff'), boundaries)
    io.imsave(os.path.join(img_dir, 'labels_resized.tiff'), labels)


def generate_crop(img_dir, row_start, col_start, length):
    DNA = io.imread(os.path.join(img_dir, 'DNA_resized.tiff'))
    DNA_cropped = DNA[row_start:(row_start + length), col_start:(col_start + length)]

    Membrane = io.imread(os.path.join(img_dir, 'Membrane_resized.tiff'))
    Membrane_cropped = Membrane[row_start:(row_start + length), col_start:(col_start + length)]

    labels = io.imread(os.path.join(img_dir, 'labels_resized.tiff'))
    labels_cropped = labels[row_start:(row_start + length), col_start:(col_start + length)]

    boundaries = io.imread(os.path.join(img_dir, 'boundaries_resized.tiff'))
    boundaries_cropped = boundaries[row_start:(row_start + length), col_start:(col_start + length)]

    io.imsave(os.path.join(img_dir, 'DNA_cropped.tiff'), DNA_cropped)
    io.imsave(os.path.join(img_dir, 'Membrane_cropped.tiff'), Membrane_cropped)
    io.imsave(os.path.join(img_dir, 'labels_cropped.tiff'), labels_cropped)
    io.imsave(os.path.join(img_dir, 'boundaries_cropped.tiff'), boundaries_cropped)


def color_labels_by_graph(labels):
    label_graph = graph.RAG(label_image=labels)
    graph_dict = nx.coloring.greedy_color(label_graph, strategy='largest_first')

    label_outline = find_boundaries(labels.astype('int'), connectivity=1)
    output_labels = copy.copy(labels)

    output_labels[label_outline > 0] = 0

    for idx in np.unique(output_labels):
        mask = output_labels == idx
        if idx == 0:
            output_labels[mask] = 0
        else:
            val = graph_dict[idx]
            output_labels[mask] = val + 1

    return output_labels


def recolor_labels(mask, values=None):
    if values is None:
        values = [100, 130, 160, 190, 220, 250]

    for idx, value in enumerate(values):
        mask[mask == (idx + 1)] = value

    return mask


def generate_inset(img_dir, row_start, col_start, length, inset_num, thickness=2):
    DNA = io.imread(os.path.join(img_dir, 'DNA_cropped.tiff'))
    DNA_inset = DNA[row_start:(row_start + length), col_start:(col_start + length)]

    Membrane = io.imread(os.path.join(img_dir, 'Membrane_cropped.tiff'))
    Membrane_inset = Membrane[row_start:(row_start + length), col_start:(col_start + length)]

    labels = io.imread(os.path.join(img_dir, 'labels_cropped.tiff'))
    labels_inset = labels[row_start:(row_start + length), col_start:(col_start + length)]

    io.imsave(os.path.join(img_dir, 'DNA_inset_{}.tiff'.format(inset_num)), DNA_inset)
    io.imsave(os.path.join(img_dir, 'Membrane_inset_{}.tiff'.format(inset_num)), Membrane_inset)
    io.imsave(os.path.join(img_dir, 'labels_inset_{}.tiff'.format(inset_num)), labels_inset)

    inset_mask = np.zeros(DNA.shape, dtype='uint8')
    inset_mask[row_start - thickness: row_start + thickness, col_start:(col_start + length)] = 128
    inset_mask[row_start + length - thickness:row_start + length + thickness,
               col_start:(col_start + length)] = 128

    inset_mask[row_start:(row_start + length), col_start - thickness: col_start + thickness] = 128
    inset_mask[row_start:(row_start + length),
               col_start + length - thickness: col_start + length + thickness] = 128

    io.imsave(os.path.join(img_dir, 'labels_inset_mask_{}.tiff'.format(inset_num)), inset_mask)


def generate_RGB_image(red=None, green=None, blue=None, percentile_cutoffs=(5, 95)):

    if red is None:
        red = np.zeros_like(green)

    combined = np.stack((red, green, blue), axis=-1)

    rgb_output = np.zeros(combined.shape, dtype='float32')

    # rescale each channel
    for idx in range(combined.shape[2]):
        if np.max(combined[:, :, idx]) == 0:
            # don't need to rescale this channel
            pass
        else:
            percentiles = np.percentile(combined[:, :, idx][combined[:, :, idx] > 0],
                                        [percentile_cutoffs[0], percentile_cutoffs[1]])
            rescaled_intensity = rescale_intensity(combined[:, :, idx].astype('float32'),
                                                   in_range=(percentiles[0], percentiles[1]),
                                                   out_range='float32')
            rgb_output[:, :, idx] = rescaled_intensity
    return rgb_output


def calculate_human_f1_scores(image_list):
    """Computes pairwise F1 scores from labeled images

    Args:
        image_list: list of predicted labels

    Returns:
        list: f1 scores for images
    """

    f1_list = []
    # loop over images to get first image
    for img1_idx in range(len(image_list) - 1):
        img1 = image_list[img1_idx]
        img1 = np.expand_dims(img1, axis=0)

        # loop over subsequent images to get corresponding predicted image
        for img2_idx in range(img1_idx + 1, len(image_list)):
            img2 = image_list[img2_idx]
            img2 = np.expand_dims(img2, axis=0)
            m = Metrics('human vs human', seg=False)
            m.calc_object_stats(y_true=img1, y_pred=img2)
            recall = m.stats['correct_detections'].sum() / m.stats['n_true'].sum()
            precision = m.stats['correct_detections'].sum() / m.stats['n_pred'].sum()
            f1 = 2 * precision * recall / (precision + recall)
            f1_list.append(f1)

    return f1_list


def calculate_alg_f1_scores(image_list, alg_pred):
    """Compare human annotations with algorithm for a given FOV

    Args:
        image_list: list of annotations from different human labelers
        alg_pred: prediction from alogrithm

    Returns:
        list: f1 scores for humans vs alg
    """

    f1_list_alg = []
    for true_img in image_list:
        true_img = np.expand_dims(true_img, axis=0)
        m = Metrics('human vs alg', seg=False)
        m.calc_object_stats(y_true=true_img, y_pred=alg_pred)
        recall = m.stats['correct_detections'].sum() / m.stats['n_true'].sum()
        precision = m.stats['correct_detections'].sum() / m.stats['n_pred'].sum()
        f1 = 2 * precision * recall / (precision + recall)
        f1_list_alg.append(f1)

    return f1_list_alg


def plot_heatmap(vals, x_labels, y_labels, title, cmap='gist_heat', save_path=None):

    fig, ax = plt.subplots()
    im = ax.imshow(vals, cmap=cmap)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            text = ax.text(j, i, vals[i, j],
                           ha="center", va="center", color="w")

    ax.set_title(title)
    fig.tight_layout()
    plt.show()

    if save_path:
        fig.imsave(save_path)
