import os
import shutil
import xarray as xr

import matplotlib.pyplot as plt
from matplotlib import cm

import skimage.io as io

import scipy.ndimage as nd
import numpy as np
import pandas as pd

from ark.utils import plot_utils
from skimage.segmentation import find_boundaries
from skimage.exposure import rescale_intensity

from skimage.measure import regionprops_table
from skimage.transform import resize
from sklearn.metrics import r2_score
from scipy.stats import pearsonr


from deepcell_toolbox.metrics import Metrics

from ark import figures
from ark.utils import data_utils, segmentation_utils, io_utils
from ark.segmentation import marker_quantification

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Figure 2
data_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/20200820_figure_2_data/'
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/analyses/20200820_figure_2_overlays'

# extract TIFs and labels from xarrays
raw_data = xr.load_dataarray(os.path.join(data_dir, 'deepcell_input.xr'))
cell_labels = xr.load_dataarray(os.path.join(data_dir, 'segmentation_labels_cell.xr'))
nuc_labels = xr.load_dataarray(os.path.join(data_dir, 'segmentation_labels_nuc.xr'))

# extract files from arrays
for fov in raw_data.fovs.values:
    fov_folder = os.path.join(base_dir, fov)
    os.makedirs(fov_folder)

    io.imsave(os.path.join(fov_folder, 'DNA.tiff'), raw_data.loc[fov, :, :, 'HH3'].astype('int16'))
    io.imsave(os.path.join(fov_folder, 'Membrane.tiff'), raw_data.loc[fov, :, :, 'Membrane'].astype('int16'))
    io.imsave(os.path.join(fov_folder, 'cell_labels.tiff'), cell_labels.loc[fov, :, :, 'whole_cell'])
    io.imsave(os.path.join(fov_folder, 'nuc_labels.tiff'), nuc_labels.loc[fov, :, :, 'nuclear'])


for idx, fov in enumerate(raw_data.fovs.values):
    DNA = io.imread(os.path.join(base_dir, fov, 'DNA.tiff'))
    Membrane = io.imread(os.path.join(base_dir, fov, 'Membrane.tiff'))
    cell_label = io.imread(os.path.join(base_dir, fov, 'cell_labels.tiff'))
    nuc_label = io.imread(os.path.join(base_dir, fov, 'nuc_labels.tiff'))

    rgb_image = figures.generate_RGB_image(red=None, blue=DNA,
                                           green=Membrane,
                                           percentile_cutoffs=(0, 100))

    # generate label images
    label_map_cell = figures.color_labels_by_graph(cell_label)
    label_map_cell = figures.recolor_labels(label_map_cell)

    label_map_nuc = figures.color_labels_by_graph(nuc_label)
    label_map_nuc = figures.recolor_labels(label_map_nuc)

    io.imsave(os.path.join(base_dir, fov, 'rgb_image.tiff'), rgb_image)
    io.imsave(os.path.join(base_dir, fov, 'greyscale_cell_label_map.tiff'),
              label_map_cell.astype('uint8'))

    io.imsave(os.path.join(base_dir, fov, 'greyscale_nuc_label_map.tiff'),
              label_map_nuc.astype('uint8'))

# specify crops for each image
row_idx_list = [[250, 550], [300, 600], [200, 500]]
col_idx_list = [[50, 250], [100, 300], [400, 600]]

# create crops
selected_fovs = ['20200116_DCIS_Point2304', 'P101_T3T4_Point18', 'hiv_Point12']
for idx, fov in enumerate(selected_fovs):
    DNA = io.imread(os.path.join(base_dir, fov, 'DNA.tiff'))
    Membrane = io.imread(os.path.join(base_dir, fov, 'Membrane.tiff'))
    label_map_nuc = io.imread(os.path.join(base_dir, fov, 'greyscale_nuc_label_map.tiff'))
    label_map_cell = io.imread(os.path.join(base_dir, fov, 'greyscale_cell_label_map.tiff'))
    rgb_image = io.imread(os.path.join(base_dir, fov, 'rgb_image.tiff'))

    # float32 images
    imgs = [DNA, Membrane, rgb_image]
    names = ['DNA', 'Membrane', 'rgb_image']

    row_start, row_end = row_idx_list[idx][0], row_idx_list[idx][1]
    col_start, col_end = col_idx_list[idx][0], col_idx_list[idx][1]
    for img, name in zip(imgs, names):
        cropped = img[row_start:row_end, col_start:col_end]
        io.imsave(os.path.join(base_dir, fov, name + '_cropped.tiff'), cropped)

    imgs_uint8 = [label_map_cell, label_map_nuc]
    names_uint8 = ['cell_label_map', 'nuc_label_map']
    for img, name in zip(imgs_uint8, names_uint8):
        cropped = img[row_start:row_end, col_start:col_end]
        io.imsave(os.path.join(base_dir, fov, name + '_cropped.tiff'), cropped.astype('uint8'))

    # phenotyping markers
    imgs = io_utils.list_files(os.path.join(base_dir, fov, 'phenotyping'))
    for img in imgs:
        current_img = io.imread(os.path.join(base_dir, fov, 'phenotyping', img))
        current_img = current_img[row_start:row_end, col_start:col_end]
        io.imsave(os.path.join(base_dir, fov, 'phenotyping', img + '_cropped.tiff'), current_img)

# accuracy plots
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/analyses/20200828_figure_2_accuracy/'
whole_cell = np.load(base_dir + 'MIBI_accuracy_whole_cell.npz', allow_pickle=True)['stats'].item()['all']
nuclear_expansion = np.load(base_dir + 'MIBI_accuracy_nuclear_expansion.npz', allow_pickle=True)['stats'].item()['all']

plotting_keys = ['missed_detections', 'gained_detections', 'merge', 'split', 'catastrophe']
error_dict = {key: [whole_cell[key], nuclear_expansion[key]] for key in plotting_keys}

error_arr = pd.DataFrame(error_dict)
error_arr['algorithm'] = ['whole-cell', 'nuclear-expansion']
error_arr_long = pd.melt(error_arr, id_vars='algorithm')

g = sns.catplot(data=error_arr_long,
                kind='bar', x='variable', y='value', hue='algorithm')
plt.savefig(os.path.join(base_dir, 'Error_Types_new.pdf'), transparent=True)


fig, axes = plt.subplots(1, len(plotting_keys), figsize=(15, 4))
figures.plot_error_types(axes, (whole_cell_plotting, nuclear_plotting),
                         method_labels=['whole_cell', 'nuclear_expansion'],
                         error_labels=plotting_keys,
                         colors=['blue', 'red'], ylim=(0, 30000))
fig.savefig(os.path.join(base_dir, 'error_metrics.pdf'))
fig, ax = plt.subplots(figsize=(5, 5))



f1_scores = [whole_cell['f1'], nuclear_expansion['f1']]
f1_array = pd.DataFrame({'variable': ['f1', 'f1'], 'value': f1_scores, 'algorithm': ['whole_cell', 'nuclear_expansion']})
g = sns.catplot(data=f1_array,
                kind='bar', x='variable', y='value', hue='algorithm')
plt.savefig(os.path.join(base_dir, 'F1_scores_new.pdf'), transparent=True)


fig.savefig(base_dir + 'f1_score.tiff')


# missing signal on a per-cell basis:
data_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/20200811_tyler_phenotyping/'
plot_dir = ''
segmentation_labels_cell = xr.open_dataarray(data_dir + '/segmentation_labels_cell.xr')
segmentation_labels_expanded = xr.open_dataarray(data_dir + '/segmentation_labels_nuc_expanded.xr')
segmentation_labels_combined = xr.DataArray(np.concatenate((segmentation_labels_cell.values,
                                                      segmentation_labels_expanded.values),
                                                     axis=-1),
                                            coords=[segmentation_labels_cell.fovs,
                                                    segmentation_labels_cell.rows,
                                                    segmentation_labels_cell.cols,
                                                    ['whole_cell', 'nuclear']],
                                            dims=segmentation_labels_cell.dims)
fovs = io_utils.list_folders(base_dir, 'Point')
missing_signal_df = pd.DataFrame()
for fov in fovs:
    channel_data = data_utils.load_imgs_from_tree(base_dir, fovs=[fov],
                                                  img_sub_folder='potential_channels')

    current_labels = segmentation_labels_combined.loc[[fov], :, :, :]

    normalized, transformed, raw = marker_quantification.generate_expression_matrix(
        segmentation_labels=current_labels,
        image_data=channel_data,
        nuclear_counts=True
    )
    missing_signal_df = missing_signal_df.append(raw, sort=False)

missing_signal_df.to_csv(os.path.join(base_dir, 'missing_signal.csv'))

missing_signal_df = pd.read_csv(os.path.join(base_dir, 'missing_signal.csv'))

# remove cells without a predicted nucleus
missing_signal_df = missing_signal_df.loc[missing_signal_df['area_nuclear'] > 0, :]
channels = np.array(['CD44', 'ECAD', 'GLUT1', 'HER2', 'HH3', 'Ki67', 'P', 'PanKRT'])

# create plotting df
plotting_df = pd.DataFrame()
for chan in channels:
    # compute ratio of cell to nuclear values
    ratio = missing_signal_df[chan].values / missing_signal_df[chan + '_nuclear'].values

    # cells without nuclear counts (divide by zero) become ratio of 10
    ratio[missing_signal_df[chan + '_nuclear'] == 0] = 10

    # only keep cells that are in top 90% for marker expression
    cell_counts = missing_signal_df[chan].values
    cutoff = np.percentile(cell_counts[cell_counts > 0], [10])
    idx = cell_counts > cutoff[0]

    ratio = ratio[idx]

    # cap maximum at 10
    ratio[ratio > 10] = 10
    current_df = pd.DataFrame({'marker': chan, 'ratio': ratio})
    plotting_df = plotting_df.append(current_df)

median_array = plotting_df.groupby('marker').median()
median_vals = median_array.values[:, 0]
median_names = median_array.index.values

idx = np.argsort(median_vals)
sns.catplot(x='marker', y='ratio', data=plotting_df, kind='box', order=median_names[idx])

plt.savefig(os.path.join(base_dir, 'signal_extraction_proportion.pdf'))
