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

# morphology comparisons
data_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/20200915_test_split/'
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/analyses/20200831_figure_4'
true_dict = np.load(os.path.join(data_dir, '20200908_multiplex_nuclear_test_256x256_subset.npz'))
true_labels = true_dict['y'].astype('int16')
true_cell_labels = true_labels[..., :1]
true_nuc_labels = true_labels[..., 1:]

tissue_list = true_dict['tissue_list']
platform_list = true_dict['platform_list']
pred_nuc_expansion_labels = np.load(os.path.join(data_dir, 'predicted_labels_nuc_expansion.npz'))['y']
pred_cell_labels = np.load(os.path.join(data_dir, 'predicted_labels_cell.npz'))['y']
pred_nuc_labels = np.load(os.path.join(data_dir, 'predicted_labels_nuc.npz'))['y']

properties = ['label', 'area', 'major_axis_length', 'perimeter', 'minor_axis_length', 'centroid']

cell_prop_df = figures.compute_morphology_metrics(true_cell_labels, pred_cell_labels,
                                                  properties=properties)

nuc_expansion_prop_df = figures.compute_morphology_metrics(true_cell_labels,
                                                           pred_nuc_expansion_labels,
                                                           properties=properties)

new_col_names = [name + '_nuc' for name in nuc_expansion_prop_df.columns.values]
nuc_expansion_prop_df.columns = new_col_names

combined = pd.concat((cell_prop_df, nuc_expansion_prop_df), axis=1)

# binned  accuracy
max_val = 3000
bin_size = 300

combined['bin'] = 10
for i in range(10):
    bin_start = i * bin_size
    bin_end = (i + 1) * bin_size
    idx = np.logical_and(combined['area_true'] > bin_start, combined['area_true'] <= bin_end)
    combined.loc[idx, 'bin'] = i


combined['pred_log2_nuc'] = np.log2(combined['area_pred_nuc'].values / combined['area_true_nuc'].values)
combined['pred_log2'] = np.log2(combined['area_pred'].values / combined['area_true'].values)

nonzero_idx = np.logical_and(combined['area_pred'] > 0, combined['area_pred_nuc'] > 0)

plot_df = combined.loc[nonzero_idx]
plot_df_long = pd.melt(plot_df, id_vars=['bin'], value_vars=['pred_log2', 'pred_log2_nuc'])
g = sns.catplot(data=plot_df_long,
                kind='violin', x='bin', y='value', hue='variable',
                order=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

x = np.linspace(0, 10, 10)
y = np.repeat(0, 10)
g = sns.lineplot(x, y, style=True, dashes=[(2,2)])
plt.savefig(os.path.join(base_dir, 'Cell_Area_Accuracy_Violin_combined.pdf'))



# N/C ratio
roshan_idx = np.logical_and(tissue_list == 'gi', platform_list == 'mibi')
# take first 5 images, curate nuclear annotations
roshan_cell_true = true_labels[roshan_idx]
roshan_cell_pred = cell_labels[roshan_idx]
roshan_nuc_pred = nuc_labels[roshan_idx]
channel_data = true_dict['X'][:660][roshan_idx]

for i in range(5):
    X = channel_data[[i], ...]
    y = np.zeros((1, 256, 256, 1))
    np.savez_compressed(os.path.join(base_dir, 'curated_nuc', 'example_fov_{}.npz'.format(i)),
                                     X=X, y=y)

roshan_nuc_true = np.zeros_like(roshan_nuc_pred)
for i in range(5):
    current_npz = np.load(os.path.join(base_dir, 'curated_nuc',
                                       'example_fov_{}_save_version_0.npz'.format(i)))
    labels = current_npz['y']
    roshan_nuc_true[i, ...] = labels[0, ...]


def generate_segmented_data(labels_tuple):
    # segment data
    label_xr = xr.DataArray(np.concatenate(labels_tuple, axis=-1),
                            coords=[range(labels_tuple[0].shape[0]), range(256), range(256),
                                    ['whole_cell', 'nuclear']],
                            dims=['fovs', 'rows', 'cols', 'compartments'])

    channel_xr = xr.DataArray(np.full_like(labels_tuple[0], 1),
                              coords=[range(labels_tuple[0].shape[0]),
                                      range(256), range(256),
                                      ['example_channel']],
                              dims=['fovs', 'rows', 'cols', 'channels'])
    normalized, _, _ = marker_quantification.generate_expression_matrix(
        segmentation_labels=label_xr,
        image_data=channel_xr,
        nuclear_counts=True)

    return normalized


def calc_nuc_dist(cell_centroids, nuc_centroids):
    x1s, y1s = cell_centroids
    x2s, y2s = nuc_centroids
    dist = np.sqrt((x2s - x1s) ** 2 + (y2s - y1s) ** 2)
    return dist


nc_df = pd.DataFrame()
for i in range(5):
    true_marker_counts = generate_segmented_data((roshan_cell_true[[i]], roshan_nuc_true[[i]]))
    pred_marker_counts = generate_segmented_data((roshan_cell_pred[[i]], roshan_nuc_pred[[i]]))

    true_nc = true_marker_counts['area_nuclear'] / true_marker_counts['area']
    true_label = true_marker_counts['label']

    pred_nc = pred_marker_counts['area_nuclear'] / pred_marker_counts['area']
    pred_label = pred_marker_counts['label']

    pred_skew = calc_nuc_dist((pred_marker_counts['centroid-0'].values,
                               pred_marker_counts['centroid-1'].values),
                              (pred_marker_counts['centroid-0_nuclear'].values,
                               pred_marker_counts['centroid-1_nuclear'].values)
                              )
    pred_skew /= pred_marker_counts['major_axis_length'].values

    true_skew = calc_nuc_dist((true_marker_counts['centroid-0'].values,
                               true_marker_counts['centroid-1'].values),
                              (true_marker_counts['centroid-0_nuclear'].values,
                               true_marker_counts['centroid-1_nuclear'].values)
                              )
    true_skew /= true_marker_counts['major_axis_length'].values


    true_df = pd.DataFrame({'label': true_label, 'nc_ratio': true_nc, 'nuc_skew': true_skew})
    pred_df = pd.DataFrame({'label': pred_label, 'nc_ratio': pred_nc, 'nuc_skew': pred_skew})

    true_ids, pred_ids = figures.get_paired_cell_ids(true_label=roshan_cell_true[i, :, :, 0],
                                                     pred_label=roshan_cell_pred[i, :, :, 0])

    paired_df = figures.get_paired_metrics(true_ids=true_ids, pred_ids=pred_ids,
                                           true_metrics=true_df,
                                           pred_metrics=pred_df)
    paired_df['fov'] = i

    nc_df = nc_df.append(paired_df)

# NC ratio
nc_df_plot = copy.copy(nc_df)
nc_df_plot.loc[nc_df_plot['nc_ratio_pred'] > 1, 'nc_ratio_pred'] = 1
nc_df_plot = nc_df_plot.loc[nc_df_plot['nc_ratio_pred'] > 0, :]

skew_df = copy.copy(nc_df)
skew_df = skew_df.loc[skew_df['nc_ratio_true'] > 0, :]
skew_df = skew_df.loc[skew_df['nc_ratio_pred'] > 0, :]


fig, ax = plt.subplots()
figures.create_density_scatter(ax, nc_df_plot['nc_ratio_true'].values, nc_df_plot['nc_ratio_pred'].values)
figures.label_morphology_scatter(ax, nc_df_plot['nc_ratio_true'].values, nc_df_plot['nc_ratio_pred'].values)
ax.set_title('NC Ratio Accuracy')
fig.savefig(os.path.join(base_dir, 'NC_ratio_Accuracy.pdf'))


# nuclear skew
fig, ax = plt.subplots()
figures.create_density_scatter(ax, skew_df['nuc_skew_true'].values, skew_df['nuc_skew_pred'].values)
ax.set_title('Nuc Skew Accuracy')
fig.savefig(os.path.join(base_dir, 'Nuc_skew_accuracy.pdf'))

# move potential images
img_list = ['CD44.tif', 'COX2.tif', 'ECAD.tif', 'GLUT1.tif', 'HER2.tif', 'HH3.tif',
            'Ki67.tif', 'P.tif', 'PanKRT.tif', 'pS6.tif']

folders = io_utils.list_folders(base_dir, 'Point')

for folder in folders:
    potential = os.path.join(base_dir, folder, 'potential_channels')
    os.makedirs(potential)
    for img in img_list:
        shutil.copy(os.path.join(base_dir, folder, img), os.path.join(potential, img))

fovs = io_utils.list_folders(base_dir, 'Point')

# copy selected membrane channel to membrane.tiff to make data loading easier
for fov in fovs:
    img_folder = os.path.join(base_dir, fov, 'segmentation_channels')
    imgs = io_utils.list_files(img_folder, '.tif')
    imgs.pop(np.where(np.isin(imgs, 'HH3.tif'))[0][0])

    shutil.copy(os.path.join(img_folder, imgs[0]),
                os.path.join(img_folder, 'membrane.tiff'))

channel_data = data_utils.load_imgs_from_tree(base_dir, img_sub_folder='segmentation_channels',
                                              channels=['HH3.tif', 'membrane.tiff'])

channel_data.to_netcdf(base_dir + 'deepcell_input.xr', format='NETCDF3_64BIT')

# Since each point has different channels, we need to segment them one at a time
segmentation_labels = xr.open_dataarray(base_dir + '/segmentation_labels_combined.xr')

core_df = pd.DataFrame()

for fov in fovs:
    channel_data = data_utils.load_imgs_from_tree(base_dir, fovs=[fov],
                                                  img_sub_folder='potential_channels')

    current_labels = segmentation_labels.loc[[fov], :, :, :]

    normalized, transformed, raw = marker_quantification.generate_expression_matrix(
        segmentation_labels=current_labels,
        image_data=channel_data,
        nuclear_counts=True
    )
    core_df = core_df.append(raw, sort=False)

core_df.to_csv(os.path.join(base_dir, 'single_cell_data.csv'))

# save segmentation mask outlines
for idx, fov in enumerate(segmentation_labels.fovs.values):
    nuc_label = segmentation_labels.loc[fov, :, :, 'nuclear']
    cell_label = segmentation_labels.loc[fov, :, :, 'whole_cell']

    nuc_boundary = find_boundaries(nuc_label.values, mode='inner').astype('uint8')
    nuc_boundary[nuc_boundary > 0] = 255
    cell_boundary = find_boundaries(cell_label.values, mode='inner').astype('uint8')
    cell_boundary[cell_boundary > 0] = 255

    io.imsave(os.path.join(base_dir, fov, 'nuc_boundary.tiff'), nuc_boundary)
    io.imsave(os.path.join(base_dir, fov, 'cell_boundary.tiff'), cell_boundary)

# read in segmented data
cell_counts = pd.read_csv(os.path.join(base_dir, 'single_cell_data.csv'))
cell_counts = cell_counts.loc[cell_counts['cell_size_nuclear'] > 20, :]


channels = np.array(['CD44', 'ECAD', 'GLUT1', 'HER2', 'HH3', 'Ki67', 'P', 'PanKRT', 'pS6'])
nuc_frac = []

# compute nuclear fraction
for i in range(len(channels)):
    chan_name = channels[i]
    channel_counts = cell_counts.loc[:, [chan_name, chan_name + '_nuclear']]
    cutoff = np.percentile(cell_counts.values[cell_counts.values[:, 0] > 0, 0], [10])
    channel_counts = channel_counts.loc[channel_counts[chan_name] > cutoff[0], :]

    ratio = channel_counts.values[:, 1] / channel_counts.values[:, 0]
    avg_ratio = np.mean(ratio)
    nuc_frac.append(avg_ratio)

cell_frac = [1 - nuc for nuc in nuc_frac]

# sort by increasing cell fraction
sort_idx = np.argsort(cell_frac)
channels, nuc_frac, cell_frac = channels[sort_idx], np.array(nuc_frac)[sort_idx], np.array(cell_frac)[sort_idx]
plt.style.use('/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/code/ark-analysis/ark/test_stylesheet.mplstyle')
plt.style.use('seaborn-colorblind')

fig, ax = plt.subplots()
width = 0.35
ax.bar(channels, nuc_frac, label='Nuclear Fraction')
ax.bar(channels, cell_frac, bottom=nuc_frac, label='Cell Fraction')
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.savefig(base_dir + 'subcellular_barchart.pdf')


# missing signal quantification
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/datasets/20200811_tyler_phenotyping/'
fovs = io_utils.list_folders(base_dir, 'Point')

# Since each point has different channels, we need to segment them one at a time
segmentation_labels_cell = xr.open_dataarray(base_dir + '/segmentation_labels_combined.xr')
segmentation_labels_expanded = xr.open_dataarray(base_dir + '/segmentation_labels_nuc_expansion.xr')
segmentation_labels_cell[..., 1] - segmentation_labels_expanded
marker_counts_df = pd.DataFrame()

for fov in fovs:
    channel_data = data_utils.load_imgs_from_tree(base_dir, fovs=[fov],
                                                  img_sub_folder='potential_channels')

    current_labels = segmentation_labels.loc[[fov], :, :, :]

    fov_counts = {}
    for channel in channel_data.channels.values:
        channel_img = channel_data.loc[fov, :, :, channel].values
        total_counts = np.sum(channel_img)
        cell_mask = current_labels.values[0, :, :, 0] > 0
        cell_counts = np.sum(channel_img[cell_mask])
        nuc_mask = current_labels.values[0, :, :, 1] > 0
        nuc_counts = np.sum(channel_img[nuc_mask])

        fov_counts[channel + '_total'] = total_counts
        fov_counts[channel + '_cell'] = cell_counts
        fov_counts[channel + '_nuc'] = nuc_counts


    marker_counts_df = marker_counts_df.append(pd.DataFrame(fov_counts, index=[fov]))

base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/analyses/20200831_figure_4/'
marker_counts_df.to_csv(os.path.join(base_dir, 'total_counts_by_mask.csv'))

marker_counts_df = pd.read_csv(os.path.join(base_dir, 'total_counts_by_mask.csv'))

summed_counts = marker_counts_df.sum(axis=0)

channels = np.array(['CD44', 'ECAD', 'GLUT1', 'HER2', 'HH3', 'Ki67', 'P', 'PanKRT', 'pS6'])

nuclear_fraction = []
for chan in channels:
    cell_frac = summed_counts[chan + '_total']
    nuc_frac = summed_counts[chan + '_nuc']
    frac = nuc_frac / cell_frac
    nuclear_fraction.append(frac)


# sort by decreasing nuc fraction
sort_idx = np.argsort([-item for item in nuclear_fraction])
channels, nuclear_fraction = channels[sort_idx], np.array(nuclear_fraction)[sort_idx]

fig, ax = plt.subplots()
width = 0.35
ax.bar(channels, nuclear_fraction, label='Nuclear Fraction')
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.savefig(base_dir + 'missed_signal.pdf')


# Number of cells without a nucleus across tissue types
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/analyses/20200831_figure_4'
nuc_labels = np.load(os.path.join(base_dir, 'predicted_labels_nuc.npz'))['y']
cell_labels = np.load(os.path.join(base_dir, 'predicted_labels_cell.npz'))['y']
true_dict = np.load(os.path.join(base_dir, '20200908_multiplex_test_256x256.npz'))
true_labels = true_dict['y'].astype('int16')[:660]
tissue_list = true_dict['tissue_list'][:660]
platform_list = true_dict['platform_list'][:660]

combined_labels = xr.DataArray(np.concatenate((cell_labels, nuc_labels), axis=-1),
                               coords=[range(nuc_labels.shape[0]), range(256), range(256),
                                             ['whole_cell', 'nuclear']],
                               dims=['fovs', 'rows', 'cols', 'compartments'])

blank_channel_data = xr.DataArray(np.full_like(cell_labels, 1),
                                  coords=[range(cell_labels.shape[0]), range(256), range(256),
                                          ['example_channel']],
                                  dims=['fovs', 'rows', 'cols', 'channels'])
normalized, _, _ = marker_quantification.generate_expression_matrix(
    segmentation_labels=combined_labels,
    image_data=blank_channel_data,
    nuclear_counts=True)

anuclear_fraction = []
for fov in np.unique(normalized['fov']):
    current_counts = normalized.loc[normalized['fov'] == fov]
    anucleated_count = np.sum(current_counts['area_nuclear'] == 0)
    total_count = len(current_counts)
    anuclear_fraction.append(anucleated_count/total_count)

anuclear_df = pd.DataFrame({'anuclear_frac': anuclear_fraction, 'tissue': tissue_list})
sums = anuclear_df.groupby('tissue')['anuclear_frac'].mean()

# sort by decreasing nuclear counts
anuclear_counts = sums.values
anuclear_tissue = sums.index.values
sort_idx = np.argsort(anuclear_counts)
anuclear_counts, anuclear_tissue = anuclear_counts[sort_idx], anuclear_tissue[sort_idx]

fig, ax = plt.subplots()
width = 0.35
ax.bar(anuclear_tissue, anuclear_counts, label='Nuclear Fraction')
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_title('Fraction anuclear cells')
fig.savefig(base_dir + '/anuclear_cell_count.pdf')