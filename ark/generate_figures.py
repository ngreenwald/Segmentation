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

# Figure 1
# Compute total annotator time
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/caliban_files/'
tissue_types = io_utils.list_folders(base_dir, substrs='20')

total_time = 0
for tissue in tissue_types:
    job_folders = io_utils.list_folders(os.path.join(base_dir, tissue))
    for job in job_folders:
        job_report = os.path.join(base_dir, tissue, job, 'logs/job_report.csv')
        if os.path.exists(job_report):
            job_csv = pd.read_csv(job_report)
            job_time = figures.calculate_annotator_time(job_report=job_csv)
            total_time += job_time
        else:
            print('No file found in {}'.format(os.path.join(tissue, job)))

internal_time = pd.read_csv('/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/annotation_hours.csv')
internal_hours = np.sum(internal_time['Total Hours'])

fig, ax = plt.subplots(figsize=(3, 3))
figures.barchart_helper(ax=ax, values=[total_time / 3600, internal_hours],
                        labels=['Annotators', 'Internal'],
                        title='Total hours',
                        colors='blue', y_max=np.max(total_time / 3600) * 1.2)
fig.tight_layout()
fig.savefig(os.path.join('/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/analyses/20200830_figure_1', 'hours.tiff'))

# counts by modality
tissue_counts = np.load('/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/analyses/20200830_figure_1/tissue_counts.npz',
                        allow_pickle=True)['stats'].item()

tissue_counts = pd.DataFrame(tissue_counts)
fig, ax = plt.subplots(figsize=(5, 5))
figures.barchart_helper(ax=ax, values=tissue_counts.iloc[0, :].values,
                        labels=tissue_counts.columns.values,
                        title='Cells per tissue type', colors='blue',
                        y_max=250000)
fig.tight_layout()
fig.savefig(os.path.join('/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/analyses/20200830_figure_1', 'annotations_per_tissue.tiff'))

# counts by modality
platform_counts = np.load('/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/analyses/20200830_figure_1/platform_counts.npz',
                        allow_pickle=True)['stats'].item()

platform_counts = pd.DataFrame(platform_counts)
fig, ax = plt.subplots(figsize=(5, 5))
figures.barchart_helper(ax=ax, values=platform_counts.iloc[0, :].values,
                        labels=platform_counts.columns.values,
                        title='Cells per platform type', colors='blue',
                        y_max=250000)
fig.tight_layout()
fig.savefig(os.path.join('/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/analyses/20200830_figure_1', 'annotations_per_platform.tiff'))

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
whole_cell_plotting = {key: whole_cell[key] for key in plotting_keys}
nuclear_plotting = {key: nuclear_expansion[key] for key in plotting_keys}

figures.plot_error_types((whole_cell_plotting, nuclear_plotting),
                         method_labels=['whole_cell', 'nuclear_expansion'],
                         error_labels=plotting_keys,
                         colors=['blue', 'red'], ylim=(0, 30000))

fig, ax = plt.subplots(figsize=(5, 5))

f1_scores = [whole_cell['f1'], nuclear_expansion['f1']]
figures.barchart_helper(ax=ax, values=f1_scores, labels=['whole_cell', 'nuclear'],
                        title='F1 score', colors=['blue', 'red'], y_lim=(0, 1))

fig.savefig(base_dir + 'f1_score.tiff')
# Figure 3

# Make sure all data is max 1024x1024
data_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/20200820_figure_3_data'
folders = io_utils.list_folders(data_dir)
for folder in folders:
    folder_path = os.path.join(data_dir, folder)
    DNA = io.imread(os.path.join(folder_path, 'DNA.tiff'))
    Membrane = io.imread(os.path.join(folder_path, 'Membrane.tiff'))
    io.imsave(os.path.join(folder_path, 'DNA_cropped.tiff'), DNA[:1000, :1000])
    io.imsave(os.path.join(folder_path, 'Membrane_cropped.tiff'), Membrane[:1000, :1000])

# plotting
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/analyses/20200821_figure_3_overlays'

# extract TIFs and labels from xarrays
raw_data = xr.load_dataarray(os.path.join(base_dir, 'deepcell_input.xr'))
labels = xr.load_dataarray(os.path.join(base_dir, 'segmentation_labels.xr'))

# extract files from arrays
for fov in raw_data.fovs.values:
    fov_folder = os.path.join(base_dir, fov)
    os.makedirs(fov_folder)

    io.imsave(os.path.join(fov_folder, 'DNA.tiff'), raw_data.loc[fov, :, :, 'DNA_cropped'].astype('float32'))
    io.imsave(os.path.join(fov_folder, 'Membrane.tiff'), raw_data.loc[fov, :, :, 'Membrane_cropped'].astype('float32'))
    io.imsave(os.path.join(fov_folder, 'labels.tiff'), labels.loc[fov, :, :, 'whole_cell'].astype('int16'))

for idx, fov in enumerate(raw_data.fovs.values):
    folder_path = os.path.join(base_dir, fov)
    figures.preprocess_overlays(folder_path)

    DNA = io.imread(os.path.join(folder_path, 'DNA_resized.tiff'))
    Membrane = io.imread(os.path.join(folder_path, 'Membrane_resized.tiff'))
    label = io.imread(os.path.join(folder_path, 'labels_resized.tiff'))
    boundary = io.imread(os.path.join(folder_path, 'boundaries_resized.tiff'))

    rgb_image = figures.generate_RGB_image(red=None, blue=DNA,
                                           green=Membrane,
                                           percentile_cutoffs=(0, 100))
    max_val = np.max(rgb_image)
    overlay = np.copy(rgb_image)
    overlay[boundary > 0, :] = max_val / 2

    io.imsave(os.path.join(folder_path, 'rgb_image_resized.tiff'), rgb_image)
    io.imsave(os.path.join(folder_path, 'rgb_overlay_resized.tiff'), overlay)

# # specify crops for each image
row_idx_list = [[400, 700]] * 12
col_idx_list = [[400, 700]] * 12

# create crops
for idx, fov in enumerate(raw_data.fovs.values):
    DNA = io.imread(os.path.join(base_dir, fov, 'DNA_resized.tiff'))
    Membrane = io.imread(os.path.join(base_dir, fov, 'Membrane_resized.tiff'))
    boundary = io.imread(os.path.join(base_dir, fov, 'boundaries_resized.tiff'))
    rgb_image = io.imread(os.path.join(base_dir, fov, 'rgb_overlay_resized.tiff'))

    imgs = [DNA, Membrane, boundary, rgb_image]
    names = ['DNA', 'Membrane', 'boundaries', 'rgb_image']

    row_start, row_end = row_idx_list[idx][0], row_idx_list[idx][1]
    col_start, col_end = col_idx_list[idx][0], col_idx_list[idx][1]
    for img, name in zip(imgs, names):
        cropped = img[row_start:row_end, col_start:col_end]
        io.imsave(os.path.join(base_dir, fov, name + '_resized_cropped.tiff'), cropped)



# Human Comparison
# save segmentation labels to each folder
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/20200413_Human_Agreement/'
prediction_xr = xr.open_dataarray(base_dir + 'segmentation_labels.xr')
for i in range(prediction_xr.shape[0]):
    prediction = prediction_xr.values[i, :, :, 0]
    io.imsave(os.path.join(base_dir, prediction_xr.fovs.values[i], 'segmentation_label.tiff'),
              prediction.astype('int16'))

# create list to hold f1 scores from each condition
f1_score_values = []
f1_score_labels = []
folders = list(prediction_xr.fovs.values)
folder_names = ['DCIS_MIBI', 'Colon_IF', 'Esophagus_MIBI', 'Hodgekins_Vectra']

for i in range(len(folders)):
    # get all of the human annotations
    folder_path = os.path.join(base_dir, folders[i], 'annotations')
    img_names = io_utils.list_files(folder_path, '.tiff')
    imgs = []
    for img in img_names:
        current_img = io.imread(os.path.join(folder_path, img))
        imgs.append(current_img)

    f1_scores_human = figures.calculate_human_f1_scores(image_list=imgs)
    f1_score_values.append(f1_scores_human)
    f1_score_labels.append(folder_names[i] + '_human')

    # compare algorithm
    pred_img = io.imread(os.path.join(base_dir, folders[i], 'segmentation_label.tiff'))
    pred_img = np.expand_dims(pred_img, axis=0)
    f1_scores_alg = figures.calculate_alg_f1_scores(image_list=imgs, alg_pred=pred_img)

    f1_score_values.append(f1_scores_alg)
    f1_score_labels.append(folder_names[i] + '_alg')


figures.plot_annotator_agreement(f1_scores_list=f1_score_values, labels=f1_score_labels)
plt.tight_layout()
plt.savefig('/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/analyses/20200720_lab_meeting/human_comparison.pdf', transparent=True)

# Accuracy heatmaps
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/analyses/20200825_figure_3_heatmaps/'
vectra = np.load(base_dir + 'vectra_accuracy.npz', allow_pickle=True)['stats'].item()
mibi = np.load(base_dir + 'mibi_accuracy.npz', allow_pickle=True)['stats'].item()
all_platform = np.load(base_dir + 'all_platform_accuracy.npz', allow_pickle=True)['stats'].item()

gi = np.load(base_dir + 'gi_accuracy.npz', allow_pickle=True)['stats'].item()
breast = np.load(base_dir + 'breast_accuracy.npz', allow_pickle=True)['stats'].item()
pancreas = np.load(base_dir + 'pancreas_accuracy.npz', allow_pickle=True)['stats'].item()
all_tissue = np.load(base_dir + 'all_tissue_accuracy.npz', allow_pickle=True)['stats'].item()

tissue_types = ['gi', 'breast', 'pancreas', 'all']
save_path = os.path.join(base_dir, 'tissue_heatmap.tiff')
platform_array = figures.create_f1_score_grid([gi, breast, pancreas, all_tissue], tissue_types)
figures.plot_heatmap(vals=platform_array.values, x_labels=tissue_types, y_labels=tissue_types,
                     title='accuracy across tissue types',
                     save_path=os.path.join(base_dir, 'tissue_heatmap.tiff'), cmap='Reds')


platform_types = ['vectra', 'mibi', 'all']
save_path = os.path.join(base_dir, 'platform_heatmap.tiff')
platform_array = figures.create_f1_score_grid([vectra, mibi, all_platform], platform_types)
figures.plot_heatmap(vals=platform_array.values, x_labels=platform_types, y_labels=platform_types,
                     title='accuracy across platforms',
                     save_path=os.path.join(base_dir, 'platform_heatmap.tiff'), cmap='Reds')


# Figure 4

# morphology comparisons
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/analyses/20200831_figure_4'
true_dict = np.load(os.path.join(base_dir, '20200908_multiplex_test_no_resize_256x256.npz'))
true_labels = true_dict['y'].astype('int16')
tissue_list = true_dict['tissue_list']
platform_list = true_dict['platform_list']
nuc_labels = np.load(os.path.join(base_dir, 'predicted_labels_nuc.npz'))['y']
cell_labels = np.load(os.path.join(base_dir, 'predicted_labels_cell.npz'))['y']

properties_df = pd.DataFrame()
properties = ['label', 'area', 'major_axis_length', 'perimeter', 'minor_axis_length']

gi_idx = np.isin(tissue_list, 'breast')
platform_idx = np.isin(platform_list, 'mibi')
idx = gi_idx * platform_idx
immune_idx = np.isin(tissue_list, 'immune')

cell_prop_df = compute_morphology_metrics(true_labels, cell_labels)
nuc_prop_df = compute_morphology_metrics(true_labels, nuc_labels)

cell_prop_df_cleaned = pd.DataFrame()
for img in np.unique(cell_prop_df['img_num'].values):
    current_df = cell_prop_df.loc[cell_prop_df['img_num'] == img]
    nuc_df = nuc_prop_df.loc[np.logical_and(nuc_prop_df['img_num'] == img, nuc_prop_df['area_pred'] > 0)]
    keep_idx = np.isin(current_df['label_true'].values, nuc_df['label_true'].values)
    current_df = current_df.loc[keep_idx, :]
    cell_prop_df_cleaned = cell_prop_df_cleaned.append(current_df)


def compute_morphology_metrics(true_labels, pred_labels):
    prop_df = pd.DataFrame()
    for idx in range(true_labels.shape[0]):
        pred_label = pred_labels[idx, :, :, 0]
        true_label = true_labels[idx, :, :, 0]

        true_ids, pred_ids = figures.get_paired_cell_ids(true_label=true_label,
                                                         pred_label=pred_label)

        true_props_table = pd.DataFrame(regionprops_table(true_label, properties=properties))
        pred_props_table = pd.DataFrame(regionprops_table(pred_label, properties=properties))

        paired_df = figures.get_paired_metrics(true_ids=true_ids, pred_ids=pred_ids,
                                               true_metrics=true_props_table,
                                               pred_metrics=pred_props_table)
        paired_df['img_num'] = idx
        prop_df = prop_df.append(paired_df)

    return prop_df


def get_skew_cells(input_df):
    ratio = input_df['major_axis_length_true'].values / input_df['minor_axis_length_true'].values
    skew_idx = np.logical_or(ratio < 0.6, ratio > 1.5)
    nonzero_idx = input_df['area_pred'] > 0
    combined_idx = skew_idx * nonzero_idx

    true_size = input_df['area_true'].values[combined_idx]
    pred_size = input_df['area_pred'].values[combined_idx]

    return true_size, pred_size


def get_round_cells(input_df):
    ratio = input_df['major_axis_length_true'].values / input_df['minor_axis_length_true'].values
    round_idx = np.logical_and(ratio > 0.8, ratio < 1.2)
    nonzero_idx = input_df['area_pred'] > 0
    combined_idx = round_idx * nonzero_idx

    true_size = input_df['area_true'].values[combined_idx]
    pred_size = input_df['area_pred'].values[combined_idx]

    return true_size, pred_size


def get_nonzero_cells(input_df):
    nonzero_idx = input_df['area_pred'] > 0
    true_size = input_df['area_true'].values[nonzero_idx]
    pred_size = input_df['area_pred'].values[nonzero_idx]

    return true_size, pred_size


true_size_cell_skew, pred_size_cell_skew = get_skew_cells(cell_prop_df_cleaned)
true_size_nuc_skew, pred_size_nuc_skew = get_skew_cells(nuc_prop_df)

true_size_cell_round, pred_size_cell_round = get_round_cells(cell_prop_df_cleaned)
true_size_nuc_round, pred_size_nuc_round = get_round_cells(nuc_prop_df)

true_size_cell, pred_size_cell = get_nonzero_cells(cell_prop_df_cleaned)
true_size_nuc, pred_size_nuc = get_nonzero_cells(nuc_prop_df)


fig, ax = plt.subplots()
figures.create_density_scatter(ax, true_size_nuc, pred_size_nuc)
figures.label_morphology_scatter(ax, true_size_nuc, pred_size_nuc)
ax.set_title('Nuc Area Accuracy')
ax.set_ylim(0, 6000)

fig.savefig(os.path.join(base_dir, 'Nuc_Area_all_scatter.pdf'))


# subcellular localization
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/analyses/20200811_subcellular_loc/DCIS/'
all_imgs = data_utils.load_imgs_from_tree(data_dir=base_dir)
stitched_imgs = data_utils.stitch_images(all_imgs, 5)

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
segmentation_labels = xr.open_dataarray(base_dir + '/segmentation_labels.xr')

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
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/analyses/20200811_subcellular_loc/DCIS/'
fovs = io_utils.list_folders(base_dir, 'Point')

# Since each point has different channels, we need to segment them one at a time
segmentation_labels = xr.open_dataarray(base_dir + '/segmentation_labels.xr')

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

# on a per-cell basis:
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/analyses/20200811_subcellular_loc/DCIS/'


# Number of cells without a nucleus across tissue types
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/analyses/20200831_figure_4'
nuc_labels = np.load(os.path.join(base_dir, 'predicted_labels_nuc.npz'))['y']
cell_labels = np.load(os.path.join(base_dir, 'predicted_labels_cell.npz'))['y']
true_dict = np.load(os.path.join(base_dir, '20200908_multiplex_test_no_resize_256x256.npz'))
true_labels = true_dict['y'].astype('int16')
tissue_list = true_dict['tissue_list']
platform_list = true_dict['platform_list']

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
fig.savefig(base_dir + 'anuclear_cell_count.pdf')

# # cluster purity
#
# datasets = ['TB_Data', 'TNBC_data']
# base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/analyses/20200809_cluster_purity/'
#
# for folder in datasets:
#
#     # cluster purity comparison
#     channel_data = data_utils.load_imgs_from_tree(os.path.join(base_dir, folder, 'channel_data'),
#                                                   dtype='float32')
#
#     segmentation_labels = xr.open_dataarray(os.path.join(base_dir, folder, 'segmentation_labels.xr'))
#     segmentation_labels = segmentation_labels.loc[channel_data.fovs]
#
#     normalized_counts, transformed_counts = segmentation_utils.generate_expression_matrix(segmentation_labels, channel_data)
#
#     normalized_counts.to_csv(os.path.join(base_dir, folder, 'normalized_counts_cell.csv'))
#     transformed_counts.to_csv(os.path.join(base_dir, folder, 'transformed_counts_cell.csv'))
#
#     segmentation_labels_nuc = xr.open_dataarray(
#         os.path.join(base_dir, folder, 'segmentation_labels_nuc.xr'))
#
#     segmentation_labels_nuc = segmentation_labels_nuc.loc[channel_data.fovs]
#
#     segmentation_labels_nuc = xr.DataArray(segmentation_labels_nuc.values,
#                                            coords=[segmentation_labels_nuc.fovs,
#                                                    segmentation_labels_nuc.rows,
#                                                    segmentation_labels_nuc.cols,
#                                                    ['whole_cell']],
#                                            dims=segmentation_labels_nuc.dims)
#
#     normalized_counts_nuc, transformed_counts_nuc = segmentation_utils.generate_expression_matrix(
#         segmentation_labels_nuc, channel_data)
#
#     normalized_counts_nuc.to_csv(os.path.join(base_dir, folder, 'normalized_counts_nuc.csv'))
#     transformed_counts_nuc.to_csv(os.path.join(base_dir, folder, 'transformed_counts_nuc.csv'))
#
#
# tnbc_cell = pd.read_csv(os.path.join(base_dir, 'TNBC_data', 'normalized_counts_cell.csv'))
# tnbc_nuc = pd.read_csv(os.path.join(base_dir, 'TNBC_data', 'normalized_counts_nuc.csv'))
#
#
# for label in segmentation_labels.fovs.values:
#     mask = segmentation_labels.loc[label, :, :, 'whole_cell']
#     io.imsave(os.path.join(base_dir, 'TNBC_data/channel_data', label, 'segmentation_label_cell.tiff'),
#               mask.astype('int16'))
#
#
# for label in segmentation_labels_nuc.fovs.values:
#     mask = segmentation_labels_nuc.loc[label, :, :, 'whole_cell']
#     io.imsave(os.path.join(base_dir, 'TNBC_data/channel_data', label, 'segmentation_label_nuc.tiff'),
#               mask.astype('int16'))
#
#
# fig, axes = plt.subplots(2, 1, figsize=(15, 15))
# axes[0].scatter(tnbc_nuc['CD45'].values, tnbc_nuc['Beta catenin'].values)
# axes[1].scatter(tnbc_cell['CD45'].values, tnbc_cell['Beta catenin'].values)
#
# axes[0].set_xlabel('CD45')
# axes[1].set_xlabel('CD45')
# axes[1].set_title('Whole cell segmentation')
# axes[0].set_ylabel('Beta Catenin')
# axes[1].set_ylabel('Beta Catenin')
# axes[0].set_title('Nuclear segmentation')
#
#
# panc_cell = pd.read_csv(os.path.join(base_dir, 'Panc_data', 'normalized_counts_cell.csv'))
# panc_nuc = pd.read_csv(os.path.join(base_dir, 'Panc_data', 'normalized_counts_nuc.csv'))
#
#
# plt.scatter(panc_nuc['Glucagon'].values, panc_nuc['Proinsulin'].values)
#
# fig, ax = plt.subplots(2, 1)
# ax[0].scatter(panc_nuc['Glucagon'].values, panc_nuc['Proinsulin'].values)
# ax[1].scatter(panc_cell['Glucagon'].values, panc_cell['Proinsulin'].values)