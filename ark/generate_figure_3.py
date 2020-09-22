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
data_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/20200413_Human_Agreement/'
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/analyses/20200825_figure_3_heatmaps/'
prediction_xr = xr.open_dataarray(data_dir + 'segmentation_labels.xr')
for i in range(prediction_xr.shape[0]):
    prediction = prediction_xr.values[i, :, :, 0]
    io.imsave(os.path.join(base_dir, prediction_xr.fovs.values[i], 'segmentation_label.tiff'),
              prediction.astype('int16'))

# create list to hold f1 scores from each condition
folders = list(prediction_xr.fovs.values)
folder_names = ['DCIS_MIBI', 'Colon_IF', 'Esophagus_MIBI', 'Hodgekins_Vectra']
human_alg_df = pd.DataFrame()

for i in range(len(folders)):
    # get all of the human annotations
    folder_path = os.path.join(base_dir, folders[i], 'annotations')
    img_names = io_utils.list_files(folder_path, '.tiff')
    imgs = []
    for img in img_names:
        current_img = io.imread(os.path.join(folder_path, img))
        imgs.append(current_img)
    f1_scores_human = figures.calculate_human_f1_scores(image_list=imgs)
    tissue_name = folder_names[i] + '_human'
    temp_df = pd.DataFrame({'tissue': np.repeat(tissue_name, len(f1_scores_human)),
                            'F1_score': f1_scores_human})
    human_alg_df = human_alg_df.append(temp_df)

    # compare algorithm
    pred_img = io.imread(os.path.join(base_dir, folders[i], 'segmentation_label.tiff'))
    pred_img = np.expand_dims(pred_img, axis=0)
    f1_scores_alg = figures.calculate_alg_f1_scores(image_list=imgs, alg_pred=pred_img)

    tissue_name = folder_names[i] + '_alg'
    temp_df = pd.DataFrame({'tissue': np.repeat(tissue_name, len(f1_scores_alg)),
                            'F1_score': f1_scores_alg})
    human_alg_df = human_alg_df.append(temp_df)

human_alg_df.to_csv(os.path.join(base_dir + 'human_alg_comparison.csv'))
g = sns.catplot(data=human_alg_df,
                kind='strip', x='tissue', y='F1_score')

plt.savefig(os.path.join(base_dir, 'human_comparison.pdf'))

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
ax = sns.heatmap(data=platform_array, annot=True, vmin=0, cmap='viridis')
plt.savefig(os.path.join(base_dir, 'tissue_heatmap_virdis_sns.pdf'))



platform_types = ['vectra', 'mibi', 'all']
save_path = os.path.join(base_dir, 'platform_heatmap.pdf')
platform_array = figures.create_f1_score_grid([vectra, mibi, all_platform], platform_types)

ax = sns.heatmap(data=platform_array, annot=True, vmin=0, cmap='viridis')
plt.savefig(os.path.join(base_dir, 'tissue_heatmap_virdis.pdf'))
