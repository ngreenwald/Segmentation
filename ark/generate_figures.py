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


from deepcell_toolbox.metrics import Metrics

from ark import figures
from ark.utils import data_utils, segmentation_utils, io_utils
from ark.segmentation import marker_quantification

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Figure 2
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/analyses/20200720_lab_meeting/updated_dcis/'

# extract TIFs and labels from xarrays
raw_data = xr.open_dataarray(os.path.join(base_dir, 'deepcell_input.xr'))
labels = xr.open_dataarray(os.path.join(base_dir, 'segmentation_labels.xr'))

for fov in raw_data.fovs.values:
    fov_folder = os.path.join(base_dir, fov)
    os.makedirs(fov_folder)

    io.imsave(os.path.join(fov_folder, 'DNA.tiff'), raw_data.loc[fov, :, :, 'DNA'].astype('int16'))
    io.imsave(os.path.join(fov_folder, 'Membrane.tiff'), raw_data.loc[fov, :, :, 'Membrane'].astype('int16'))
    io.imsave(os.path.join(fov_folder, 'cell_labels.tiff'), labels.loc[fov, :, :, 'whole_cell'])
    io.imsave(os.path.join(fov_folder, 'nuc_labels.tiff'), labels.loc[fov, :, :, 'nuclear'])


for fov in raw_data.fovs.values:
    # figures.preprocess_overlays(os.path.join(base_dir, folder))
    DNA = io.imread(os.path.join(base_dir, fov, 'DNA.tiff'))
    Membrane = io.imread(os.path.join(base_dir, fov, 'Membrane.tiff'))
    cell_label = io.imread(os.path.join(base_dir, fov, 'cell_labels.tiff'))
    nuc_label = io.imread(os.path.join(base_dir, fov, 'nuc_labels.tiff'))
    nuc_label_expanded = figures.nuclear_expansion_pixel(label_map=nuc_label, expansion_radius=4)
    io.imsave(os.path.join(base_dir, fov, 'nuc_label_expanded.tiff'), nuc_label_expanded)

    # rgb_image = figures.generate_RGB_image(red=None, blue=DNA[100:400, 200:350],
    #                                        green=Membrane[100:400, 200:350],
    #                                        percentile_cutoffs=(0, 100))

    rgb_image = figures.generate_RGB_image(red=None, blue=DNA,
                                           green=Membrane,
                                           percentile_cutoffs=(0, 100))
    map_color = figures.color_labels_with_map(cell_label)

    map_color[map_color == 1] = 100
    map_color[map_color == 2] = 130
    map_color[map_color == 3] = 160
    map_color[map_color == 4] = 190
    map_color[map_color == 5] = 220
    map_color[map_color == 6] = 250

    io.imsave(os.path.join(base_dir, fov, 'rgb_overlay.tiff'), rgb_image)
    io.imsave(os.path.join(base_dir, fov, 'greyscale_cell_label_map.tiff'),
              map_color.astype('uint8'))

    map_color = figures.color_labels_with_map(nuc_label_expanded)

    map_color[map_color == 1] = 100
    map_color[map_color == 2] = 130
    map_color[map_color == 3] = 160
    map_color[map_color == 4] = 190
    map_color[map_color == 5] = 220
    map_color[map_color == 6] = 250

    io.imsave(os.path.join(base_dir, fov, 'greyscale_nuc_label_map.tiff'),
              map_color.astype('uint8'))

# fov_dir = os.path.join(base_dir, '20200424_TB')
# overlay = io.imread(os.path.join(fov_dir, 'rgb_overlay.tiff'))
# row_idxs = [112, 512]
# col_idxs = [0, 300]
# io.imsave(os.path.join(fov_dir, 'rgb_overlay_cropped.tiff'),
#           overlay[row_idxs[0]:row_idxs[1], col_idxs[0]:col_idxs[1]])
# label_map = io.imread(os.path.join(fov_dir, 'greyscale_cell_label_map.tiff'))
# io.imsave(os.path.join(fov_dir, 'greyscale_cell_label_map_cropped.tiff'),
#           label_map[row_idxs[0]:row_idxs[1], col_idxs[0]:col_idxs[1]])

# DCIS
fov_dir = os.path.join(base_dir, '20200116_DCIS')
row_idxs = [50, 400]
col_idxs = [100, 450]

# Roshan
fov_dir = os.path.join(base_dir, '20200219_Roshan')
row_idxs = [50, 400]
col_idxs = [0, 350]

DNA = io.imread(os.path.join(fov_dir, 'DNA.tiff'))
io.imsave(os.path.join(fov_dir, 'DNA_cropped_fat.tiff'),
          DNA[row_idxs[0]:row_idxs[1], col_idxs[0]:col_idxs[1]])

Membrane = io.imread(os.path.join(fov_dir, 'Membrane.tiff'))
io.imsave(os.path.join(fov_dir, 'Membrane_cropped_fat.tiff'),
          Membrane[row_idxs[0]:row_idxs[1], col_idxs[0]:col_idxs[1]])

overlay = io.imread(os.path.join(fov_dir, 'rgb_overlay.tiff'))
io.imsave(os.path.join(fov_dir, 'rgb_overlay_cropped_fat.tiff'),
          overlay[row_idxs[0]:row_idxs[1], col_idxs[0]:col_idxs[1]])

label_map = io.imread(os.path.join(fov_dir, 'greyscale_cell_label_map.tiff'))
io.imsave(os.path.join(fov_dir, 'greyscale_cell_label_map_cropped_fat.tiff'),
          label_map[row_idxs[0]:row_idxs[1], col_idxs[0]:col_idxs[1]])

nuc_label_map = io.imread(os.path.join(fov_dir, 'greyscale_nuc_label_map.tiff'))
io.imsave(os.path.join(fov_dir, 'greyscale_nuc_label_map_cropped_fat.tiff'),
          nuc_label_map[row_idxs[0]:row_idxs[1], col_idxs[0]:col_idxs[1]])



figures.preprocess_overlays(base_dir + '20200219_Roshan_test')

figures.generate_crop(base_dir + '20200219_Roshan', row_start=500, col_start=300, length=400)

figures.generate_inset(base_dir + '20200219_Roshan', row_start=100, col_start=200, length=100,
                       inset_num=1, thickness=2)

figures.generate_inset(base_dir + '20200219_Roshan', row_start=50, col_start=25, length=100,
                       inset_num=2, thickness=2)

# create overlay
DNA = io.imread(os.path.join(base_dir, '20200624_graham_pancreas', 'DNA_resized.tiff'))
Membrane = io.imread(os.path.join(base_dir, '20200624_graham_pancreas', 'Membrane_resized.tiff'))
label = io.imread(os.path.join(base_dir, '20200624_graham_pancreas', 'labels_resized.tiff'))

plot_utils.plot_overlay(predicted_contour=label, plotting_tif=np.stack((DNA, Membrane), axis=-1),
                        percentile_cutoffs=(20, 95))


# create overlay
DNA = io.imread(os.path.join(base_dir, '20200219_Roshan', 'DNA_cropped.tiff'))
Membrane = io.imread(os.path.join(base_dir, '20200219_Roshan', 'Membrane_cropped.tiff'))
label = io.imread(os.path.join(base_dir, '20200219_Roshan', 'labels_cropped.tiff'))

figures.preprocess_overlays(base_dir + '20200624_graham_pancreas')

figures.generate_crop(base_dir + '20200226_Melanoma', row_start=300, col_start=300, length=400)

figures.generate_inset(base_dir + '20200226_Melanoma', row_start=100, col_start=200, length=100,
                       inset_num=1, thickness=2)

figures.generate_inset(base_dir + '20200226_Melanoma', row_start=50, col_start=25, length=100,
                       inset_num=2, thickness=2)

# create paired overlay and crop
plot_dir = base_dir + '20200219_Roshan/'
DNA = io.imread(plot_dir + 'DNA_cropped.tiff')
Membrane = io.imread(plot_dir + 'Membrane_cropped.tiff')
label = io.imread(plot_dir + 'labels.tiff')
label = resize(label, [label.shape[0] * 2, label.shape[1] * 2], order=0, preserve_range=True)
label = label[300:700, 300:700]
label = label[500:900, 300:700]

io.imsave(plot_dir + 'labels_cropped_whole_label.tiff', label.astype('int16'))


DNA = DNA[:, :300]
Membrane = Membrane[:, :300]
label = label[:, :300]

io.imsave(plot_dir + 'DNA_cropped_side_by_side.tiff', DNA)
io.imsave(plot_dir + 'Membrane_cropped_side_by_side.tiff', Membrane)

label = label / np.max(label)
label_colormap = cm.jet(label)
io.imsave(plot_dir + 'Label_cropped_side_by_side_new.tiff', label_colormap.astype('float32'))


io.imsave(plot_dir + 'Label_cropped_Side_by_side_new_greyscale.tiff', label_shrunk.astype('uint8'))

test_img = np.zeros((10, 10))
test_img[:5, :5] = 1
test_img[:5, 5:10] = 2

test_img_small = erosion(test_img, selem=skimage.morphology.disk(1))

# expansion
expansion_labels = figures.nuclear_expansion_pixel(label_map=nuclear_label, expansion_radius=4)

expansion_outline = find_boundaries(expansion_labels, connectivity=1, mode='inner').astype('uint8')
expansion_outline[expansion_outline > 0] = 255
io.imsave(os.path.join(base_dir, 'watershed_outline.tiff'), expansion_outline)

plot_utils.plot_overlay(predicted_contour=watershed_labels,  #[100:200, :100],
                        plotting_tif=combined_data,  # [100:200, :100, :],
                        path=os.path.join(base_dir, 'watershed_expansion.tiff'))

plot_utils.plot_overlay(predicted_contour=watershed_labels[120:180, 20:80],
                        plotting_tif=combined_data[120:180, 20:80, :],
                        path=os.path.join(base_dir, 'watershed_expansion_cropped.tiff'))

plot_utils.plot_overlay(predicted_contour=expansion_labels, plotting_tif=combined_data,
                        path=os.path.join(base_dir, 'nuclear_expansion.tiff'))

plot_utils.plot_overlay(predicted_contour=expansion_labels[120:180, 20:80],
                        plotting_tif=combined_data[120:180, 20:80],
                        path=os.path.join(base_dir, 'nuclear_expansion_cropped.tiff'))


plot_utils.plot_overlay(predicted_contour=true_label, plotting_tif=combined_data,
                        path=os.path.join(base_dir, 'true_label.tiff'))

plot_utils.plot_overlay(predicted_contour=true_label[120:180, 20:80],
                        plotting_tif=combined_data[120:180, 20:80],
                        path=os.path.join(base_dir, 'true_label_cropped.tiff'))

relabeled, _, _ = skimage.segmentation.relabel_sequential(true_label)
randomized = plot_utils.randomize_labels(relabeled)

x = cm.cubehelix(true_label)
x[expansion_outline > 0] = 255
io.imshow(x)
io.imsave(os.path.join(base_dir, 'true_labels_border_outline.tiff'), x.astype('float32'))
# CMYK image generation


rescaled = np.zeros((200, 200, 3), dtype='uint8')

for idx in range(1, 3):

    percentiles = np.percentile(combined_data[:, :, idx][combined_data[:, :, idx] > 0],
                                [5, 95])
    rescaled_intensity = rescale_intensity(combined_data[:, :, idx],
                                           in_range=(percentiles[0], percentiles[1]),
                                           out_range='uint8')
    rescaled[:, :, idx] = rescaled_intensity

io.imsave(os.path.join(base_dir, 'overlay_channels.tiff'), rescaled)

from skimage.exposure import rescale_intensity

DNA = io.imread(base_dir + 'DNA.tiff')
Membrane = io.imread(base_dir + 'Membrane.tiff')

DNA_unscaled = rescale_intensity(DNA, out_range='uint8').astype('uint8')
DNA_256 = Image.fromarray(rescale_intensity(DNA, out_range='uint8').astype('uint8'))
DNA_256_invert = DNA_256.point(lambda i: 256 - i)

Membrane_256 = Image.fromarray(rescale_intensity(Membrane, out_range='uint8').astype('uint8'))
Membrane_256_invert = Membrane_256.point(lambda i: 256 - i)
blank_256 = Image.fromarray(np.zeros((512, 512), dtype='uint8'))

merged_5 = Image.merge('CMYK', (DNA_256, blank_256, blank_256, Membrane_256))
merged_5.save(base_dir + 'combined_5.jpg')


# Figure 3

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
    img_names = os.listdir(folder_path)
    img_names = [img for img in img_names if '.tiff' in img]
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
plt.savefig('/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/analyses/20200720_lab_meeting/human_comparison.pdf', transparent=True)


# Figure 4
#
#

true_labels = io.imread('/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/analyses/20200701_morphology/Point072_true.tiff')
pred_labels = io.imread('/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/analyses/20200701_morphology/Point072_predicted.tiff')

pred_props_table = regionprops_table(pred_labels,properties=['label', 'eccentricity', 'centroid'])
true_props_table = regionprops_table(true_labels,properties=['label', 'eccentricity', 'centroid'])

true_ecc, pred_ecc = figures.get_paired_regionprops(true_labels=true_labels,
                                                    pred_labels=pred_labels,
                                                    true_props_table=true_props_table,
                                                    pred_props_table=pred_props_table,
                                                    field='eccentricity')

plt.scatter(true_ecc, pred_ecc)
r2_score(true_ecc, pred_ecc)

datasets = ['TB_Data', 'TNBC_data']
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/analyses/20200809_cluster_purity/'

for folder in datasets:

    # cluster purity comparison
    channel_data = data_utils.load_imgs_from_tree(os.path.join(base_dir, folder, 'channel_data'),
                                                  dtype='float32')

    segmentation_labels = xr.open_dataarray(os.path.join(base_dir, folder, 'segmentation_labels.xr'))
    segmentation_labels = segmentation_labels.loc[channel_data.fovs]

    normalized_counts, transformed_counts = segmentation_utils.generate_expression_matrix(segmentation_labels, channel_data)

    normalized_counts.to_csv(os.path.join(base_dir, folder, 'normalized_counts_cell.csv'))
    transformed_counts.to_csv(os.path.join(base_dir, folder, 'transformed_counts_cell.csv'))

    segmentation_labels_nuc = xr.open_dataarray(
        os.path.join(base_dir, folder, 'segmentation_labels_nuc.xr'))

    segmentation_labels_nuc = segmentation_labels_nuc.loc[channel_data.fovs]

    segmentation_labels_nuc = xr.DataArray(segmentation_labels_nuc.values,
                                           coords=[segmentation_labels_nuc.fovs,
                                                   segmentation_labels_nuc.rows,
                                                   segmentation_labels_nuc.cols,
                                                   ['whole_cell']],
                                           dims=segmentation_labels_nuc.dims)

    normalized_counts_nuc, transformed_counts_nuc = segmentation_utils.generate_expression_matrix(
        segmentation_labels_nuc, channel_data)

    normalized_counts_nuc.to_csv(os.path.join(base_dir, folder, 'normalized_counts_nuc.csv'))
    transformed_counts_nuc.to_csv(os.path.join(base_dir, folder, 'transformed_counts_nuc.csv'))


tnbc_cell = pd.read_csv(os.path.join(base_dir, 'TNBC_data', 'normalized_counts_cell.csv'))
tnbc_nuc = pd.read_csv(os.path.join(base_dir, 'TNBC_data', 'normalized_counts_nuc.csv'))


for label in segmentation_labels.fovs.values:
    mask = segmentation_labels.loc[label, :, :, 'whole_cell']
    io.imsave(os.path.join(base_dir, 'TNBC_data/channel_data', label, 'segmentation_label_cell.tiff'),
              mask.astype('int16'))


for label in segmentation_labels_nuc.fovs.values:
    mask = segmentation_labels_nuc.loc[label, :, :, 'whole_cell']
    io.imsave(os.path.join(base_dir, 'TNBC_data/channel_data', label, 'segmentation_label_nuc.tiff'),
              mask.astype('int16'))


fig, axes = plt.subplots(2, 1, figsize=(15, 15))
axes[0].scatter(tnbc_nuc['CD45'].values, tnbc_nuc['Beta catenin'].values)
axes[1].scatter(tnbc_cell['CD45'].values, tnbc_cell['Beta catenin'].values)

axes[0].set_xlabel('CD45')
axes[1].set_xlabel('CD45')
axes[1].set_title('Whole cell segmentation')
axes[0].set_ylabel('Beta Catenin')
axes[1].set_ylabel('Beta Catenin')
axes[0].set_title('Nuclear segmentation')


panc_cell = pd.read_csv(os.path.join(base_dir, 'Panc_data', 'normalized_counts_cell.csv'))
panc_nuc = pd.read_csv(os.path.join(base_dir, 'Panc_data', 'normalized_counts_nuc.csv'))


plt.scatter(panc_nuc['Glucagon'].values, panc_nuc['Proinsulin'].values)

fig, ax = plt.subplots(2, 1)
ax[0].scatter(panc_nuc['Glucagon'].values, panc_nuc['Proinsulin'].values)
ax[1].scatter(panc_cell['Glucagon'].values, panc_cell['Proinsulin'].values)

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

ecad = pd.DataFrame()

for fov in fovs:
    channel_data = data_utils.load_imgs_from_tree(base_dir, fovs=[fov],
                                                  img_sub_folder='potential_channels')

    current_labels = segmentation_labels.loc[[fov], :, :, :]

    normalized, transformed, raw = marker_quantification.generate_expression_matrix(
        segmentation_labels=current_labels,
        image_data=channel_data,
        nuclear_counts=True
    )
    if 'ECAD' in raw.columns:
        ecad = ecad.append(raw.loc[:, ['cell_size', 'ECAD', 'HH3', 'label',
                             'cell_size_nuclear', 'ECAD_nuclear', 'HH3_nuclear', 'label_nuclear']])