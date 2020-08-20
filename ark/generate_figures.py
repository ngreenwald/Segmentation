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
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/analyses/20200820_figure_2_overlays'

# extract TIFs and labels from xarrays
raw_data = xr.load_dataarray(os.path.join(base_dir, 'deepcell_input.xr'))
labels = xr.load_dataarray(os.path.join(base_dir, 'segmentation_labels.xr'))

# extract files from arrays
for fov in raw_data.fovs.values:
    fov_folder = os.path.join(base_dir, fov)
    os.makedirs(fov_folder)

    io.imsave(os.path.join(fov_folder, 'DNA.tiff'), raw_data.loc[fov, :, :, 'DNA'].astype('int16'))
    io.imsave(os.path.join(fov_folder, 'Membrane.tiff'), raw_data.loc[fov, :, :, 'Membrane'].astype('int16'))
    io.imsave(os.path.join(fov_folder, 'cell_labels.tiff'), labels.loc[fov, :, :, 'segmentation_label'])
    io.imsave(os.path.join(fov_folder, 'nuc_labels.tiff'), labels.loc[fov, :, :, 'segmentation_label'])


# specify crops for each image
row_idx_list = [[50, 400], [50, 400], [50, 400]]
col_idx_list = [[100, 450], [0, 350], [0, 350]]

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

    imgs = [DNA, Membrane, label_map_cell, label_map_nuc, rgb_image]
    names = ['DNA', 'Membrane', 'cell_label_map', 'nuc_label_map', 'rgb_image']

    row_start, row_end = row_idx_list[idx][0], row_idx_list[idx][1]
    col_start, col_end = col_idx_list[idx][0], col_idx_list[idx][1]
    for img, name in zip(imgs, names):
        cropped = img[row_start:row_end, col_start:col_end]
        io.imsave(os.path.join(base_dir, fov, name + '_cropped.tiff'), cropped)



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

# morphology comparisons
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/analyses/20200816_val_accuracy'
true_labels = np.load(os.path.join(base_dir, '20200816_all_data_normalized_512x512val_split.npz'))['y']
pred_labels = np.load(os.path.join(base_dir, 'cell_labels.npz'))['y']
pred_labels = np.load(os.path.join(base_dir, 'nuc_labels_expanded.npz'))['y']


properties_df = pd.DataFrame()
for i in range(true_labels.shape[0]):
    properties = ['label', 'area', 'eccentricity', 'major_axis_length', 'minor_axis_length',
                  'perimeter', ]
    pred = pred_labels[i, :, :, 0]
    true = true_labels[i, :, :, 0]

    if np.max(pred) == 0 or np.max(true) == 0:
        continue
    pred_props_table = regionprops_table(pred, properties=properties)
    true_props_table = regionprops_table(true, properties=properties)
    properties_dict = {}
    for prop in properties[1:]:
        true_prop, pred_prop = figures.get_paired_regionprops(true_labels=true_labels[i, :, :, 0],
                                                              pred_labels=pred_labels[i, :, :, 0],
                                                              true_props_table=true_props_table,
                                                              pred_props_table=pred_props_table,
                                                              field=prop)
        properties_dict[prop] = true_prop
        properties_dict[prop + '_predicted'] = pred_prop

    properties_df = properties_df.append(pd.DataFrame(properties_dict))


fig, ax = plt.subplots(2, 3, figsize=(15, 10))
row_idx = 0
for i in range(1, len(properties)):
    prop_name = properties[i]
    if i > 2:
        row_idx = 1
    col_idx = i % 3
    true_vals = properties_df[prop_name].values
    predicted_vals = properties_df[prop_name + '_predicted'].values

    import numpy as np
    from scipy.stats import gaussian_kde

    # Calculate the point density
    xy = np.vstack([true_vals, predicted_vals])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = true_vals[idx], predicted_vals[idx], z[idx]

    #ax[row_idx, col_idx].scatter(x=true_vals, y=predicted_vals, alpha=0.01)
    ax[row_idx, col_idx].scatter(x, y, c=z, s=50, edgecolor='')

    ax[row_idx, col_idx].set_xlabel('True Value')
    ax[row_idx, col_idx].set_ylabel('Predicted Value')
    ax[row_idx, col_idx].set_title('Correlation of {}'.format(prop_name))

    import numpy as np
    from numpy.polynomial.polynomial import polyfit
    import matplotlib.pyplot as plt

    # Fit with polyfit
    b, m = polyfit(true_vals,
                   predicted_vals,
                   deg=1)

    x = np.arange(0, np.max(properties_df[prop_name].values))
    ax[row_idx, col_idx].plot(x, b + m * x, '-', color='red')
    # r2_val = r2_score(properties_df[prop_name].values,
    #                   properties_df[prop_name + '_predicted'].values)
    p_r, _ = pearsonr(true_vals, predicted_vals)
    x_pos = np.max(true_vals) * 0.05
    y_pos = np.max(predicted_vals) * 0.9
    ax[row_idx, col_idx].text(x_pos, y_pos, 'Pearson Correlation: {}'.format(np.round(p_r, 2)))

fig.savefig(os.path.join(base_dir, 'morphology_correlation_nuc.png'))




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

# read in segmented data
cell_counts = pd.read_csv(os.path.join(base_dir, 'single_cell_data.csv'))
cell_counts = cell_counts.loc[cell_counts['cell_size_nuclear'] > 20, :]


channels = ['CD44', 'ECAD', 'GLUT1', 'HER2', 'HH3', 'Ki67', 'P', 'PanKRT', 'pS6']

fig, ax = plt.subplots(2, 5, figsize=(30, 20))
row_idx = 0
for i in range(len(channels)):
    chan_name = channels[i]
    if i > 4:
        row_idx = 1
    col_idx = i % 5
    channel_counts = cell_counts.loc[:, [chan_name, chan_name + '_nuclear']]
    cutoff = np.percentile(cell_counts.values[cell_counts.values[:, 0] > 0, 0], [10])
    channel_counts = channel_counts.loc[channel_counts[chan_name] > cutoff[0], :]

    ratio = channel_counts.values[:, 1] / channel_counts.values[:, 0]

    ax[row_idx, col_idx].hist(ratio, bins=np.arange(0, 1.01, 0.05))
    avg = np.median(ratio)
    ax[row_idx, col_idx].axvline(avg, color='red')
    ax[row_idx, col_idx].set_title('Fraction nuc signal for {}'.format(chan_name))

fig.savefig(os.path.join(base_dir, 'nuclear_fraction.pdf'))
