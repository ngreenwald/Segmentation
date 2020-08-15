import os

import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import skimage.io as io
import shutil
from skimage.measure import label

from ark.utils import data_utils, io_utils

# extract cHL data
import h5py, re, os
import pandas as pd


base_dir = "/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/20200114_cHL/data/"
fname = 'cHL-MIF-Noah.20200114.h5'
f = h5py.File(base_dir + fname, 'r')

deidentified = pd.read_hdf(base_dir + fname, 'key')
deidentified[['label']].drop_duplicates()


def get_image(fname,image_id):
    return h5py.File(fname,'r')['images/'+image_id]


# extract TIFs
for i, r in deidentified.iloc[:, :].iterrows():
    img = np.array(get_image(base_dir + fname,r['image_id']))

    if r['label'] in ["CD3 (Opal 540)", "DAPI", "CD8 (Opal 540)", "CD4 (Opal 620)"]:
        if not os.path.isdir(base_dir + r['frame_id']):
            os.makedirs(base_dir + r['frame_id'])

        io.imsave(base_dir + r['frame_id'] + "/" + r['label'] + ".tiff", img.astype('float32'))

# combine CD4 and CD8
base_dir = "/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/20200114_cHL/data/Good/"

good_dir = os.listdir(base_dir)
good_dir = [x for x in good_dir if ".DS" not in x]
for i in good_dir:
    if "CD4 (Opal 620).tiff" in os.listdir(base_dir + i):
        CD4 = io.imread(base_dir + i + "/CD4 (Opal 620).tiff")
        CD8 = io.imread(base_dir + i + "/CD8 (Opal 540).tiff")
        combined = CD4 + CD8
        io.imsave(base_dir + i + "/Membrane.tiff", combined)


base_dir = "/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/20200114_cHL/Great/"

fovs = os.listdir(base_dir + 'fovs')

for fov in fovs:
    imgs = os.listdir(os.path.join(base_dir, 'fovs', fov))
    if 'Membrane.tiff' in imgs:
        membrane_name = 'Membrane.tiff'
    else:
        membrane_name = 'CD3 (Opal 540).tiff'

    data = data_utils.load_imgs_from_dir(data_dir=os.path.join(base_dir, 'fovs'),
                                         fovs=[fov], dtype='float32',
                                         imgs=['DAPI.tiff', membrane_name])

    cropped_data = data_utils.crop_image_stack(data.values, 512, 1)
    for crop in range(cropped_data.shape[0]):
        folder = os.path.join(base_dir, 'cropped/{}_crop_{}'.format(fov, crop))
        os.makedirs(folder)
        io.imsave(os.path.join(folder, 'DNA.tiff'), cropped_data[crop, :, :, 0])
        io.imsave(os.path.join(folder, 'Membrane.tiff'), cropped_data[crop, :, :, 1])


# Tyler BRCA IF data
base_dir = "/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/20191213_Tyler_BRCA/clean/"

fovs = os.listdir(base_dir + 'trim_borders')
fovs = [fov for fov in fovs if 'Point' in fov]

for fov in fovs:
    data = data_utils.load_imgs_from_dir(data_dir=os.path.join(base_dir, 'trim_borders'),
                                         fovs=[fov], dtype='int32',
                                         imgs=['DAPI.tif', 'Membrane.tif'])

    cropped_data = data_utils.crop_image_stack(data.values, 512, 1)
    for crop in range(cropped_data.shape[0]):
        folder = os.path.join(base_dir, 'cropped/{}_crop_{}'.format(fov, crop))
        os.makedirs(folder)
        io.imsave(os.path.join(folder, 'DNA.tiff'), cropped_data[crop, :, :, 0])
        io.imsave(os.path.join(folder, 'Membrane.tiff'), cropped_data[crop, :, :, 1])


# Eliot data preprocessing
base_dir = "/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/20191219_Eliot/Good/"

fovs = os.listdir(base_dir + 'fovs')
fovs = [fov for fov in fovs if 'Point' in fov]

for fov in fovs:
    data = data_utils.load_imgs_from_dir(data_dir=os.path.join(base_dir, 'trim_borders'),
                                         fovs=[fov], dtype='int32',
                                         imgs=['DAPI.tif', 'Membrane.tif'])

    cropped_data = data_utils.crop_image_stack(data.values, 512, 1)
    for crop in range(cropped_data.shape[0]):
        folder = os.path.join(base_dir, 'cropped/{}_crop_{}'.format(fov, crop))
        os.makedirs(folder)
        io.imsave(os.path.join(folder, 'DNA.tiff'), cropped_data[crop, :, :, 0])
        io.imsave(os.path.join(folder, 'Membrane.tiff'), cropped_data[crop, :, :, 1])


# DCIS processing
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/20200116_DCIS/'
fovs = os.listdir(os.path.join(base_dir, 'Great_Membrane/selected_fovs'))
fovs = [fov for fov in fovs if 'Point' in fov]

# copy files from no_bg folder to selected_no_bg folder so these can be used for training
for fov in fovs:
    original_folder = os.path.join(base_dir, 'Great_Membrane/selected_fovs', fov)
    new_folder = os.path.join(base_dir, 'Great_Membrane/no_bg_fovs', fov)
    os.makedirs(new_folder)
    imgs = os.listdir(original_folder)
    imgs = [img for img in imgs if '.tif' in img]

    for img in imgs:
        shutil.copy(os.path.join(base_dir, 'no_background', fov, 'TIFs', img), new_folder)

# load HH3 and whatever membrane marker is in each folder, crop to 512, save with consistent name
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/20200116_DCIS/Great_Membrane/'
fovs = os.listdir(base_dir + 'no_bg_fovs')
fovs = [fov for fov in fovs if 'Point' in fov]

for fov in fovs:
    imgs = os.listdir(os.path.join(base_dir, 'no_bg_fovs', fov))
    imgs = [img for img in imgs if 'tif' in img]

    # remove DNA, remaining channel is membrane
    imgs.pop(np.where(np.isin(imgs, 'HH3.tif'))[0][0])
    membrane_channel = imgs[0]

    data = data_utils.load_imgs_from_dir(data_dir=os.path.join(base_dir, 'no_bg_fovs'),
                                         fovs=[fov], imgs=['HH3.tif', membrane_channel])

    cropped_data = data_utils.crop_image_stack(data.values, 512, 1)
    for crop in range(cropped_data.shape[0]):
        folder = os.path.join(base_dir, 'cropped/{}_crop_{}'.format(fov, crop))
        os.makedirs(folder)
        io.imsave(os.path.join(folder, 'DNA.tiff'), cropped_data[crop, :, :, 0])
        io.imsave(os.path.join(folder, 'Membrane.tiff'), cropped_data[crop, :, :, 1])


# preprocessing for phenotyping channels
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/20200116_DCIS/'
fov_names = io_utils.list_folders(base_dir + 'Okay_Membrane')

for fov in fov_names:
    # shutil.copytree(os.path.join(base_dir, 'no_background', fov),
    #             os.path.join(base_dir, 'phenotyping_okay', fov))

    # CD45 needs to be denoised
    shutil.copy(os.path.join(base_dir, 'no_noise', fov, 'TIFs/CD45.tif'),
                os.path.join(base_dir, 'phenotyping_okay', fov, 'TIFs/CD45_denoised.tif'))

phenotype_data = data_utils.load_imgs_from_dir(base_dir + 'phenotyping_okay', img_sub_folder='TIFs',
                                               imgs=['CD45_denoised.tif', 'HH3.tif', 'PanKRT.tif',
                                                     'SMA.tif', 'ECAD.tif', 'CD44.tif'])
stitched_data = data_utils.stitch_images(phenotype_data, 5)
for i in range(stitched_data.shape[-1]):
    io.imsave(os.path.join(base_dir, 'phenotyping_okay', stitched_data.channels.values[i] + '.tiff'),
              stitched_data.values[0, :, :, i])



# SMA-specific network
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/20200116_DCIS/'
fovs = io_utils.list_folders(base_dir + 'Great_Membrane/selected_fovs')

for fov in fovs:
    sma_data = data_utils.load_imgs_from_dir(base_dir + 'no_noise', img_sub_folder='TIFs',
                                             imgs=['SMA.tif'], fovs=[fov])

    cropped_sma = data_utils.crop_image_stack(sma_data, 512, 1)

    for crop in range(cropped_sma.shape[0]):
        folder = os.path.join(base_dir, 'Great_Membrane/cropped_sma/{}_crop_{}'.format(fov, crop))
        os.makedirs(folder)
        io.imsave(os.path.join(folder, 'SMA.tiff'), cropped_sma[crop, :, :, 0])


# create modified NPZ files with SMA added as 3rd channel
sma_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/20200116_DCIS/Great_Membrane/cropped_sma/'
npz_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/caliban_files/20200116_DCIS/20200619_DCIS/new/'
save_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/caliban_files/20200116_DCIS/20200705_DCIS_SMA/'

npz_files = os.listdir(npz_dir)
npz_files = [npz for npz in npz_files if '.npz' in npz]

for npz in npz_files:
    current_npz = np.load(os.path.join(npz_dir, npz))
    X = current_npz['X']
    fov_name = npz.split('_save_version')[0]
    sma = io.imread(os.path.join(sma_dir, fov_name, 'SMA.tiff'))

    X[0, :, :, 0] = sma
    np.savez(os.path.join(save_dir, fov_name + '.npz'), X=X, y=current_npz['y'])

# IMC 20191211 preprocessing
base_dir = "/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/20191211_IMC/Great/"

fovs = os.listdir(base_dir + 'fovs')
fovs = [point for point in fovs if "Point" in point]

for fov in fovs:
    data = data_utils.load_imgs_from_dir(data_dir=os.path.join(base_dir, 'fovs'), dtype='float32',
                                         fovs=[fov], imgs=['DNA.tiff', 'Membrane.tiff'])

    # only one crop per image since images are quite small: we'll center 512 in the FOV
    row_len = data.shape[1]
    col_len = data.shape[2]
    row_crop_start = math.floor((row_len - 512) / 2)
    col_crop_start = math.floor((col_len - 512) / 2)

    cropped_data = data.values[:, row_crop_start:(row_crop_start + 512),
                   col_crop_start:(col_crop_start + 512), :]

    folder = os.path.join(base_dir, 'cropped/{}_crop_0'.format(fov))
    os.makedirs(folder)
    io.imsave(os.path.join(folder, 'DNA.tiff'), cropped_data[0, :, :, 0])
    io.imsave(os.path.join(folder, 'Membrane.tiff'), cropped_data[0, :, :, 1])


# IMC 20200120 preprocessing
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/20200120_IMC/great/'
fovs = os.listdir(base_dir + 'fovs')
fovs = [fov for fov in fovs if os.path.isdir(base_dir + 'fovs/' + fov)]

for fov in fovs:
    imgs = os.listdir(os.path.join(base_dir, 'fovs', fov))
    imgs = [img for img in imgs if 'tif' in img]

    # remove DNA, remaining channel is membrane
    imgs.pop(np.where(np.isin(imgs, 'Histone.tiff'))[0][0])
    membrane_channel = imgs[0]

    data = data_utils.load_imgs_from_dir(data_dir=os.path.join(base_dir, 'fovs'), dtype='float32',
                                         fovs=[fov], imgs=['Histone.tiff', membrane_channel])

    # some images are small than 512, others are only marginally bigger
    row_len, col_len = data.shape[1:3]
    new_data = np.zeros((1, max(512, row_len), max(512, col_len), 2), dtype='float32')

    # if either dimension is less than 512, we'll expand to 512
    new_data[:, :row_len, :col_len, :] = data.values

    # for dimensions that are only marginally larger than 512, we'll use center 512 crop
    if 512 < row_len < 768:
        row_crop_start = math.floor((row_len - 512) / 2)
        new_data = new_data[:, row_crop_start:(row_crop_start + 512), :, :]

    if 512 < col_len < 768:
        col_crop_start = math.floor((col_len - 512) / 2)
        new_data = new_data[:, :, col_crop_start:(col_crop_start + 512), :]

    cropped_data = data_utils.crop_image_stack(new_data, 512, 1)
    for crop in range(cropped_data.shape[0]):
        folder = os.path.join(base_dir, 'cropped/{}_crop_{}'.format(fov, crop))
        os.makedirs(folder)
        io.imsave(os.path.join(folder, 'DNA.tiff'), cropped_data[crop, :, :, 0])
        io.imsave(os.path.join(folder, 'Membrane.tiff'), cropped_data[crop, :, :, 1])


# 2019 CyCIF paper
# extract channels
base_dir = "/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/20200209_CyCIF_SciRep/Tonsil-1/"
composite = io.imread(base_dir + "TONSIL-1_40X.ome.tif")

for chan in range(composite.shape[0]):
    io.imsave(base_dir + "Channel_{}.tif".format(chan + 1), composite[chan, :, :])

# generate crops
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/20200209_CyCIF_SciRep/Great/'
fovs = os.listdir(base_dir + 'renamed_channels')
fovs = [fov for fov in fovs if os.path.isdir(os.path.join(base_dir, 'renamed_channels', fov))]

for fov in fovs:
    data = data_utils.load_imgs_from_dir(data_dir=os.path.join(base_dir, 'renamed_channels'),
                                         fovs=[fov], dtype='int32',
                                         imgs=['DNA.tif', 'Membrane.tif'])

    cropped_data = data_utils.crop_image_stack(data.values, 512, 1)
    for crop in range(cropped_data.shape[0]):
        folder = os.path.join(base_dir, 'cropped/{}_crop_{}'.format(fov, crop))
        os.makedirs(folder)
        io.imsave(os.path.join(folder, 'DNA.tiff'), cropped_data[crop, :, :, 0])
        io.imsave(os.path.join(folder, 'Membrane.tiff'), cropped_data[crop, :, :, 1])


# Roshan processing
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/20200219_Roshan/'
fovs = os.listdir(base_dir + 'fovs')
fovs = [fov for fov in fovs if os.path.isdir(os.path.join(base_dir, 'fovs', fov))]

for fov in fovs:
    data = data_utils.load_imgs_from_dir(data_dir=os.path.join(base_dir, 'fovs'),
                                         fovs=[fov],
                                         imgs=['HH3.tif', 'CD138.tif'])

    cropped_data = data_utils.crop_image_stack(data.values, 512, 1)
    for crop in range(cropped_data.shape[0]):
        folder = os.path.join(base_dir, 'cropped/{}_crop_{}'.format(fov, crop))
        os.makedirs(folder)
        io.imsave(os.path.join(folder, 'DNA.tiff'), cropped_data[crop, :, :, 0])
        io.imsave(os.path.join(folder, 'Membrane.tiff'), cropped_data[crop, :, :, 1])


# melanoma preprocessing
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/20200226_Melanoma/Okay_Membrane/'
fovs = os.listdir(base_dir + 'fovs')
fovs = [fov for fov in fovs if os.path.isdir(os.path.join(base_dir, 'fovs', fov))]

for fov in fovs:
    data = data_utils.load_imgs_from_dir(data_dir=os.path.join(base_dir, 'fovs'),
                                         fovs=[fov],
                                         imgs=['HH3.tif', 'NAKATPASE.tif'])

    cropped_data = data_utils.crop_image_stack(data.values, 512, 1)
    for crop in range(cropped_data.shape[0]):
        folder = os.path.join(base_dir, 'cropped/{}_crop_{}'.format(fov, crop))
        os.makedirs(folder)
        io.imsave(os.path.join(folder, 'DNA.tiff'), cropped_data[crop, :, :, 0])
        io.imsave(os.path.join(folder, 'Membrane.tiff'), cropped_data[crop, :, :, 1])

# generate stitched images for viewing
all_data = data_utils.load_imgs_from_dir(base_dir + 'fovs', img_sub_folder='TIFs')
stitched_imgs = data_utils.stitch_images(all_data, 10)

for chan in range(stitched_imgs.shape[-1]):
    img = stitched_imgs[0, :, :, chan]
    io.imsave(os.path.join(base_dir, 'fovs/stitched', stitched_imgs.channels.values[chan] + '_stitched.tiff'), img.astype('int8'))

# IMC 20200411 preprocessing
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/20200411_IMC_METABRIC/'
stacks = os.listdir(base_dir + 'full_stacks')
stacks = [stack for stack in stacks if '.tiff' in stack]
data_utils.split_img_stack(stack_dir=os.path.join(base_dir, 'full_stacks'),
                           output_dir=os.path.join(base_dir, 'fovs'),
                           stack_list=stacks,
                           indices=[0, 7, 17, 25, 32, 40],
                           names=['HH3.tiff', 'CK5.tiff', 'HER2.tiff', 'CD44.tiff',
                                  'ECAD.tiff', 'PanCK.tiff'])

# copy files into folders of 50 images each
all_fovs = os.listdir(base_dir + 'fovs')
all_fovs = [fov for fov in all_fovs if os.path.isdir(os.path.join(base_dir, 'fovs', fov))]
for folder_idx in range(math.ceil(len(all_fovs) / 50)):
    folder_path = os.path.join(base_dir, 'fovs/sub_folder_{}'.format(folder_idx))
    os.makedirs(folder_path)
    for fov in range(50):
        current_fov = all_fovs[folder_idx * 50 + fov]
        shutil.move(os.path.join(base_dir, 'fovs', current_fov),
                    os.path.join(base_dir, 'fovs', folder_path, current_fov))

# create stitched overlays of each to determine which markers will be included
folders = os.listdir(base_dir + 'fovs')
folders = [folder for folder in folders if 'sub' in folder]

for folder in folders:
    image_stack = data_utils.load_imgs_from_dir(base_dir + '/fovs/' + folder, variable_sizes=True,
                                                dtype='float32')
    stitched = data_utils.stitch_images(image_stack, 10)
    for img in range(stitched.shape[-1]):
        current_img = stitched[0, :, :, img].values
        io.imsave(os.path.join(base_dir, 'fovs', folder, stitched.channels.values[img] + '.tiff'),
                  current_img)

# after manual inspection, move selected FOVs in each channel sub-folder to same overall folder
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/20200411_IMC_METABRIC/fovs/'
fovs = os.listdir(base_dir + 'HER2')
fovs = [fov for fov in fovs if 'MB' in fov]

for fov in fovs:
    new_dir = os.path.join(base_dir, 'combined', fov)
    old_dir = os.path.join(base_dir, 'HER2', fov)
    if not os.path.isdir(new_dir):
        os.makedirs(new_dir)
    shutil.copy(old_dir + '/HER2.tiff', new_dir + '/HER2.tiff')
    shutil.copy(old_dir + '/HH3.tiff', new_dir + '/HH312.tiff')


# after manual inspection to select best channel for each FOV, generate standard crops
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/20200411_IMC_METABRIC/'
fovs = os.listdir(base_dir + 'fovs')
fovs = [fov for fov in fovs if os.path.isdir(base_dir + 'fovs/' + fov)]

for fov in fovs:
    imgs = os.listdir(os.path.join(base_dir, 'fovs', fov))
    imgs = [img for img in imgs if 'tif' in img]

    # remove DNA, remaining channel is membrane
    imgs.pop(np.where(np.isin(imgs, 'HH3.tiff'))[0][0])
    membrane_channel = imgs[0]

    data = data_utils.load_imgs_from_dir(data_dir=os.path.join(base_dir, 'fovs'), dtype='float32',
                                         fovs=[fov], imgs=['HH3.tiff', membrane_channel])

    # some images are small than 512, others are only marginally bigger
    row_len, col_len = data.shape[1:3]
    new_data = np.zeros((1, max(512, row_len), max(512, col_len), 2), dtype='float32')

    # if either dimension is less than 512, we'll expand to 512
    new_data[:, :row_len, :col_len, :] = data.values

    # for dimensions that are only marginally larger than 512, we'll use center 512 crop
    if 512 < row_len < 768:
        row_crop_start = math.floor((row_len - 512) / 2)
        new_data = new_data[:, row_crop_start:(row_crop_start + 512), :, :]

    if 512 < col_len < 768:
        col_crop_start = math.floor((col_len - 512) / 2)
        new_data = new_data[:, :, col_crop_start:(col_crop_start + 512), :]

    cropped_data = data_utils.crop_image_stack(new_data, 512, 1)
    for crop in range(cropped_data.shape[0]):
        folder = os.path.join(base_dir, 'cropped/{}_crop_{}'.format(fov, crop))
        os.makedirs(folder)
        io.imsave(os.path.join(folder, 'DNA.tiff'), cropped_data[crop, :, :, 0])
        io.imsave(os.path.join(folder, 'Membrane.tiff'), cropped_data[crop, :, :, 1])



# Leeat TNBC data processing
base_dir = "/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/20200328_TNBC/"
files = os.listdir(base_dir + 'fovs')

files = [file for file in files if ".tiff" in file]

for file in files:
    m_tiff = tiff.read(os.path.join(base_dir, file))
    folder_name = os.path.splitext(file)[0]
    tiff.write(os.path.join(base_dir, folder_name), m_tiff, multichannel=False)


all_data = data_utils.load_imgs_from_dir(base_dir + '/fovs',
                                         imgs=['CD3.tiff', 'CD4.tiff', 'CD8.tiff', 'CD20.tiff',
                                               'CD56.tiff'])

all_data_stitched = data_utils.stitch_images(all_data, 10)
for idx in range(all_data_stitched.shape[-1]):
    img_path = os.path.join(base_dir, all_data_stitched.channels.values[idx] + '_stitched.tiff')
    io.imsave(img_path, all_data_stitched.values[0, :, :, idx])

# after identifying good images, sum membrane channels together
images = data_utils.load_imgs_from_dir(base_dir + 'fovs/good_membrane/', imgs=['dsDNA.tiff', 'Beta catenin.tiff',
                                                                                'CD45.tiff'])

for img in range(images.shape[0]):
    folder = os.path.join(base_dir, 'selected_images', images.fovs.values[img])
    os.makedirs(folder)
    membrane = images[img, :, :, 1] + images[img, :, :, 2]
    io.imsave(os.path.join(folder, 'combined_membrane.tiff'), membrane)
    io.imsave(os.path.join(folder, 'DNA.tiff'), images[img, :, :, 0])

# generate crops
fovs = os.listdir(base_dir + 'selected_images')
fovs = [fov for fov in fovs if 'Point' in fov]

for fov in fovs:
    data = data_utils.load_imgs_from_dir(data_dir=os.path.join(base_dir, 'selected_images'),
                                         fovs=[fov], imgs=['DNA.tiff', 'combined_membrane.tiff'])

    cropped_data = data_utils.crop_image_stack(data.values, 512, 1)
    for crop in range(cropped_data.shape[0]):
        folder = os.path.join(base_dir, 'cropped/{}_crop_{}'.format(fov, crop))
        os.makedirs(folder)
        io.imsave(os.path.join(folder, 'DNA.tiff'), cropped_data[crop, :, :, 0])
        io.imsave(os.path.join(folder, 'Membrane.tiff'), cropped_data[crop, :, :, 1])




# NUH_DLBCL
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/20200416_NUH_DLBCL/'

folders = os.listdir(base_dir + '20200411 Unmixing Algorithm for CD20')
folders = [folder for folder in folders if '.jpg' in folder]

for folder in folders:
    folder_path = os.path.join(base_dir, 'FOVs', folder.split('.jpg')[0])
    os.makedirs(folder_path)
    jpg = io.imread(os.path.join(base_dir, '20200411 Unmixing Algorithm for CD20', folder))
    io.imsave(os.path.join(folder_path, 'Membrane.tiff'), jpg[:, :, 0].astype('float32'))

    jpg = io.imread(os.path.join(base_dir, '20200411 Unmixing Algorithm for DAPI', folder))
    io.imsave(os.path.join(folder_path, 'DNA.tiff'), jpg[:, :, 0].astype('float32'))


crop_dir = base_dir + 'Great/crops/'
fov_dir = base_dir + 'Great/FOVs/'
fovs = io_utils.list_folders(fov_dir)

for fov in fovs:
    data = data_utils.load_imgs_from_dir(fov_dir, fovs=[fov], dtype='float32',
                                         imgs=['DNA.tiff', 'Membrane.tiff'])

    cropped_data = data_utils.crop_image_stack(data, 512, 1)
    for crop in range(cropped_data.shape[0]):
        folder = os.path.join(crop_dir, '{}_crop_{}'.format(fov, crop))
        os.makedirs(folder)
        io.imsave(os.path.join(folder, 'DNA.tiff'), cropped_data[crop, :, :, 0])
        io.imsave(os.path.join(folder, 'Membrane.tiff'), cropped_data[crop, :, :, 1])


# La Jolla Institute for Immunology Tonsil
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/20200509_LJI_Tonsil/Great'
stacks = os.listdir(base_dir)
stacks = [stack for stack in stacks if 'tif' in stack]
data_utils.split_img_stack(base_dir, base_dir, stacks,  [0,2], ['DAPI.tif', 'CD4.tif'],
                           channels_first=False)

folder = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/20200509_LJI_Tonsil/Very_Dense/FOVs/Tile_14/'
DNA = io.imread(folder + 'DAPI.tif')
DNA = DNA[:, :2560]
io.imsave(folder + 'DAPI_cropped.tiff', DNA.astype('int32'))

Membrane = io.imread(folder + 'CD4.tif')
Membrane = Membrane[:, :2560]
io.imsave(folder + 'CD4_cropped.tiff', Membrane.astype('int32'))

crop_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/20200509_LJI_Tonsil/Very_Dense/crops/'
fov_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/20200509_LJI_Tonsil/Very_Dense/FOVs/'
fovs = os.listdir(fov_dir)
fovs = [fov for fov in fovs if 'Tile' in fov]

for fov in fovs:
    data = data_utils.load_imgs_from_dir(fov_dir, fovs=[fov], dtype='int32',
                                         imgs=['DAPI_cropped.tiff', 'CD4_cropped.tiff'])

    cropped_data = data_utils.crop_image_stack(data, 512, 1)
    for crop in range(cropped_data.shape[0]):
        folder = os.path.join(crop_dir, '{}_crop_{}'.format(fov, crop))
        os.makedirs(folder)
        io.imsave(os.path.join(folder, 'DNA.tiff'), cropped_data[crop, :, :, 0])
        io.imsave(os.path.join(folder, 'Membrane.tiff'), cropped_data[crop, :, :, 1])


# COH CRC processing
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/20200507_COH_CRC/'

fovs = io_utils.list_folders(base_dir + 'FOVs')

for idx, fov in enumerate(fovs):
    files = os.listdir(os.path.join(base_dir, 'FOVs', fov))
    files = [file for file in files if '.tif' in file]
    files.pop(np.where(np.isin(files, 'DAPI.tif'))[0][0])
    data = data_utils.load_imgs_from_dir(base_dir + 'FOVs', fovs=[fov], dtype='float32',
                                         imgs=['DAPI.tif', files[0]])

    cropped_data = data_utils.crop_image_stack(data, 512, 1)
    for crop in range(cropped_data.shape[0]):
        folder = os.path.join(base_dir, 'crops/img_{}_crop_{}'.format(idx, crop))
        os.makedirs(folder)
        io.imsave(os.path.join(folder, 'DNA.tiff'), cropped_data[crop, :, :, 0])
        io.imsave(os.path.join(folder, 'Membrane.tiff'), cropped_data[crop, :, :, 1])


# COH LN processing
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/20200508_COH_LN/'

fovs = io_utils.list_folders(base_dir + 'FOVs')

for idx, fov in enumerate(fovs):
    data = data_utils.load_imgs_from_dir(base_dir + 'FOVs', fovs=[fov], dtype='float32',
                                         imgs=['DAPI.tif', 'CD3.tif'])

    cropped_data = data_utils.crop_image_stack(data, 512, 1)
    for crop in range(cropped_data.shape[0]):
        folder = os.path.join(base_dir, 'crops/img_{}_crop_{}'.format(idx, crop))
        os.makedirs(folder)
        io.imsave(os.path.join(folder, 'DNA.tiff'), cropped_data[crop, :, :, 0])
        io.imsave(os.path.join(folder, 'Membrane.tiff'), cropped_data[crop, :, :, 1])


# COH BC processing
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/20200526_COH_BC/'
stacks = os.listdir(base_dir + 'fovs/great')
stacks = [stack for stack in stacks if 'tif' in stack]
data_utils.split_img_stack(base_dir + 'fovs/great', base_dir + 'fovs/great', stacks,  [0,2],
                           ['DAPI.tif', 'CK.tif'], channels_first=True)

fovs = os.listdir(base_dir + 'fovs/great')
fovs = [fov for fov in fovs if os.path.isdir(os.path.join(base_dir, 'fovs/great', fov))]

for fov in fovs:
    data = data_utils.load_imgs_from_dir(base_dir + 'fovs/great', fovs=[fov], dtype='float32',
                                         imgs=['DAPI.tif', 'CK.tif'])

    cropped_data = data_utils.crop_image_stack(data, 512, 1)
    for crop in range(cropped_data.shape[0]):
        folder = os.path.join(base_dir, 'cropped', '{}_crop_{}'.format(fov, crop))
        os.makedirs(folder)
        io.imsave(os.path.join(folder, 'DNA.tiff'), cropped_data[crop, :, :, 0])
        io.imsave(os.path.join(folder, 'Membrane.tiff'), cropped_data[crop, :, :, 1])

# remove invalid characters from folder name
fov_dir = os.path.join(base_dir, 'cropped')
fovs = os.listdir(fov_dir)
fovs = [fov for fov in fovs if os.path.isdir(os.path.join(fov_dir, fov))]

for fov in fovs:
    new_name = copy.copy(fov)
    new_name = new_name.replace(' ', '_')
    new_name = new_name.replace('[', '')
    new_name = new_name.replace(']', '')
    new_name = new_name.replace(',', '_')

    shutil.copytree(os.path.join(fov_dir, fov),
                    os.path.join(fov_dir, new_name))
    shutil.rmtree(os.path.join(fov_dir, fov))


# Travis labeled data processing

# fix idiotic google drive zip file architecture
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/20200512_Travis_PDAC/'

zip_folders = io_utils.list_folders(base_dir)

os.makedirs(os.path.join(base_dir, 'combined'))

for zip in zip_folders:
    image_folders = io_utils.list_folders(os.path.join(base_dir, zip))

    for image_folder in image_folders:
        image_folder_path = os.path.join(base_dir, 'combined', image_folder)
        if not os.path.isdir(image_folder_path):
            os.makedirs(image_folder_path)

        # check and see if original image and mask are in this version
        image_mask = os.path.join(base_dir, zip, image_folder, image_folder + '-Crop_Cell_Mask_Png.png')
        image_crop = os.path.join(base_dir, zip, image_folder, image_folder + '-Crop_Tif.tif')

        if os.path.exists(image_mask):
            shutil.copy(image_mask, os.path.join(image_folder_path, 'segmentation_label.png'))

        if os.path.exists(image_crop):
            shutil.copy(image_crop, os.path.join(image_folder_path, 'image_crop.tiff'))

# sum membrane channels together

combined_dir = os.path.join(base_dir, 'combined')
folders = io_utils.list_folders(combined_dir)

for folder in folders:
    total_tiff = io.imread(os.path.join(combined_dir, folder, 'image_crop.tiff'))
    DNA = total_tiff[5, :, :].astype('float32')
    Membrane =  np.sum(total_tiff[[1, 2, 4, 6], :, :], axis=0).astype('float32')
    io.imsave(os.path.join(combined_dir, folder, 'Membrane.tiff'), Membrane)
    io.imsave(os.path.join(combined_dir, folder, 'DNA.tiff'), DNA)

# load DNA and Membrane data
channel_data = data_utils.load_imgs_from_dir(data_dir=combined_dir,
                                             imgs=['DNA.tiff', 'Membrane.tiff'], dtype='float32')

label_data = data_utils.load_imgs_from_dir(data_dir=combined_dir, imgs=['segmentation_label.png'])

channel_data_resized = resize(channel_data.values, [24, 800, 800, 2], order=1)
label_data_resized = resize(label_data.values, [24, 800, 800, 1], order=0, preserve_range=True)

channel_data_resized = channel_data_resized[:, :768, :768, :]
label_data_resized = label_data_resized[:, :768, :768, :]

channel_data_cropped = data_utils.crop_image_stack(channel_data_resized, 256, 0.5)
label_data_cropped = data_utils.crop_image_stack(label_data_resized, 256, 0.5)

labeled_data = np.zeros_like(label_data_cropped)

for crop in range(labeled_data.shape[0]):
    labeled = label(label_data_cropped[crop, :, :, 0])
    labeled_data[crop, :, :, 0] = labeled
np.savez(combined_dir + '20200512_Travis_data.npz', X=channel_data_cropped, y=labeled_data)


# make labeled version at 512x512 resolution
labeled_data = np.zeros_like(label_data)
for crop in range(labeled_data.shape[0]):
    labeled = label(label_data[crop, :, :, 0])
    labeled_data[crop, :, :, 0] = labeled

new_labeled = np.zeros((24, 512, 512, 1))
new_labeled[:, :400, :400, :] = labeled_data

new_channel = np.zeros((24, 512, 512, 2))
new_channel[:, :400, :400, :] = channel_data
np.savez(combined_dir + '20200512_Travis_512x512.npz', X=new_channel, y=new_labeled)


# Magda processing
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/20200407_Magda/Bg_removed/'
all_imgs = data_utils.load_imgs_from_dir(data_dir=base_dir, img_sub_folder='TIFs',
                                         imgs=['CD3.tif', 'CD20.tif', 'CD45.tif', 'CD68.tif',
                                               'HH3.tif', 'PanCK.tif', 'CD4.tif'])

stitched_imgs = data_utils.stitch_images(all_imgs, 10)

for chan in range(stitched_imgs.shape[-1]):
    img = stitched_imgs[0, :, :, chan]
    io.imsave(os.path.join(base_dir, stitched_imgs.channels.values[chan] + '_stitched.tiff'), img)


# HIV preprocessing
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/20200520_HIV'
all_imgs = data_utils.load_imgs_from_dir(data_dir=base_dir)

stitched_imgs = data_utils.stitch_images(all_imgs, 5)

for chan in range(stitched_imgs.shape[-1]):
    img = stitched_imgs[0, :, :, chan]
    io.imsave(os.path.join(base_dir, stitched_imgs.channels.values[chan] + '_stitched.tiff'), img)

# after manually sorting into good and bad
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/20200520_HIV/Good_Membrane/'


fovs = os.listdir(base_dir + 'fovs')
fovs = [fov for fov in fovs if 'Point' in fov]

for fov in fovs:
    data = data_utils.load_imgs_from_dir(data_dir=os.path.join(base_dir, 'fovs'),
                                         fovs=[fov], imgs=['HH3.tif', 'CD45.tif'])

    cropped_data = data_utils.crop_image_stack(data.values, 512, 1)
    for crop in range(cropped_data.shape[0]):
        folder = os.path.join(base_dir, 'cropped/{}_crop_{}'.format(fov, crop))
        os.makedirs(folder)
        io.imsave(os.path.join(folder, 'DNA.tiff'), cropped_data[crop, :, :, 0])
        io.imsave(os.path.join(folder, 'Membrane.tiff'), cropped_data[crop, :, :, 1])


# TB preprocessing
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/20200424_TB/no_background/'
all_imgs = data_utils.load_imgs_from_dir(data_dir=base_dir + 'fovs', imgs=['HH3.tif', 'CD45.tif', 'HLA-Class-1.tif',
                                                                           'Keratin-pan.tif', 'Vimentin.tif'])

stitched_imgs = data_utils.stitch_images(all_imgs, 10)

for chan in range(stitched_imgs.shape[-1]):
    img = stitched_imgs[0, :, :, chan]
    io.imsave(os.path.join(base_dir + 'stitched', stitched_imgs.channels.values[chan] + '_stitched.tiff'), img)


# after manual classification
fovs = os.listdir(base_dir + 'fovs/good')
fovs = [fov for fov in fovs if 'Point' in fov]

for fov in fovs:
    data = data_utils.load_imgs_from_dir(data_dir=os.path.join(base_dir, 'fovs/good'),
                                         fovs=[fov], imgs=['HH3.tif', 'CD45.tif'])

    cropped_data = data_utils.crop_image_stack(data.values, 512, 1)
    for crop in range(cropped_data.shape[0]):
        folder = os.path.join(base_dir, 'cropped/{}_crop_{}'.format(fov, crop))
        os.makedirs(folder)
        io.imsave(os.path.join(folder, 'DNA.tiff'), cropped_data[crop, :, :, 0])
        io.imsave(os.path.join(folder, 'Membrane.tiff'), cropped_data[crop, :, :, 1])



# PICI processing

base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/20200615_PICI/_good/'
stacks = os.listdir(base_dir)
stacks = [stack for stack in stacks if '.tif' in stack]
data_utils.split_img_stack(stack_dir=base_dir,
                           output_dir=os.path.join(base_dir, 'fovs'),
                           stack_list=stacks,
                           indices=[0, 1, 2, 3, 4, 5, 6, 7],
                           names=['chan0.tiff', 'chan1.tiff', 'chan2.tiff', 'chan3.tiff',
                                  'chan4.tiff', 'chan5.tiff', 'chan6.tiff', 'chan7.tiff'])

# Sizun Epidermis processing
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/20200623_sizun_epidermis/'

all_tifs = os.listdir(base_dir + 'all_tifs')
for fov in range(1, 10):
    current_tifs = [tif for tif in all_tifs if 'Point_' + str(fov) in tif]
    selected_tifs = ['cadherin', 'CD138', 'DNA', 'H3K27', 'Keratin', 'NaK']
    current_folder = os.path.join(base_dir, 'fovs', 'fov{}'.format(fov))
    os.makedirs(current_folder)
    for tif in selected_tifs:
        tif_name = [img for img in current_tifs if tif in img]
        shutil.copy(os.path.join(base_dir, 'all_tifs', tif_name[0]),
                    os.path.join(current_folder, tif + '.tiff'))

# stitch images
all_imgs = data_utils.load_imgs_from_dir(data_dir=base_dir + 'fovs')

stitched_imgs = data_utils.stitch_images(all_imgs, 5)

for chan in range(stitched_imgs.shape[-1]):
    img = stitched_imgs[0, :, :, chan]
    io.imsave(os.path.join(base_dir + 'stitched', stitched_imgs.channels.values[chan] + '_stitched.tiff'), img)


# load HH3 and whatever membrane marker was manually selected
fovs = os.listdir(base_dir + 'fovs')
fovs = [fov for fov in fovs if 'fov' in fov]

for fov in fovs:
    imgs = os.listdir(os.path.join(base_dir, 'fovs', fov))
    imgs = [img for img in imgs if 'tif' in img]

    # remove DNA, remaining channel is membrane
    imgs.pop(np.where(np.isin(imgs, 'DNA.tiff'))[0][0])
    membrane_channel = imgs[0]

    data = data_utils.load_imgs_from_dir(data_dir=os.path.join(base_dir, 'fovs'),
                                         fovs=[fov], imgs=['DNA.tiff', membrane_channel])

    cropped_data = data_utils.crop_image_stack(data.values, 512, 1)
    for crop in range(cropped_data.shape[0]):
        folder = os.path.join(base_dir, 'cropped/{}_crop_{}'.format(fov, crop))
        os.makedirs(folder)
        io.imsave(os.path.join(folder, 'DNA.tiff'), cropped_data[crop, :, :, 0])
        io.imsave(os.path.join(folder, 'Membrane.tiff'), cropped_data[crop, :, :, 1])


# travis 20200626 processing

base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/20200626_travis/'

# get membrane channel from each tissue type
working_dir = base_dir + 'Bladder carcinoma'
stacks = os.listdir(working_dir)
stacks = [img for img in stacks if '.tif' in img]
data_utils.split_img_stack(stack_dir=working_dir,
                           output_dir=working_dir,
                           stack_list=stacks,
                           indices=[5, 6],
                           names=['PanCK.tiff', 'DAPI.tiff'])

# get membrane channel from each tissue type
working_dir = base_dir + 'Colon Carcinoma P20'
stacks = os.listdir(working_dir)
stacks = [img for img in stacks if '.tif' in img]
data_utils.split_img_stack(stack_dir=working_dir,
                           output_dir=working_dir,
                           stack_list=stacks,
                           indices=[5, 6],
                           names=['DAPI.tiff', 'PanCK.tiff'])


# get membrane channel from each tissue type
working_dir = base_dir + 'PDAC panel 20'
stacks = os.listdir(working_dir)
stacks = [img for img in stacks if '.tif' in img]
data_utils.split_img_stack(stack_dir=working_dir,
                           output_dir=working_dir,
                           stack_list=stacks,
                           indices=[5, 6],
                           names=['DAPI.tiff', 'PanCK.tiff'])


# get membrane channel from each tissue type
working_dir = base_dir + 'SCC H_N Linda Chen Panel 1'
stacks = os.listdir(working_dir)
stacks = [img for img in stacks if '.tif' in img]
data_utils.split_img_stack(stack_dir=working_dir,
                           output_dir=working_dir,
                           stack_list=stacks,
                           indices=[5, 6],
                           names=['PanCK.tiff', 'DAPI.tiff'])


# CODEX CRC preprocessing
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/20200627_CODEX_CRC/TMA_A/'
channel_names = pd.read_csv(base_dir + 'channelNames.txt', sep='\t', header=None)

stacks = os.listdir(base_dir + 'fovs')
stacks = [img for img in stacks if '.tif' in img]
channel_indices = [5, 13, 15, 17, 33, 37, 91]
names = ['CD44.tiff', 'CD45.tiff', 'BCat.tiff', 'HLA_2.tiff', 'NaK.tiff', 'PanCK.tiff',  'DRAQ5.tiff']

# convert index from numerical channel to position-based channel
for i in range(len(stacks)):
    img = io.imread(os.path.join(base_dir, 'fovs', stacks[i]))
    img_folder = os.path.join(base_dir, 'fovs', stacks[i].split('.tif')[0])
    os.makedirs(img_folder)
    for idx, val in enumerate(channel_indices):
        batch = val // 4
        channel = val % 4
        io.imsave(os.path.join(img_folder, names[idx]), img[batch, :, :, channel].astype('int32'))

# stitch images together for easy visualization
all_imgs = data_utils.load_imgs_from_dir(data_dir=base_dir + 'fovs', dtype='int32')

stitched_imgs = data_utils.stitch_images(all_imgs, 10)

for chan in range(stitched_imgs.shape[-1]):
    img = stitched_imgs[0, :, :, chan]
    io.imsave(os.path.join(base_dir + 'stitched', stitched_imgs.channels.values[chan] + '_stitched.tiff'), img)


# combine good FOVs from both TMAs into single folder
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/20200627_CODEX_CRC/'

TMA_B_fovs = io_utils.list_folders(base_dir + 'TMA_B/fovs/good')
for fov in TMA_B_fovs:
    shutil.copytree(os.path.join(base_dir, 'TMA_B/fovs/good', fov),
                    os.path.join(base_dir, 'selected', 'TMA_B_' + fov))


# add all membrane markers together into super membrane channel
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/20200627_CODEX_CRC/selected'
fovs = io_utils.list_folders(base_dir)

for fov in fovs:
    current_folder = os.path.join(base_dir, fov)
    imgs = os.listdir(current_folder)
    imgs = [img for img in imgs if '.tiff' in img and 'DRAQ' not in img]

    membrane = np.zeros((1440, 1920), dtype='float32')
    for img in imgs:
        img_file = io.imread(os.path.join(current_folder, img))
        membrane += img_file

    io.imsave(os.path.join(current_folder, 'Membrane.tiff'), membrane)

# crop images
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/20200627_CODEX_CRC/'
for fov in fovs:
    data = data_utils.load_imgs_from_dir(data_dir=os.path.join(base_dir, 'selected'),
                                         fovs=[fov], imgs=['DRAQ5.tiff', 'Membrane.tiff'],
                                         dtype='float32')

    cropped_data = data_utils.crop_image_stack(data.values, 512, 1)
    for crop in range(cropped_data.shape[0]):
        folder = os.path.join(base_dir, 'cropped/{}_crop_{}'.format(fov, crop))
        os.makedirs(folder)
        io.imsave(os.path.join(folder, 'DNA.tiff'), cropped_data[crop, :, :, 0])
        io.imsave(os.path.join(folder, 'Membrane.tiff'), cropped_data[crop, :, :, 1])

# CODEX TMA preprocessing
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/20200628_CODEX_TMA/'
channel_names = pd.read_csv(base_dir + 'channelNames.txt', sep='\t', header=None)

stacks = os.listdir(base_dir + 'fovs')
stacks = [img for img in stacks if '.tif' in img]
channel_indices = [17, 21, 33, 53, 57, 87]
names = ['CD45.tiff', 'Class_1.tiff', 'CK7.tiff', 'NaK.tiff', 'PanCK.tiff',  'DRAQ5.tiff']

# convert index from numerical channel to position-based channel
for i in range(len(stacks)):
    img = io.imread(os.path.join(base_dir, 'fovs', stacks[i]))
    img_folder = os.path.join(base_dir, 'fovs', stacks[i].split('.tif')[0])
    os.makedirs(img_folder)
    for idx, val in enumerate(channel_indices):
        batch = val // 4
        channel = val % 4
        io.imsave(os.path.join(img_folder, names[idx]), img[batch, :, :, channel].astype('int32'))

# stitch images together for easy visualization
all_imgs = data_utils.load_imgs_from_dir(data_dir=base_dir + 'fovs', dtype='int32')

stitched_imgs = data_utils.stitch_images(all_imgs, 10)

for chan in range(stitched_imgs.shape[-1]):
    img = stitched_imgs[0, :, :, chan]
    io.imsave(os.path.join(base_dir + 'stitched', stitched_imgs.channels.values[chan] + '_stitched.tiff'), img)

# MIBI SCC processing
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/20200625_SCC/'
folders = io_utils.list_folders(base_dir)

os.makedirs(os.path.join(base_dir, 'fovs'))

for folder in folders:
    new_folder = os.path.join(base_dir, 'fovs', folder)
    os.makedirs(new_folder)

    tifs = ['dsDNA.tiff', 'E-Cadherin.tiff', 'Histone H3.tiff', 'Pan-Keratin.tiff']
    for tif in tifs:
        shutil.copy(os.path.join(base_dir, folder, 'TIFsNoBg', tif),
                    os.path.join(new_folder, tif))


# stitch images together for easy visualization
all_imgs = data_utils.load_imgs_from_dir(data_dir=base_dir + 'fovs')

stitched_imgs = data_utils.stitch_images(all_imgs, 5)

for chan in range(stitched_imgs.shape[-1]):
    img = stitched_imgs[0, :, :, chan]
    io.imsave(os.path.join(base_dir + 'stitched', stitched_imgs.channels.values[chan] + '_stitched.tiff'), img)


# load selected fovs
fovs = io_utils.list_folders(base_dir + 'fovs/good')
os.makedirs(os.path.join(base_dir, 'fovs/good/cropped'))

for fov in fovs:
    data = data_utils.load_imgs_from_dir(data_dir=os.path.join(base_dir, 'fovs/good'),
                                         fovs=[fov], imgs=['Histone H3.tiff', 'E-Cadherin.tiff'])

    cropped_data = data_utils.crop_image_stack(data.values, 512, 1)
    for crop in range(cropped_data.shape[0]):
        folder = os.path.join(base_dir, 'fovs/good/cropped/{}_crop_{}'.format(fov, crop))
        os.makedirs(folder)
        io.imsave(os.path.join(folder, 'DNA.tiff'), cropped_data[crop, :, :, 0])
        io.imsave(os.path.join(folder, 'Membrane.tiff'), cropped_data[crop, :, :, 1])

# CODEX Pancreas preprocessing
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/20200624_graham_pancreas/'

stacks = os.listdir(base_dir + 'fovs')
stacks = [img for img in stacks if '.tif' in img]
channel_indices = [0, 1, 2, 3, 4, 5, 6]
names = ['CD45.tiff', 'NaK-ATPase.tiff', 'Ki-67.tiff', 'CD163.tiff', 'Glucagon.tiff',
         'Proinsulin.tiff', 'DRAQ5.tiff']

# convert index from numerical channel to position-based channel
data_utils.split_img_stack(stack_dir=base_dir + 'fovs',
                           output_dir=base_dir + 'fovs',
                           stack_list=stacks,
                           indices=channel_indices,
                           names=names)

# stitch images together for easy visualization
all_imgs = data_utils.load_imgs_from_dir(data_dir=base_dir + 'fovs', dtype='float16')

stitched_imgs = data_utils.stitch_images(all_imgs, 10)

for chan in range(stitched_imgs.shape[-1]):
    img = stitched_imgs[0, :, :, chan]
    io.imsave(os.path.join(base_dir + 'stitched', stitched_imgs.channels.values[chan] + '_125_stitched.tiff'), img.astype('float32'))

# add CD45 and NaK together into single membrane channel
# add all membrane markers together into super membrane channel
fovs = io_utils.list_folders(base_dir + 'fovs')

for fov in fovs:
    current_folder = os.path.join(base_dir, 'fovs', fov)
    NaK = io.imread(os.path.join(current_folder, 'NaK-ATPase.tiff')).astype('float32')
    CD45 = io.imread(os.path.join(current_folder, 'CD45.tiff')).astype('float32')
    membrane = NaK + CD45
    io.imsave(os.path.join(current_folder, 'Membrane.tiff'), membrane)



# create crops
os.makedirs(os.path.join(base_dir, 'cropped'))

for fov in fovs:
    data = data_utils.load_imgs_from_tree(data_dir=os.path.join(base_dir, 'fovs'),
                                         fovs=[fov], imgs=['DRAQ5.tiff', 'Membrane.tiff'],
                                          dtype='float32')

    cropped_data = data_utils.crop_image_stack(data.values, 512, 1)
    for crop in range(cropped_data.shape[0]):
        folder = os.path.join(base_dir, 'cropped/{}_crop_{}'.format(fov, crop))
        os.makedirs(folder)
        io.imsave(os.path.join(folder, 'DNA.tiff'), cropped_data[crop, :, :, 0])
        io.imsave(os.path.join(folder, 'Membrane.tiff'), cropped_data[crop, :, :, 1])
# CODEX normal colon preprocessing
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/20200707_CODEX_GI/'
channel_names = pd.read_csv(base_dir + 'channelNames_SB.txt', sep='\t', header=None)

stacks = os.listdir(base_dir + 'fovs')
stacks = [img for img in stacks if '.tif' in img]
channel_indices = [0, 46, 54]
# SB 2 sample panel offset by one cycle
channel_indices = [0, 50, 58]
names = ['HOESCHT1.tiff', 'CD45.tiff', 'Keratin.tiff']

# convert index from numerical channel to position-based channel
for i in range(len(stacks)):
    img = io.imread(os.path.join(base_dir, 'fovs', stacks[i]))
    img_folder = os.path.join(base_dir, 'fovs', stacks[i].split('.tif')[0])
    os.makedirs(img_folder)
    for idx, val in enumerate(channel_indices):
        batch = val // 4
        channel = val % 4
        current_img = img[batch, :, :, channel]
        while np.max(current_img) > 32000:
            current_img = current_img / 2
        io.imsave(os.path.join(img_folder, names[idx]), current_img.astype('int16'))

# stitch images together for easy visualization
all_imgs = data_utils.load_imgs_from_dir(data_dir=base_dir + 'fovs', dtype='int16')

stitched_imgs = data_utils.stitch_images(all_imgs, 9)

for chan in range(stitched_imgs.shape[-1]):
    img = stitched_imgs[0, :, :, chan]
    io.imsave(os.path.join(base_dir + 'stitched', stitched_imgs.channels.values[chan] + '_stitched.tiff'), img)


# combine good FOVs from both TMAs into single folder
CL_fovs = io_utils.list_folders(base_dir + '_useable')
for fov in CL_fovs:
    shutil.copytree(os.path.join(base_dir, '_useable', fov),
                    os.path.join(base_dir, 'fovs', 'CL_' + fov))


# add all membrane markers together into super membrane channel
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/20200627_CODEX_CRC/selected'
fovs = io_utils.list_folders(base_dir)

for fov in fovs:
    current_folder = os.path.join(base_dir, fov)
    imgs = os.listdir(current_folder)
    imgs = [img for img in imgs if '.tiff' in img and 'DRAQ' not in img]

    membrane = np.zeros((1440, 1920), dtype='float32')
    for img in imgs:
        img_file = io.imread(os.path.join(current_folder, img))
        membrane += img_file

    io.imsave(os.path.join(current_folder, 'Membrane.tiff'), membrane)

# crop images
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/20200627_CODEX_CRC/'
for fov in fovs:
    data = data_utils.load_imgs_from_dir(data_dir=os.path.join(base_dir, 'selected'),
                                         fovs=[fov], imgs=['DRAQ5.tiff', 'Membrane.tiff'],
                                         dtype='float32')

    cropped_data = data_utils.crop_image_stack(data.values, 512, 1)
    for crop in range(cropped_data.shape[0]):
        folder = os.path.join(base_dir, 'cropped/{}_crop_{}'.format(fov, crop))
        os.makedirs(folder)
        io.imsave(os.path.join(folder, 'DNA.tiff'), cropped_data[crop, :, :, 0])
        io.imsave(os.path.join(folder, 'Membrane.tiff'), cropped_data[crop, :, :, 1])

# phenotype preprocessing

# TNBC preprocessing
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/'

tnbc_dir = base_dir + '20200328_TNBC/fovs/new_good/'
fovs = data_utils.load_imgs_from_dir(tnbc_dir, imgs=['Beta catenin.tiff', 'Pan-Keratin.tiff',
                                                     'dsDNA.tiff', 'CD8.tiff', 'CD20.tiff',
                                                     'CD45.tiff', 'CD56.tiff'])

stitched_fovs = data_utils.stitch_images(fovs, 5)

for i in range(stitched_fovs.shape[-1]):
    io.imsave(tnbc_dir + stitched_fovs.channels.values[i] + '.tiff', stitched_fovs[0, :, :, i].values)


tifs = os.listdir('/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/data/datasets/20200328_TNBC/fovs')
tifs = [tif for tif in tifs if '.tif' in tif]
tifs.sort()

# DCIS preprocessing
base_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/analyses/20200811_subcellular_loc/DCIS'
fovs = io_utils.list_folders(base_dir, 'Point')

# copy selected membrane channel to membrane.tiff to make data loading easier
for fov in fovs:
    img_folder = os.path.join(base_dir, fov, 'segmentation_channels')
    imgs = io_utils.list_files(img_folder, '.tif')
    imgs.pop(np.where(np.isin(imgs, 'HH3.tif'))[0][0])

    shutil.copy(os.path.join(img_folder, imgs[0]),
                os.path.join(img_folder, 'membrane.tiff'))

