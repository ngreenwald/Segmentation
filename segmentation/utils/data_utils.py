import os
import math
import warnings

import skimage.io as io
import numpy as np
import xarray as xr

import skimage.filters.rank as rank
import scipy.ndimage as nd


def load_imgs_from_dir(data_dir, img_sub_folder=None, folder_names=None, imgs=None, load_axis="fovs", dtype="int16"):
    """Takes a set of imgs from a directory structure and loads them into a numpy array.

        Args:
            data_dir: directory containing folders of images
            img_sub_folder: optional name of sub-folder containing the images within each identifier folder
            folder_names: optional list of folder_names to load imgs from, otherwise loads from all folders
            imgs: optional list of imgs to load, otherwise loads all .tif, .tiff, .jpg, or .png files
            load_axis: axis that images will get loaded onto. Must be one of ["fovs", "stacks"]
            dtype: dtype of array which will be used to store values

        Returns:
            Numpy array with shape [fovs, tifs, x_dim, y_dim]
    """

    if not os.path.isdir(data_dir):
        raise ValueError("Directory does not exist")

    if not np.isin(load_axis, ["fovs", "stacks"]):
        raise ValueError("Invalid value for load_axis, must be one of [fovs, stacks]")

    if folder_names is None:
        # get all point folders
        folder_names = os.listdir(data_dir)
        folder_names = [folder for folder in folder_names if os.path.isdir(os.path.join(data_dir, folder))]
        folder_names.sort()
    else:
        # use supplied list, but check to make sure they all exist
        for folder in folder_names:
            if not os.path.isdir(os.path.join(data_dir, folder)):
                raise ValueError("Could not find img folder {}".format(folder))

    if len(folder_names) == 0:
        raise ValueError("No points found in directory")

    # check to make sure img subfolder name within point directory is correct
    if img_sub_folder is not None:
        if not os.path.isdir(os.path.join(data_dir, folder_names[0], img_sub_folder)):
            raise ValueError("Invalid img_sub_folder name")
    else:
        # no img_sub_folder, change to empty string to read directly from base folder
        img_sub_folder = ""

    # get imgs from first point directory if no img names supplied
    if imgs is None:
        imgs = os.listdir(os.path.join(data_dir, folder_names[0], img_sub_folder))
        imgs = [img for img in imgs if np.isin(img.split(".")[-1], ["tif", "tiff", "jpg", "png"])]

    if len(imgs) == 0:
        raise ValueError("No imgs found in designated folder")

    # check to make sure supplied imgs exist
    for img in imgs:
        if not os.path.isfile(os.path.join(data_dir, folder_names[0], img_sub_folder, img)):
            raise ValueError("Could not find {} in supplied directory {}".format(img, os.path.join(data_dir, folder_names[0], img_sub_folder, img)))

    test_img = io.imread(os.path.join(data_dir, folder_names[0], img_sub_folder, imgs[0]))

    # check to make sure that float dtype was supplied if image data is float
    data_dtype = test_img.dtype
    if np.issubdtype(data_dtype, np.floating):
        if not np.issubdtype(dtype, np.floating):
            raise ValueError("supplied dtype is not a float, but the images loaded are floats")

    img_data = np.zeros((len(folder_names), test_img.shape[0], test_img.shape[1], len(imgs)), dtype=dtype)

    for folder in range(len(folder_names)):
        for img in range(len(imgs)):
            img_data[folder, :, :, img] = io.imread(os.path.join(data_dir, folder_names[folder],
                                                                 img_sub_folder, imgs[img]))

    # check to make sure that dtype wasn't too small for range of data
    if np.min(img_data) < 0:
        raise ValueError("Integer overflow from loading TIF image, try a larger dtype")

    # remove .tif or .tiff from image name
    img_names = [os.path.splitext(img)[0] for img in imgs]

    img_xr = xr.DataArray(img_data, coords=[folder_names, range(test_img.shape[0]), range(test_img.shape[1]), img_names],
                          dims=[load_axis, "rows", "cols", "channels"])

    return img_xr


def create_blank_channel(img_size, grid_size, dtype):
    """Creates a blank TIF of a given size that has a small number of positive pixels to avoid divide by zero errors
    Inputs:
        img_size: tuple specifying the size of the image to create
        grid_size: int that determines how many pieces to randomize within
        dtype: dtype for image

    Outputs:
        blank_arr: a (mostly) blank array with positive pixels in random values
    """

    blank = np.zeros(img_size, dtype=dtype)
    row_step = math.floor(blank.shape[0] / grid_size)
    col_step = math.floor(blank.shape[1] / grid_size)

    for row in range(grid_size):
        for col in range(grid_size):
            row_rand = np.random.randint(0, row_step - 1)
            col_rand = np.random.randint(0, col_step - 1)
            blank[row * row_step + row_rand, col * col_step + col_rand] = np.random.randint(1, 15)

    return blank


def reorder_xarray_channels(channel_order, channel_xr, non_blank_channels=None):
    """Adds blank channels or changes the order of existing channels to match the ordering given by channel_order list
    Inputs:
        channel_order: list of channel names, which dictates final order of output xarray
        channel_xr: xarray containing the channel data for the available channels
        non_blank_channels: optional list of channels which aren't missing, and hence won't be replaced with blank tif:
            if not supplied, will default to assuming all channels in channel_order

    Outputs:
        xarray with the supplied channels in channel order"""

    if non_blank_channels is None:
        non_blank_channels = channel_order

    # error checking
    channels_in_xr = np.isin(non_blank_channels, channel_xr.channels)
    if len(channels_in_xr) != np.sum(channels_in_xr):
        bad_chan = non_blank_channels[np.where(~channels_in_xr)[0][0]]
        raise ValueError("{} was listed as a non-blank channel, but it is not in the channel xarray".format(bad_chan))

    channels_in_order = np.isin(non_blank_channels, channel_order)
    if len(channels_in_order) != np.sum(channels_in_order):
        bad_chan = non_blank_channels[np.where(~channels_in_order)[0][0]]
        raise ValueError("{} was listed as a non-blank channel, but it is not in the channel order".format(bad_chan))

    vals, counts = np.unique(channel_order, return_counts=True)
    duplicated = np.where(counts > 1)
    if len(duplicated[0] > 0):
        raise ValueError("The following channels are duplicated in the channel order: {}".format(vals[duplicated[0]]))

    vals, counts = np.unique(channel_xr.channels.values, return_counts=True)
    duplicated = np.where(counts > 1)
    if len(duplicated[0] > 0):
        raise ValueError("The following channels are duplicated in the xarray: {}".format(vals[duplicated[0]]))

    # create array to hold all channels, including blank ones
    full_array = np.zeros((channel_xr.shape[:3] + (len(channel_order),)), dtype=channel_xr.dtype)
    print(full_array.shape)

    for i in range(len(channel_order)):
        if channel_order[i] in non_blank_channels:
            current_channel = channel_xr.loc[:, :, :, channel_order[i]].values
            print(current_channel.shape)
            print(channel_order[i])
            full_array[:, :, :, i] = current_channel
        else:
            im_crops = channel_xr.shape[1] // 32
            blank = create_blank_channel(channel_xr.shape[1:3], im_crops, dtype=channel_xr.dtype)
            full_array[:, :, :, i] = blank

    channel_xr_blanked = xr.DataArray(full_array, coords=[channel_xr.points, range(channel_xr.shape[1]),
                                                          range(channel_xr.shape[2]), channel_order],
                                      dims=["points", "rows", "cols", "channels"])

    return channel_xr_blanked


def combine_xarrays(xarrays, axis):
    """Combines a number of xarrays together

    Inputs:
        xarrays: a tuple of xarrays
        axis: either 0, if the xarrays will combined over different points, or -1, if they will be combined over channels

    Outputs:
        combined_xr: an xarray that is the combination of all inputs"""

    first_xr = xarrays[0]
    np_arr = first_xr.values

    # define iterator to hold coord values of dimension that is being stacked
    if axis == 0:
        iterator = first_xr.points.values
        shape_slice = slice(1, 4)
    else:
        iterator = first_xr.channels.values
        shape_slice = slice(0, 3)

    # loop through each xarray, stack the coords, and concatenate the values
    for cur_xr in xarrays[1:]:
        cur_arr = cur_xr.values

        if cur_arr.shape[shape_slice] != first_xr.shape[shape_slice]:
            raise ValueError("xarrays have conflicting sizes")

        if axis == 0:
            if not np.array_equal(cur_xr.channels, first_xr.channels):
                raise ValueError("xarrays have different channel names")
        else:
            if not np.array_equal(cur_xr.points, first_xr.points):
                raise ValueError("xarrays have different point names")

        np_arr = np.concatenate((np_arr, cur_arr), axis=axis)
        if axis == 0:
            iterator = np.append(iterator, cur_xr.points.values)
        else:
            iterator = np.append(iterator, cur_xr.channels.values)

    # assign iterator to appropriate coord label
    if axis == 0:
        points = iterator
        channels = first_xr.channels.values
    else:
        points = first_xr.points
        channels = iterator

    combined_xr = xr.DataArray(np_arr, coords=[points, range(first_xr.shape[1]), range(first_xr.shape[2]), channels],
                               dims=["points", "rows", "cols", "channels"])

    return combined_xr


def pad_xr_dims(input_xr, padded_dims):
    """Takes an xarray and pads it with dimensions of size 1 according to the supplied dims list

    Inputs
        input_xr: xarray to padd
        padded_dims: list of dims to be included in output xarray

    Outputs
        padded_xr: xarray that has had additional dims added of size 1"""

    # make sure that dimensions which are present in both lists are in same order
    old_dims = [dim for dim in padded_dims if dim in input_xr.dims]

    if not old_dims == list(input_xr.dims):
        raise ValueError("existing dimensions must be in same order in input_dims list")

    # create new output data
    output_vals = input_xr.values
    output_coords = []

    for idx, dim in enumerate(padded_dims):

        if dim in input_xr.dims:
            # dimension already exists, using existing values and coords
            output_coords.append(input_xr[dim])
        else:
            output_vals = np.expand_dims(output_vals, axis=idx)
            output_coords.append(range(1))

    padded_xr = xr.DataArray(output_vals, coords=output_coords, dims=padded_dims)

    return padded_xr


def crop_helper(image_stack, crop_size):
    """"Helper function to take an image, and return crops of size crop_size

    Inputs:
        image_stack (np.array): A 4D numpy array of shape(points, rows, columns, channels)
        crop_size (int): Size of the crop to take from the image. Assumes square crops

    Outputs:
        cropped_images (np.array): A 4D numpy array of shape (crops, rows, columns, channels)"""

    if len(image_stack.shape) != 4:
        raise ValueError("Incorrect dimensions of input image. Expecting 3D, got {}".format(image_stack.shape))

    # figure out number of crops for final image
    crop_num_row = math.ceil(image_stack.shape[1] / crop_size)
    crop_num_col = math.ceil(image_stack.shape[2] / crop_size)
    cropped_images = np.zeros((crop_num_row * crop_num_col * image_stack.shape[0], crop_size, crop_size,
                               image_stack.shape[3]), dtype=image_stack.dtype)

    # Determine if image will need to be padded with zeros due to uneven division by crop
    if image_stack.shape[1] % crop_size != 0 or image_stack.shape[2] % crop_size != 0:
        # create new array that is padded by one crop size on image dimensions
        new_shape = image_stack.shape[0], image_stack.shape[1] + crop_size, image_stack.shape[2] + crop_size, image_stack.shape[3]
        new_stack = np.zeros(new_shape, dtype=image_stack.dtype)
        new_stack[:, :image_stack.shape[1], :image_stack.shape[2], :] = image_stack
        image_stack = new_stack

    # iterate through the image row by row, cropping based on identified threshold
    img_idx = 0
    for point in range(image_stack.shape[0]):
        for row in range(crop_num_row):
            for col in range(crop_num_col):
                cropped_images[img_idx, :, :, :] = image_stack[point, (row * crop_size):((row + 1) * crop_size),
                                                       (col * crop_size):((col + 1) * crop_size), :]
                img_idx += 1

    return cropped_images


def crop_image_stack(image_stack, crop_size, stride_fraction):
    """Function to generate a series of tiled crops across an image. The tiled crops can overlap each other, with the
       overlap between tiles determined by the stride fraction. A stride fraction of 0.333 will move the window over
       1/3 of the crop_size in x and y at each step, whereas a stride fraction of 1 will move the window the entire crop
       size at each iteration.

    Inputs:
        image_stack (np.array): A 4D numpy array of shape(points, rows, columns, channels)
        crop_size (int): size of the crop to take from the image. Assumes square crops
        stride_fraction (float): the relative size of the stride for overlapping crops as a function of
        the crop size.
    Outputs:
        cropped_images (np.array): A 4D numpy array of shape(crops, rows, cols, channels)"""

    if len(image_stack.shape) != 4:
        raise ValueError("Incorrect dimensions of input image. Expecting 3D, got {}".format(image_stack.shape))

    if crop_size > image_stack.shape[1]:
        raise ValueError("Invalid crop size: img shape is {} and crop size is {}".format(image_stack.shape, crop_size))

    if stride_fraction > 1:
        raise ValueError("Invalid stride fraction. Must be less than 1, passed a value of {}".format(stride_fraction))

    # Determine how many distinct grids will be generated across the image
    stride_step = math.floor(crop_size * stride_fraction)
    num_strides = math.floor(1 / stride_fraction)

    for row_shift in range(num_strides):
        for col_shift in range(num_strides):

            if row_shift == 0 and col_shift == 0:
                # declare data holder
                cropped_images = crop_helper(image_stack, crop_size)
            else:
                # crop the image by the shift prior to generating grid of crops
                img_shift = image_stack[:, (row_shift * stride_step):, (col_shift * stride_step):, :]
                # print("shape of the input image is {}".format(img_shift.shape))
                temp_images = crop_helper(img_shift, crop_size)
                cropped_images = np.append(cropped_images, temp_images, axis=0)

    return cropped_images


def combine_point_directories(dir_path):
    """Combines a folder containing multiple imaging runs into a single folder

    Inputs
        dir_path: path to directory containing the sub directories

    Outputs
        None"""

    if not os.path.exists(dir_path):
        raise ValueError("Directory does not exist")

    # gets all sub folders
    folders = os.listdir(dir_path)
    folders = [folder for folder in folders if os.path.isdir(os.path.join(dir_path, folder))]

    os.makedirs(os.path.join(dir_path, "combined_folder"))

    # loop through sub folders, get all contents, and transfer to new folder
    for folder in folders:
        points = os.listdir(os.path.join(dir_path, folder))
        print(points)
        for point in points:
            os.rename(os.path.join(dir_path, folder, point), os.path.join(dir_path, "combined_folder", folder + "_" + point))

