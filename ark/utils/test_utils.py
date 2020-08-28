import os
import numpy as np
import xarray as xr
import skimage.io as io

from mibidata import mibi_image as mi, tiff

# required metadata for mibitiff writing (barf)
METADATA = {
    'run': '20180703_1234_test', 'date': '2017-09-16T15:26:00',
    'coordinates': (12345, -67890), 'size': 500., 'slide': '857',
    'fov_id': 'Point1', 'fov_name': 'R1C3_Tonsil',
    'folder': 'Point1/RowNumber0/Depth_Profile0',
    'dwell': 4, 'scans': '0,5', 'aperture': 'B',
    'instrument': 'MIBIscope1', 'tissue': 'Tonsil',
    'panel': '20170916_1x', 'mass_offset': 0.1, 'mass_gain': 0.2,
    'time_resolution': 0.5, 'miscalibrated': False, 'check_reg': False,
    'filename': '20180703_1234_test', 'description': 'test image',
    'version': 'alpha',
}


def _gen_tif_data(fov_number, chan_number, img_shape, fills, dtype):
    if fills is None:
        tif_data = np.random.randint(0, 100,
                                     fov_number * img_shape[0] * img_shape[1] * chan_number)
        tif_data = tif_data.reshape((fov_number, *img_shape, chan_number)).astype(dtype)
    else:
        tif_data = np.full((*img_shape, fov_number, chan_number),
                           list(fills.values()), dtype=dtype)
        tif_data = np.moveaxis(tif_data, 2, 0)

    return tif_data


def _create_tifs(base_dir, fov_names, img_names, shape, sub_dir, fills, dtype):
    tif_data = _gen_tif_data(len(fov_names), len(img_names), shape, fills, dtype)

    if sub_dir is None:
        sub_dir = ""

    filelocs = {}

    for i, fov in enumerate(fov_names):
        filelocs[fov] = []
        fov_path = os.path.join(base_dir, fov, sub_dir)
        os.makedirs(fov_path)
        for j, name in enumerate(img_names):
            io.imsave(os.path.join(base_dir, name), tif_data[i, :, :, j])
            filelocs[fov].append(os.path.join(base_dir, fov, sub_dir, name))

    return filelocs, tif_data


def _create_multitiff(base_dir, fov_names, channel_names, shape, sub_dir, fills, dtype):
    tif_data = _gen_tif_data(len(fov_names), len(channel_names), shape, fills, dtype)

    filelocs = {}

    for i, fov in enumerate(fov_names):
        tiffpath = os.path.join(base_dir, f'{fov}.tiff')
        io.imsave(tiffpath, tif_data[i, :, :, :], plugin='tifffile')
        filelocs[fov] = tiffpath

    return filelocs, tif_data


def _create_mibitiff(base_dir, fov_names, channel_names, shape, sub_dir, fills, dtype):
    tif_data = _gen_tif_data(len(fov_names), len(channel_names), shape, fills, dtype)

    filelocs = {}

    mass_map = tuple(enumerate(channel_names, 1))

    for i, fov in enumerate(fov_names):
        tif_obj = mi.MibiImage(tif_data[i, :, :, :],
                               mass_map,
                               **METADATA)

        tiffpath = os.path.join(base_dir, f'{fov}.tiff')
        tiff.write(tiffpath, tif_obj, dtype=dtype)
        filelocs[fov] = tiffpath

    return filelocs, tif_data


TIFFMAKERS = {
    'tiff': _create_tifs,
    'multitiff': _create_multitiff,
    'mibitiff': _create_mibitiff,
}


def create_paired_xarray_fovs(base_dir, fov_names, channel_names, img_shape=(1024, 1024),
                              mode='tiff', delimiter=None, sub_dir=None, fills=None, dtype="int8"):

    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f'{base_dir} is not a directory')
        return None

    if fov_names is None or fov_names is []:
        raise ValueError('No fov names were given...')
        return None

    if channel_names is None or channel_names is []:
        raise ValueError('No image names were given...')
        return None

    if not isinstance(fov_names, list):
        fov_names = [fov_names]

    if not isinstance(channel_names, list):
        channel_names = [channel_names]

    # verify fills structure -> dict(fov_name : list [num_channels])
    if fills is not None:
        if fills.keys() is not fov_names:
            raise ValueError(f'fills keys, {fills.keys()}, must match fov_names, {fov_names}')
            return None
        for fill in fills.values():
            if len(fill) != len(channel_names):
                raise ValueError(f'Not all fills have length {len(channel_names)}')
                return None

    if fills is not None and not isinstance(fills, list):
        fills = [fills]

    if fills is not None and (len(fills) != len(channel_names)):
        raise ValueError('channel_names and fills must be the same length')
        return None

    filelocs, tif_data = TIFFMAKERS[mode](base_dir, fov_names, channel_names, img_shape, sub_dir,
                                          fills, dtype)

    if delimiter is not None:
        fov_ids = [fov.split(delimiter)[0] for fov in fov_names]

    data_xr = xr.DataArray(tif_data,
                           coords=[fov_ids,
                                   range(img_shape[0]),
                                   range(img_shape[1]),
                                   channel_names],
                           dims=["fovs", "rows", "cols", "channels"])

    return filelocs, data_xr


def clear_directory(dirpath):
    if not os.path.isdir(dirpath):
        raise FileNotFoundError(f'{dirpath} is not a valid directory')
        return

    for subpath in os.listdir(dirpath):
        if os.path.isfile(subpath):
            os.remove(subpath)
        if os.path.isdir(subpath):
            os.rmdir(subpath)


def _create_img_dir(temp_dir, fovs, imgs, img_sub_folder="TIFs", dtype="int8"):
    tif = np.random.randint(0, 100, 1024 ** 2).reshape((1024, 1024)).astype(dtype)

    for fov in fovs:
        fov_path = os.path.join(temp_dir, fov, img_sub_folder)
        if not os.path.exists(fov_path):
            os.makedirs(fov_path)
        for img in imgs:
            io.imsave(os.path.join(fov_path, img), tif)