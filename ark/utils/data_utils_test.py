import numpy as np
import os
import pytest
import tempfile
from shutil import rmtree

from ark.utils import data_utils, test_utils
import skimage.io as io


def test_load_imgs_from_mibitiff():

    with tempfile.TemporaryDirectory() as temp_dir:

        # config test environment
        fovs = ["Point8_otherinfo", "Point9"]
        channels = ["HH3", "Membrane", "Other"]

        filelocs, data_xr = test_utils.create_paired_xarray_fovs(
            temp_dir, fovs, channels, img_shape=(10, 10), mode='mibitiff', delimiter='_',
            fills=True, dtype=np.float32
        )

        # check unspecified point loading
        loaded_xr = data_utils.load_imgs_from_mibitiff(temp_dir,
                                                       channels=channels,
                                                       delimiter='_')

        assert test_utils.xarrays_are_equal(data_xr, loaded_xr)

        fovnames = [f'{fov}.tiff' for fov in fovs]

        # check specified point loading
        loaded_xr = data_utils.load_imgs_from_mibitiff(temp_dir,
                                                       mibitiff_files=[fovnames[-1]],
                                                       channels=channels,
                                                       delimiter='_')

        assert test_utils.xarrays_are_equal(data_xr.loc[[fovs[-1]], :, :, :], loaded_xr)

        # test automatic all channels loading
        loaded_xr = data_utils.load_imgs_from_mibitiff(temp_dir,
                                                       delimiter='_',
                                                       dtype=np.float32)

        assert test_utils.xarrays_are_equal(data_xr, loaded_xr)

        # test delimiter agnosticism
        loaded_xr = data_utils.load_imgs_from_mibitiff(temp_dir,
                                                       mibitiff_files=fovnames,
                                                       channels=channels,
                                                       delimiter='_',
                                                       dtype=np.float32)

        assert test_utils.xarrays_are_equal(data_xr, loaded_xr)
        assert np.issubdtype(loaded_xr.dtype, np.floating)

        # test float overwrite
        with pytest.warns(UserWarning):
            loaded_xr = data_utils.load_imgs_from_mibitiff(temp_dir,
                                                           mibitiff_files=[fovnames[-1]],
                                                           channels=channels,
                                                           delimiter='_',
                                                           dtype='int16')

            assert test_utils.xarrays_are_equal(data_xr.loc[[fovs[-1]], :, :, :], loaded_xr)
            assert np.issubdtype(loaded_xr.dtype, np.floating)


def test_load_imgs_from_multitiff():

    with tempfile.TemporaryDirectory() as temp_dir:
        # config test environment
        fovs = ["Point8_otherinfo", "Point9"]
        channels = ["HH3", "Membrane", "Other"]

        filelocs, data_xr = test_utils.create_paired_xarray_fovs(
            temp_dir, fovs, channels, img_shape=(10, 10), mode='multitiff', delimiter='_',
            fills=True, dtype=np.float32
        )

        fovnames = [f'{fov}.tiff' for fov in fovs]

        # test all channels loading w/ specified file
        loaded_xr = data_utils.load_imgs_from_multitiff(temp_dir,
                                                        multitiff_files=[fovnames[-1]],
                                                        delimiter='_')

        assert test_utils.xarrays_are_equal(data_xr.loc[[fovs[-1]], :, :, :], loaded_xr)

        # test single channel load
        loaded_xr = data_utils.load_imgs_from_multitiff(temp_dir,
                                                        multitiff_files=fovnames,
                                                        channels=[0],
                                                        delimiter='_')

        assert test_utils.xarrays_are_equal(data_xr.loc[:, :, :, [0]], loaded_xr)

        # test all channels w/ unspecified files + delimiter agnosticism
        loaded_xr = data_utils.load_imgs_from_multitiff(temp_dir,
                                                        multitiff_files=None,
                                                        channels=None,
                                                        delimiter='_')

        assert test_utils.xarrays_are_equal(data_xr, loaded_xr)

        # test float overwrite
        with pytest.warns(UserWarning):
            loaded_xr = data_utils.load_imgs_from_multitiff(temp_dir,
                                                            delimiter='_',
                                                            dtype='int16')

            assert test_utils.xarrays_are_equal(data_xr, loaded_xr)
            assert(np.issubdtype(loaded_xr.dtype, np.floating))


def test_load_imgs_from_tree():
    # test loading from within fov directories
    with tempfile.TemporaryDirectory() as temp_dir:
        fovs = ["fov1", "fov2", "fov3"]
        imgs = ["img1.tiff", "img2.tiff", "img3.tiff"]
        chans = [chan.split(".tiff")[0] for chan in imgs]

        filelocs, data_xr = test_utils.create_paired_xarray_fovs(
            temp_dir, fovs, chans, img_shape=(10, 10), delimiter='_', fills=True, sub_dir="TIFs",
            dtype="int16"
        )

        # check default loading of all files
        loaded_xr = \
            data_utils.load_imgs_from_tree(temp_dir, img_sub_folder="TIFs", dtype="int16")

        assert test_utils.xarrays_are_equal(data_xr, loaded_xr)

        # check loading of specific files
        some_fovs = fovs[:2]
        some_imgs = imgs[:2]
        some_chans = chans[:2]

        loaded_xr = \
            data_utils.load_imgs_from_tree(temp_dir, img_sub_folder="TIFs", dtype="int16",
                                           fovs=some_fovs, channels=some_imgs)

        assert test_utils.xarrays_are_equal(
            data_xr[:2, :, :, :2], loaded_xr
        )

        # check loading w/o file extension
        loaded_xr = \
            data_utils.load_imgs_from_tree(temp_dir, img_sub_folder="TIFs", dtype="int16",
                                           channels=some_chans)

        assert test_utils.xarrays_are_equal(
            data_xr[:, :, :, :2], loaded_xr
        )

        # check mixed extension presence
        loaded_xr = \
            data_utils.load_imgs_from_tree(temp_dir, img_sub_folder="TIFs", dtype="int16",
                                           channels=[chans[i] if i % 2 else imgs[i]
                                                     for i in range(3)])

        assert test_utils.xarrays_are_equal(data_xr, loaded_xr)

    with tempfile.TemporaryDirectory() as temp_dir:

        fovs = ["fov1"]
        imgs = ["img1.tiff", "img2.tiff"]
        chans = [chan.split(".tiff")[0] for chan in imgs]

        filelocs, data_xr = test_utils.create_paired_xarray_fovs(
            temp_dir, fovs, chans, img_shape=(10, 10), delimiter='_', fills=True, sub_dir="TIFs",
            dtype=np.float32
        )

        with pytest.warns(UserWarning):
            loaded_xr = \
                data_utils.load_imgs_from_tree(temp_dir, img_sub_folder="TIFs", dtype="int16")

            assert test_utils.xarrays_are_equal(data_xr, loaded_xr)

            # test swap int16 -> float
            assert np.issubdtype(loaded_xr.dtype, np.floating)


def test_load_imgs_from_dir():
    # test loading from 'free' directory
    with tempfile.TemporaryDirectory() as temp_dir:
        imgs = ["fov1_img1.tiff", "fov2_img2.tiff", "fov3_img3.tiff"]
        fovnames = [img.split(".")[0] for img in imgs]
        filelocs, data_xr = test_utils.create_paired_xarray_fovs(temp_dir, fovnames, 'img_data',
                                                                 img_shape=(10, 10), mode='labels',
                                                                 delimiter='_', dtype=np.float32)

        # check default loading
        loaded_xr = \
            data_utils.load_imgs_from_dir(temp_dir, delimiter='_', dtype=np.float32)

        assert test_utils.xarrays_are_equal(data_xr, loaded_xr, sortdims=("fovs", "compartments"))

        # test swap int16 -> float
        with pytest.warns(UserWarning):
            loaded_xr = \
                data_utils.load_imgs_from_dir(temp_dir, delimiter='_', dtype="int16")

            assert test_utils.xarrays_are_equal(data_xr, loaded_xr,
                                                sortdims=("fovs", "compartments"))
            assert np.issubdtype(loaded_xr.dtype, np.floating)


def test_generate_deepcell_input():
    with tempfile.TemporaryDirectory() as temp_dir:
        fovs = ['fov1', 'fov2']
        chans = ['nuc1', 'nuc2', 'mem1', 'mem2']

        data_xr = test_utils.make_images_xarray(tif_data=None, fov_ids=fovs, row_size=10,
                                                col_size=10, channel_names=chans, dtype='int16')

        fov1path = os.path.join(temp_dir, 'fov1.tif')
        fov2path = os.path.join(temp_dir, 'fov2.tif')

        # test 1 nuc, 1 mem (no summing)
        nucs = ['nuc2']
        mems = ['mem2']

        data_utils.generate_deepcell_input(data_xr, temp_dir, nucs, mems)
        fov1 = io.imread(fov1path)
        fov2 = io.imread(fov2path)

        assert np.array_equal(fov1, data_xr.loc['fov1', :, :, ['nuc2', 'mem2']].values)
        assert np.array_equal(fov2, data_xr.loc['fov2', :, :, ['nuc2', 'mem2']].values)

        # test 2 nuc, 2 mem (summing)
        nucs = ['nuc1', 'nuc2']
        mems = ['mem1', 'mem2']

        data_utils.generate_deepcell_input(data_xr, temp_dir, nucs, mems)
        fov1 = io.imread(fov1path)
        fov2 = io.imread(fov2path)

        nuc_sums = data_xr.loc[:, :, :, nucs].sum(dim='channels').values
        mem_sums = data_xr.loc[:, :, :, mems].sum(dim='channels').values

        assert np.array_equal(fov1[:, :, 0], nuc_sums[0, :, :])
        assert np.array_equal(fov1[:, :, 1], mem_sums[0, :, :])
        assert np.array_equal(fov2[:, :, 0], nuc_sums[1, :, :])
        assert np.array_equal(fov2[:, :, 1], mem_sums[1, :, :])

        # test nuc None
        nucs = None

        data_utils.generate_deepcell_input(data_xr, temp_dir, nucs, mems)
        fov1 = io.imread(fov1path)
        fov2 = io.imread(fov2path)

        assert np.array_equal(fov1[:, :, 0], 0)
        assert np.array_equal(fov1[:, :, 1], mem_sums[0, :, :])
        assert np.array_equal(fov2[:, :, 0], 0)
        assert np.array_equal(fov2[:, :, 1], mem_sums[1, :, :])

        # test mem None
        nucs = ['nuc2']
        mems = None

        data_utils.generate_deepcell_input(data_xr, temp_dir, nucs, mems)
        fov1 = io.imread(fov1path)
        fov2 = io.imread(fov2path)

        assert np.array_equal(fov1[:, :, 1], 0)
        assert np.array_equal(fov1[:, :, 0], data_xr.loc['fov1', :, :, 'nuc2'].values)
        assert np.array_equal(fov2[:, :, 1], 0)
        assert np.array_equal(fov2[:, :, 0], data_xr.loc['fov2', :, :, 'nuc2'].values)


def test_combine_xarrays():
    # test combining along points axis
    fov_ids = [f'Point{i}' for i in range(5)]
    chan_ids = [f'chan{i}' for i in range(3)]

    base_xr = test_utils.make_images_xarray(tif_data=None, fov_ids=fov_ids, row_size=30,
                                            col_size=30, channel_names=chan_ids)

    test_xr = data_utils.combine_xarrays((base_xr[:3, :, :, :], base_xr[3:, :, :, :]), axis=0)
    assert test_utils.xarrays_are_equal(base_xr, test_xr)

    # test combining along channels axis
    fov_ids = [f'Point{i}' for i in range(3)]
    chan_ids = [f'chan{i}' for i in range(5)]

    base_xr = test_utils.make_images_xarray(
        None, fov_ids, 30, 30, chan_ids
    )

    test_xr = data_utils.combine_xarrays((base_xr[:, :, :, :3], base_xr[:, :, :, 3:]), axis=-1)
    assert test_utils.xarrays_are_equal(base_xr, test_xr)


def test_crop_helper():
    # test crops that divide evenly
    crop_input = np.zeros((4, 256, 256, 4))
    crop_size = 64

    cropped = data_utils.crop_helper(crop_input, crop_size)
    num_crops = crop_input.shape[0] * \
        (crop_input.shape[1] // crop_size) * (crop_input.shape[2] // crop_size)
    assert np.array_equal(cropped.shape, (num_crops, crop_size, crop_size, crop_input.shape[3]))

    # test crops that don't divide evenly
    crop_input = np.zeros((4, 256, 256, 4))
    crop_size = 56

    cropped = data_utils.crop_helper(crop_input, crop_size)
    num_crops = crop_input.shape[0] * \
        ((crop_input.shape[1] + crop_size) // crop_size) * \
        ((crop_input.shape[2] + crop_size) // crop_size)
    assert np.array_equal(cropped.shape, (num_crops, crop_size, crop_size, crop_input.shape[3]))


def test_crop_image_stack():
    # test without overlap (stride_fraction = 1)
    crop_input = np.zeros((4, 256, 256, 4), dtype="int16")
    crop_size = 64
    stride_fraction = 1

    cropped = data_utils.crop_image_stack(crop_input, crop_size, stride_fraction)
    num_crops = crop_input.shape[0] * (crop_input.shape[1] // crop_size) * \
        (crop_input.shape[2] // crop_size) * (1 / stride_fraction)

    assert np.array_equal(cropped.shape, (num_crops, crop_size, crop_size, crop_input.shape[3]))

    # test with overlap
    crop_input = np.zeros((4, 256, 256, 4), dtype="int16")
    crop_size = 64
    stride_fraction = 0.25

    cropped = data_utils.crop_image_stack(crop_input, crop_size, stride_fraction)
    num_crops = crop_input.shape[0] * (crop_input.shape[1] // crop_size) * \
        (crop_input.shape[2] // crop_size) * (1 / stride_fraction) * (1 / stride_fraction)

    assert np.array_equal(cropped.shape, (num_crops, crop_size, crop_size, crop_input.shape[3]))


def test_combine_point_directories():
    # first test the case where the directory specified doesn't exist
    with pytest.raises(ValueError):
        data_utils.combine_point_directories(os.path.join("path", "to", "undefined", "folder"))

    # now we do the "real" testing...
    with tempfile.TemporaryDirectory() as temp_dir:
        os.mkdir(os.path.join(temp_dir, "test"))

        os.mkdir(os.path.join(temp_dir, "test", "subdir1"))
        os.mkdir(os.path.join(temp_dir, "test", "subdir2"))

        os.mkdir(os.path.join(temp_dir, "test", "subdir1", "point1"))
        os.mkdir(os.path.join(temp_dir, "test", "subdir2", "point2"))

        data_utils.combine_point_directories(os.path.join(temp_dir, "test"))

        assert os.path.exists(os.path.join(temp_dir, "test", "combined_folder"))
        assert os.path.exists(os.path.join(temp_dir, "test", "combined_folder", "subdir1_point1"))
        assert os.path.exists(os.path.join(temp_dir, "test", "combined_folder", "subdir2_point2"))


def test_stitch_images():
    fovs = ['fov' + str(i) for i in range(40)]
    chans = ['nuc1', 'nuc2', 'mem1', 'mem2']

    data_xr = test_utils.make_images_xarray(None, fovs, 10, 10, chans, dtype='int16')

    stitched_xr = data_utils.stitch_images(data_xr, 5)

    assert stitched_xr.shape == (1, 80, 50, 4)


def test_split_img_stack():
    with tempfile.TemporaryDirectory() as temp_dir:

        fovs = ['stack_sample']
        chans = [f'chan{i}' for i in range(10)]
        names = [f'{chan}.tiff' for chan in chans]

        stack_list = ["stack_sample.tiff"]
        stack_dir = os.path.join(temp_dir, fovs[0])
        os.mkdir(stack_dir)

        output_dir = os.path.join(temp_dir, "output_sample")
        os.mkdir(output_dir)

        filelocs, data_xr = test_utils.create_paired_xarray_fovs(stack_dir, fovs,
                                                                 chans, img_shape=(128, 128),
                                                                 mode='multitiff')

        # first test channel_first=False
        data_utils.split_img_stack(stack_dir, output_dir, stack_list, [0, 1], names[0:2],
                                   channels_first=False)

        # raise ValueError(f"{os.listdir(os.path.join(output_dir, 'stack_sample'))}")

        assert os.path.exists(os.path.join(output_dir, "stack_sample", "chan0.tiff"))
        assert os.path.exists(os.path.join(output_dir, "stack_sample", "chan1.tiff"))

        sample_chan_1 = io.imread(os.path.join(output_dir, "stack_sample", "chan0.tiff"))
        sample_chan_2 = io.imread(os.path.join(output_dir, "stack_sample", "chan1.tiff"))

        assert np.array_equal(sample_chan_1, data_xr[0, :, :, 0].values)
        assert np.array_equal(sample_chan_2, data_xr[0, :, :, 1].values)

        rmtree(os.path.join(output_dir, 'stack_sample'))

        # now overwrite old stack_sample.jpg file and test channel_first=True
        filelocs, data_xr = test_utils.create_paired_xarray_fovs(stack_dir, fovs,
                                                                 chans, img_shape=(128, 128),
                                                                 mode='reverse_multitiff')

        data_utils.split_img_stack(stack_dir, output_dir, stack_list, [0, 1], names[0:2],
                                   channels_first=True)

        assert os.path.exists(os.path.join(output_dir, "stack_sample", "chan0.tiff"))
        assert os.path.exists(os.path.join(output_dir, "stack_sample", "chan1.tiff"))

        sample_chan_1 = io.imread(os.path.join(output_dir, "stack_sample", "chan0.tiff"))
        sample_chan_2 = io.imread(os.path.join(output_dir, "stack_sample", "chan1.tiff"))

        assert np.array_equal(sample_chan_1, data_xr[0, :, :, 0].values)
        assert np.array_equal(sample_chan_2, data_xr[0, :, :, 1].values)
