import subprocess
import tempfile
import os
from shutil import rmtree

from testbook import testbook

from ark.utils import test_utils
from ark.utils import misc_utils

import numpy as np
import skimage.io as io


SEGMENT_IMAGE_DATA_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                       '..', '..', 'templates', 'Segment_Image_Data.ipynb')


def _exec_notebook(nb_filename):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        '..', '..', 'templates', nb_filename)
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as fout:
        args = ["jupyter", "nbconvert", "--to", "notebook", "--execute",
                "--ExecutePreprocessor.timeout=1000",
                "--output", fout.name, path]
        subprocess.check_call(args)


# test runs with default inputs
def test_segment_image_data(mocker):
    _exec_notebook('Segment_Image_Data.ipynb')


def test_example_spatial_analysis():
    _exec_notebook('example_spatial_analysis_script.ipynb')


def test_example_neighborhood_analysis():
    _exec_notebook('example_neighborhood_analysis_script.ipynb')


# testing specific inputs for Segment_Image_Data
def segment_notebook_setup(tb, deepcell_tiff_dir="test_tiff", deepcell_input_dir="test_input",
                           deepcell_output_dir="test_output",
                           single_cell_dir="test_single_cell",
                           is_mibitiff=False, mibitiff_suffix="-MassCorrected-Filtered",
                           num_fovs=3, num_chans=3, dtype=np.uint16):
    # import modules and define file paths
    tb.execute_cell('import')

    # define custom mibitiff paths
    define_mibitiff_paths = """
        base_dir = "../data/example_dataset"
        input_dir = os.path.join(base_dir, "input_data")
        tiff_dir = "os.path.join(input_dir, %s)"
        deepcell_input_dir = os.path.join(input_dir, "%s/")
        deepcell_output_dir = os.path.join(base_dir, "%s/")
        single_cell_dir = os.path.join(base_dir, "%s/")
    """ % (deepcell_tiff_dir, deepcell_input_dir, deepcell_output_dir, single_cell_dir)
    tb.inject(define_mibitiff_paths, after='file_path')

    # create the tif files, don't do this in notebook it's too tedious to format this
    # also, because this is technically an input that would be created beforehand
    tiff_path = os.path.join('..', 'data', 'example_dataset', 'input_data', deepcell_tiff_dir)

    if is_mibitiff:
        fovs, chans = test_utils.gen_fov_chan_names(num_fovs=num_fovs,
                                                    num_chans=num_chans,
                                                    use_delimiter=True)
        fovs = [f + mibitiff_suffix for f in fovs]

        filelocs, data_xr = test_utils.create_paired_xarray_fovs(
            tiff_path, fovs, chans, img_shape=(1024, 1024), mode='mibitiff',
            delimiter='_', fills=False, dtype=dtype
        )
    else:
        fovs, chans = test_utils.gen_fov_chan_names(num_fovs=num_fovs,
                                                    num_chans=num_chans,
                                                    return_imgs=False)

        filelocs, data_xr = test_utils.create_paired_xarray_fovs(
            tiff_path, fovs, chans, img_shape=(1024, 1024), delimiter='_', fills=False,
            sub_dir="TIFs", dtype=dtype)

    # create the directories as listed by define_mibitiff_paths
    tb.execute_cell('create_dirs')

    # validate the paths, and in Jupyter, this should always pass
    # NOTE: any specific testing of validate_paths should be done in io_utils_test.py
    tb.execute_cell('validate_path')

    # will set MIBItiff and MIBItiff_suffix
    # if is_mibitiff is True, then we need to correct MIBITiff to True
    tb.execute_cell('mibitiff_set')
    if is_mibitiff:
        tb.inject("MIBItiff = True", after='mibitiff_set')


def fov_channel_input_set(tb, fovs_to_load=None, nucs_list=None, mems_list=None):
    # now load the fovs in the notebook
    if fovs_to_load is not None:
        tb.inject("fovs = %s" % str(fovs_to_load))
    else:
        tb.execute_cell('load_fovs')

    # we need to set the nucs_list and the mems_list accordingly
    if nucs_list is None:
        nucs_list_str = "None"
    else:
        nucs_list_str = "%s" % str(nucs_list)

    if mems_list is None:
        mems_list_str = "None"
    else:
        mems_list_str = "%s" % str(mems_list)

    nuc_mem_set = """
        nucs = %s\n
        mems = %s
    """ % (nucs_list_str, mems_list_str)
    tb.inject(nuc_mem_set, after='nuc_mem_set')

    # set the channels accordingly
    tb.execute_cell('set_channels')

    # load data accordingly
    tb.execute_cell('load_data_xr')

    # generate the deepcell input files
    # NOTE: any specific testing of generate_deepcell_input should be done in data_utils_test
    tb.execute_cell('gen_input')


def save_seg_labels(tb, delimiter='_feature_0', xr_dim_name='compartments',
                    xr_channel_names=None, force_ints=True):
    delimiter_str = delimiter if delimiter is not None else "None"
    xr_channel_names_str = str(xr_channel_names) if xr_channel_names is not None else "None"

    # load the segmentation label with the proper command
    # NOTE: any specific testing of load_imgs_from_dir should be done in load_utils_test.py
    load_seg_cmd = """
        segmentation_labels = load_utils.load_imgs_from_dir(
            data_dir=deepcell_output_dir,
            delimiter="%s",
            xr_dim_name="%s",
            xr_channel_names=%s,
            force_ints=%s
        )
    """ % (delimiter_str,
           xr_dim_name,
           xr_channel_names,
           str(force_ints))
    tb.inject(load_seg_cmd, after='load_seg_labels')

    tb.execute_cell('save_seg_labels')

    # now overlay data_xr
    tb.execute_cell('load_summed')
    tb.execute_cell('overlay_mask')


def create_exp_mat(tb, is_mibitiff=False, batch_size=5):
    # NOTE: segmentation labels will already have been created
    exp_mat_gen = """
        cell_table_size_normalized, cell_table_arcsinh_transformed = \
            marker_quantification.generate_cell_table(segmentation_labels=segmentation_labels,
                                                      tiff_dir=tiff_dir,
                                                      img_sub_folder="TIFs",
                                                      is_mibitiff=%s,
                                                      fovs=fovs,
                                                      batch_size=%s)
    """ % (is_mibitiff, str(batch_size))
    tb.inject(exp_mat_gen)

    # save expression matrix
    tb.execute_cell('save_exp_mat')


def remove_dirs(tb):
    remove_dirs = """
        from shutil import rmtree
        rmtree(tiff_dir)
        rmtree(deepcell_input_dir)
        rmtree(deepcell_output_dir)
        rmtree(single_cell_dir)
    """
    tb.inject(remove_dirs)


# test mibitiff, 6000 seconds = default timeout on Travis
@testbook(SEGMENT_IMAGE_DATA_PATH, timeout=6000)
def test_segment_image_data_mibitiff(tb):
    # create input files, set separate names for mibitiffs to avoid confusion
    segment_notebook_setup(tb, deepcell_tiff_dir="test_mibitiff_imgs",
                           deepcell_input_dir="test_mibitiff_input",
                           deepcell_output_dir="test_mibitiff_output",
                           single_cell_dir="test_mibitiff_single_cell",
                           is_mibitiff=True)

    # default fov setting, standard nucs/mems setting
    fov_channel_input_set(tb, nucs_list=['chan0'], mems_list=['chan1', 'chan2'])

    # default fov setting, nucs set to None
    fov_channel_input_set(tb, nucs_list=None, mems_list=['chan1', 'chan2'])

    # default fov setting, mems set to None
    fov_channel_input_set(tb, nucs_list=['chan0', 'chan1'], mems_list=None)

    # hard coded fov setting, standard nucs/mems setting, this is what we'll be testing on
    # TODO: this will fail if fovs_to_load is set without file extensions
    fov_channel_input_set(tb,
                          fovs_to_load=['fov0_otherinfo-MassCorrected-Filtered.tiff',
                                        'fov1-MassCorrected-Filtered.tiff'],
                          nucs_list=['chan0'],
                          mems_list=['chan1', 'chan2'])

    # generate the deepcell output files from the server
    tb.execute_cell('create_output')

    # run the segmentation labels saving and summed channel overlay processes
    save_seg_labels(tb, xr_channel_names=['whole_cell'])

    # create the expression matrix
    create_exp_mat(tb, is_mibitiff=True)

    # clean up the directories
    remove_dirs(tb)


# test folder loading
# also handles case when user doesn't specify all channels across nucs and mems
@testbook(SEGMENT_IMAGE_DATA_PATH, timeout=6000)
def test_segment_image_data_folder(tb):
    # create input files
    segment_notebook_setup(tb)

    # default fov setting, standard nucs/mems setting
    fov_channel_input_set(tb, nucs_list=['chan0'], mems_list=['chan1', 'chan2'])

    # default fov setting, nucs set to None
    fov_channel_input_set(tb, nucs_list=None, mems_list=['chan1', 'chan2'])

    # default fov setting, mems set to None
    fov_channel_input_set(tb, nucs_list=['chan0', 'chan1'], mems_list=None)

    # hard coded fov setting, standard nucs/mems setting, this is what we'll be testing on
    fov_channel_input_set(tb,
                          fovs_to_load=['fov0', 'fov1'],
                          nucs_list=['chan0'],
                          mems_list=['chan1', 'chan2'])

    # generate the deepcell output files from the server
    tb.execute_cell('create_output')

    # run the segmentation labels saving and summed channel overlay processes
    save_seg_labels(tb, xr_channel_names=['whole_cell'])

    # create the expression matrix
    create_exp_mat(tb)

    # clean up the directories
    remove_dirs(tb)
