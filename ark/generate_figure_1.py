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
plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/Segmentation_Project/analyses/20200830_figure_1/'

fig, ax = plt.subplots(figsize=(3, 3))
figures.barchart_helper(ax=ax, values=[total_time / 3600, internal_hours],
                        labels=['Crowdsource', 'QC'],
                        title='Total hours',
                        colors='blue')
fig.tight_layout()
fig.savefig(os.path.join(plot_dir, 'hours.tiff'))

# counts by modality
tissue_counts = np.load(plot_dir + 'tissue_counts.npz',
                        allow_pickle=True)['stats'].item()

tissue_counts = pd.DataFrame(tissue_counts)
tissue_vals = tissue_counts.iloc[0, :].values
tissue_names = tissue_counts.columns.values

sort_idx = np.argsort(-tissue_vals)
fig, ax = plt.subplots(figsize=(5, 5))
figures.barchart_helper(ax=ax, values=tissue_vals[sort_idx],
                        labels=tissue_names[sort_idx],
                        title='Cells per tissue type', colors='blue')
fig.tight_layout()
fig.savefig(os.path.join(plot_dir, 'annotations_per_tissue.tiff'))

# counts by modality
platform_counts = np.load(plot_dir + 'platform_counts.npz',
                        allow_pickle=True)['stats'].item()


platform_counts = pd.DataFrame(platform_counts)
platform_counts = pd.DataFrame(platform_counts)
platform_vals = platform_counts.iloc[0, :].values
platform_names = platform_counts.columns.values

sort_idx = np.argsort(-platform_vals)
fig, ax = plt.subplots(figsize=(5, 5))
figures.barchart_helper(ax=ax, values=platform_vals[sort_idx],
                        labels=platform_names[sort_idx],
                        title='Cells per platform type', colors='blue')
fig.tight_layout()
fig.savefig(os.path.join(plot_dir, 'annotations_per_platform.tiff'))
