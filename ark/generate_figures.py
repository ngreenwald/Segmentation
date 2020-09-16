





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