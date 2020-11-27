import numpy as np


def positive_pixels_extraction(cell_coords, image_data, **kwargs):
    """Extract channel counts by summing over the number of non-zero pixels in the cell.

    Improves on default_extraction by not distinguishing between pixel expression values

    Args:
        cell_coords (numpy.ndarray): values representing pixels within one cell
        image_data (xarray.DataArray): array containing channel counts
        kwargs: arbitrary keyword arguments

    Returns:
        numpy.ndarray:
            Sums of counts for each channel
    """

    if 'threshold' not in kwargs.keys():
        kwargs['threshold'] = 0

    # index into image_data
    channel_values = image_data.values[tuple(cell_coords.T)]

    # create binary mask based on threshold
    channel_counts = np.sum(channel_values > kwargs['threshold'], axis=0)

    return channel_counts


def center_weighting_extraction(cell_coords, image_data, **kwargs):
    """Extract channel counts by summing over weighted expression values based on distance from
    center.

    Improves upon default extraction by including a level of certainty/uncertainty.

    Args:
        cell_coords (numpy.ndarray):
            values representing pixels within one cell
        image_data (xarray.DataArray):
            array containing channel counts
        **kwargs:
            arbitrary keyword arguments

    Returns:
        numpy.ndarray:
            Sums of counts for each channel
    """

    # check for centroid in kwargs
    if 'centroid' not in kwargs.keys():
        raise TypeError("center_weighting_extraction() missing keyowrd-only argument: 'centroid'")

    # compute the distance box-level from the center outward
    weights = np.linalg.norm(cell_coords - kwargs['centroid'], ord=np.inf, axis=1)

    # center the weights around the middle value
    weights = 1 - (weights / (np.max(weights) + 1))

    # retrieve the channel counts
    channel_values = image_data.values[tuple(cell_coords.T)]
    channel_counts = weights.dot(channel_values)

    return channel_counts


def default_extraction(cell_coords, image_data, **kwargs):
    """ Extract channel counts for an individual cell via basic summation for each channel

    Args:
        cell_coords (numpy.ndarray):
            values representing pixels within one cell
        image_data (xarray.DataArray):
            array containing channel counts
        kwargs:
            arbitrary keyword arguments

    Returns:
        numpy.ndarray:
            Sum of counts for each channel
    """

    # index into image_data to get the channel values we're interested in
    channel_values = image_data.values[tuple(cell_coords.T)]

    # collapse along channels dimension to get counts per channel
    channel_counts = np.sum(channel_values, axis=0)

    return channel_counts


extraction_function = {
    'positive_pixel': positive_pixels_extraction,
    'center_weighting': center_weighting_extraction,
    'default': default_extraction,
}
