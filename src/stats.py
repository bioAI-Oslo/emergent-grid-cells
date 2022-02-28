import numpy as np
import spatial_maps as sm
import scipy


def mad(x):
    """
    mean absolute deviation
    """
    return np.median(np.abs(x - np.median(x)))


def find_peaks_idxs(ratemap):
    """
    Overwrites find_peaks from spatial-maps
    such that the output is indecies of the peaks of the input
    """
    peaks = sm.find_peaks(img)
    return tuple(peaks.T)


def grid_spacing(ratemap, boxsize=2.2, p=0.1, verbose=False, **kwargs):
    """
    Calculate the median distance to all 6 nearest peaks from center peak.
    """
    autocorr = scipy.signal.correlate(ratemap, ratemap, **kwargs)
    peaks = sm.find_peaks(autocorr)
    idx = num_closest_isodistance_points(ratemap)
    hexagonal_dist = np.linalg.norm(peaks[1:idx] - peaks[0], axis=-1)
    hexagonal_dist = hexagonal_dist / ratemap.shape[0]
    median_dist = np.median(hexagonal_dist)
    sigma = mad(hexagonal_dist)
    if sigma > p and verbose:
        print(f"{sigma=}>{p=}. Grid spacing might NOT be ROBUST")

    return median_dist * boxsize, sigma


def phase_shift(ratemap1, ratemap2, mode="mode", **kwargs):
    """
    Calculate the shift between two ratemaps.
    Mode is either 'mode' or 'mean'

    Method is more robust with smoothed ratemaps
    """
    crosscorr = scipy.signal.correlate(ratemap1, ratemap2, **kwargs)
    peaks = find_peaks_idxs(crosscorr)
    peak_shift = crosscorr[peaks][0]
    return peak_shift


def optimal_orientation(ratemap1, ratemap2, **kwargs):
    """
    Calculate the optimal orientation between two ratemaps
    """
    autocorr = scipy.signal.correlate(ratemap1, ratemap2, **kwargs)
    peaks = find_peaks(autocorr)
    idx = num_closest_isodistance_points(ratemap)
    hexagonal_dist = peaks[1:idx] - peaks[0]  # shift coordinate system
    return np.sort(np.arctan2(*peaks.T) % np.pi)[::-1]


def num_closest_isodistance_points(ratemap, **kwargs):
    """
    Determine the number of grid points that are closest to the center
    """
    autocorr = scipy.signal.correlate(ratemap, ratemap, **kwargs)
    peaks = sm.find_peaks(autocorr)
    dists = np.linalg.norm(
        peaks[1:] - peaks[0], axis=1
    )  # shift coordinate system to the center
    # dists = scipy.ndimage.correlate(dists, np.array([1,2,1]))
    dddx = scipy.signal.correlate(dists, np.array([-1, 1]), mode="valid")
    outliers = dddx > np.mean(dddx) + 2 * np.std(dddx)
    return np.argmax(outliers)  # get the index of the first "True" occurence
