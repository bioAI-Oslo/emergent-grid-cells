import numpy as np
import spatial_maps as sm
import scipy
from scipy.special import i0


def circular_kernel(data, kappa):
    """Generate a von Mises kernel for kernel density estimation
    Based on https://www.stata.com/meeting/mexico14/abstracts/materials/mex14_salgado.pdf
    Call this function to obtain a kernel, which can then subsequently be used on new data

    Args:
        data (np.array): circular data to be used for the kernel; shape (N,).
        kappa (float): shape parameter, sets smoothness of KDE
    """

    def kernel(x):
        """Create the actual kernel to be returned

        Args:
            x (np.array): circular data to be evaluated vs kernel; shape (M,)
        """
        von_mises = (
            1
            / (2 * np.pi * i0(kappa))
            * np.exp(kappa * np.cos(x[:, None] - data[None, :]))
        )
        return np.mean(von_mises, axis=-1) / np.trapz(np.mean(von_mises, axis=-1), x=x)  # sum over data dimension

    return kernel


def mad(x):
    """
    mean absolute deviation
    """
    return np.median(np.abs(x - np.median(x)))


def find_peaks_idxs(img):
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
    if idx is None:
        return np.nan, np.nan
    hexagonal_dist = np.linalg.norm(peaks[1 : idx + 1] - peaks[0], axis=-1)
    hexagonal_dist = hexagonal_dist / ratemap.shape[0]
    median_dist = np.median(hexagonal_dist)
    sigma = mad(hexagonal_dist)
    if sigma > p and verbose:
        print(f"{sigma=}>{p=}. Grid spacing might NOT be ROBUST")

    return median_dist * boxsize, sigma

def phase_shift2(img):
    """
    Calculate the shift between two ratemaps.
    Mode is either 'mode' or 'mean'

    Method is more robust with smoothed ratemaps
    """
    peaks = sm.find_peaks(img)
    closest_peak = peaks[0]
    center = (np.array(img.shape) - 1) / 2
    return closest_peak - center if np.std(img) != 0 else np.array([np.nan, np.nan])
 

def phase_shift(ratemap1, ratemap2, norm=True, **kwargs):
    """
    Calculate the shift between two ratemaps.
    Mode is either 'mode' or 'mean'

    Method is more robust with smoothed ratemaps
    """
    crosscorr = scipy.signal.correlate(ratemap1, ratemap2, **kwargs)
    peaks = sm.find_peaks(crosscorr)
    closest_peak = peaks[0]
    center = (np.array(crosscorr.shape) - 1) / 2
    if not norm:
        return closest_peak - center if np.std(crosscorr) != 0 else np.array([np.nan, np.nan])
    dist = np.linalg.norm(closest_peak - center)
    return dist if np.std(crosscorr) != 0 else np.nan
        
    

def grid_orientation(ratemap, **kwargs):
    """
    calculate orientation of grids in ratemap

    also calculate difference between angles (dangles)
    """
    autocorr = scipy.signal.correlate(ratemap, ratemap, **kwargs)
    peaks = sm.find_peaks(autocorr)
    idx = num_closest_isodistance_points(ratemap)
    if idx is None:
        return np.nan, np.nan
    centered_peaks = peaks[1 : idx + 1] - peaks[0]  # shift coordinate system
    angles = np.sort(np.arctan2(*centered_peaks.T) % (2 * np.pi))  # [::-1]
    dangles = np.diff(angles)
    return angles[0], dangles


def num_closest_isodistance_points(ratemap, **kwargs):
    """
    Determine the number of grid points that are closest to the center
    """
    autocorr = scipy.signal.correlate(ratemap, ratemap, **kwargs)
    peaks = sm.find_peaks(autocorr)
    if len(peaks) < 2:
        # need multiple peaks
        return None
    dists = np.linalg.norm(
        peaks[1:] - peaks[0], axis=1
    )  # shift coordinate system to the center
    # dists = scipy.ndimage.correlate(dists, np.array([1,2,1]))
    dddx = scipy.signal.correlate(dists, np.array([-1, 1]), mode="valid")
    outliers = dddx > np.mean(dddx) + 2 * np.std(dddx)
    return np.argmax(outliers) + 1  # get the index of the first "True" occurence


