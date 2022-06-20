from collections.abc import Callable

import numpy as np
import spatial_maps as sm
import scipy
from scipy.special import i0
import os
import pickle
from astropy.convolution import Gaussian2DKernel, convolve

from Experiment import Experiment
from methods import filenames


class IllegalArgumentError(ValueError):
    pass


def apply_to_selection(
    fn: Callable[[np.array], tuple], ratemaps: np.array, masks: np.array
) -> list:
    """
    Apply a function fn(ratemaps: np.array) -> (float, ...) to a selection of ratemaps
    specified by masks
    
    Args:
        ratemaps (np.array):    Ratemaps of the recurrent neural layer
                                    shape = (Num_Environments, Num_Reccurent_Neurons, Width_Field, Height_Field)
        masks (np.array):       Mask array specifying which grid cells are shall be selected for evaluation
                                Two modes of operation:
                                    (1) -- shape = (Num_selected_cells,) - same mask accross environments
                                    (2) -- shape = (Num_Environments, Num_selected_cells) - specified mask for each environment separately
                                    Anything incompatible will throw IllegalArgumentError
                                    
    Val:
        fn_value_list (list(np.array)):   Values of the function
                                        len(fn_values) = Num_Environments
    """

    # check for mode of operation
    # either one mask selects for grid cells across environments
    # or each environment has its designated selection mask
    num_of_envs = ratemaps.shape[0]
    if len(masks.shape) != 1:
        # one mask for each environment individually
        if num_of_envs != masks.shape[0]:
            raise IllegalArgumentError(
                "masks must either be one dimensional or have same length as number of environments"
            )
            return None

    if len(masks.shape) == 1:
        # one mask across environment
        # copy to the other environments
        masks = np.repeat(masks[np.newaxis, :], num_of_envs, axis=0)
    
    fn_value_list = []
    for env_i, rmaps_env in enumerate(ratemaps):
        selected_ratemaps_env = rmaps_env[masks[env_i]]
        num_of_selected_cells = masks[env_i].sum()
        fn_values_env = np.zeros((num_of_selected_cells, ))
        for rmap_i, rmap in enumerate(selected_ratemaps_env):
            fn_values_env[rmap_i], _ = fn(rmap)

        # remove NaN-values from array before adding to the list
        idx_not_nan = ~np.isnan(fn_values_env)
        fn_values_env = fn_values_env[idx_not_nan]

        fn_value_list.append(fn_values_env)

    return fn_value_list


def get_smooth_ratemaps(experiment: Experiment, sigma: float = 1.0) -> np.array:
    """
    Retrieve rat maps associated with the experiment

    Args:
        experiment (Experiment):    Experiment object containing information about data
        sigma (float):              Standard Deviation ("width") of 2D Gaussian Kernel the raw 
                                    ratemaps are smoothed with
                                        default: 1.0

    Val:
        ratemaps (np.array):        Ratemap array
                                        shape: (Num of Envs, Num of Neurons, X, Y)
    """
    kernel = Gaussian2DKernel(x_stddev=sigma)
    smooth_ratemaps = []

    for env_i, env in enumerate(experiment.environments):
        rmap_env_path = experiment.paths["ratemaps"] / f"env_{env_i}"
        sorted_filenames = filenames(rmap_env_path)

        with open(rmap_env_path / sorted_filenames[-1], "rb") as f:
            raw_rmap = pickle.load(f)
            smooth_rmap = convolve(raw_rmap, kernel.array[None])
            smooth_ratemaps.append(smooth_rmap)

    return np.array(smooth_ratemaps)


def grid_score_masks(
    experiment: Experiment, percentile: float = 0.4, mode="intersection"
) -> np.array:
    """
    Generate masks for relevant neurons based on their grid score in all environments

    Args:
        experiment (Experiment):    Experiment object containing information about data
        percentile (float):         percentile threshold for discrimination
                                        default: 0.4 =~ top 30 %
        mode (str):                 Key word specifying mode of operation
                                    default:    'intersection' - intersection set: select neurons that
                                                                 suffice percentile threshold in all 
                                                                 environments
                                                'union'        - union set: select neurons that suffice
                                                                 percentile threshold in any environment
                                                'separate'     - get masks for each environment separately
    
    Val:
        mask (np.array):            Mask for the given experiment
                                        shape: (Nx, Ny)
    """

    def load_gc_scores(experiment: Experiment) -> np.array:
        """
        Load the grid cell scores associated with the experiment 
        for different environments and return them in one single 
        array object

        Args:
            experiment (Experiment):    Experiment object containing the data

        Val:
            gc_scores (np.array):       Grid cell scores for all environments
                                            shape: (Number of Environments, Nx, Ny)
        """

        gc_scores = []
        score_filenames = os.listdir(experiment.paths["grid_scores"])
        for fname in score_filenames:
            if not "novel" in fname:
                with open(experiment.paths["grid_scores"] / f"{fname}", "rb") as f:
                    gc_scores.append(pickle.load(f))

        return np.array(gc_scores)

    grid_cell_scores = load_gc_scores(experiment)

    # Shift grid cell scores to > 0
    shifted_gc_scores = grid_cell_scores - np.amin(grid_cell_scores, axis=-1)[:, None]

    # determine neurons that belong to the selection
    sorted_gc_scores = np.sort(shifted_gc_scores, axis=-1)
    cummulative_distribution = np.cumsum(sorted_gc_scores, axis=-1) / np.sum(
        shifted_gc_scores, axis=-1, keepdims=True
    )
    cum_mask = cummulative_distribution > percentile

    no_envs = len(experiment.environments)

    thresholds = [
        np.amin(sorted_gc_scores[env_i, cum_mask[env_i]]) for env_i in range(no_envs)
    ]
    thresholds += np.amin(grid_cell_scores, axis=-1)  # shift back

    masks = np.array(  # select for scores higher than thresholds per environment
        [grid_cell_scores[env_i] > thresholds[env_i] for env_i in range(no_envs)]
    )
    if mode == "separate":
        return masks

    if mode == "intersection":
        # check if apparent in all envs
        return np.prod(masks, axis=0).astype("bool")

    if mode == "union":
        # ehck if apparent in any env
        return np.sum(masks, axis=0).astype("bool")

    # if key word unknown return None
    return None


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
        return np.mean(von_mises, axis=-1) / np.trapz(
            np.mean(von_mises, axis=-1), x=x
        )  # sum over data dimension

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
        return (
            closest_peak - center
            if np.std(crosscorr) != 0
            else np.array([np.nan, np.nan])
        )
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
