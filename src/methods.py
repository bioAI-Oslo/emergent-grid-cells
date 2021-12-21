import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import interpolate
import scipy
import pickle

from ratsimulator import Agent


class Dataset_old(torch.utils.data.Dataset):
    def __init__(
        self,
        environments,
        place_cell_ensembles,
        seq_len=20,
        dataset_size=int(1e6),
        **kwargs,
    ):
        # Dataset can use multiple environments/place cell ensembles. If only one of each,
        # encapsulate in lists to comply with class logic.
        self.environments = (
            environments if isinstance(environments, list) else [environments]
        )
        self.agents = [
            Agent(environment, **kwargs) for environment in self.environments
        ]
        self.place_cell_ensembles = (
            place_cell_ensembles
            if isinstance(place_cell_ensembles, list)
            else [place_cell_ensembles]
        )
        self.num_environments = len(self.place_cell_ensembles)

        self.seq_len = seq_len
        self.dataset_size = dataset_size

    def __len__(self):
        """
        Define number of samples in dataset. Since dataset is generated this number
        can be chosen arbitrarily.
        """
        return self.dataset_size

    def get_place_cells(self, index):
        """
        A data sample depends on a particular place cell basis - retrieve this
        basis, given the dataset index.
        """
        return self.place_cell_ensembles[index % self.num_environments]

    def __getitem__(self, index):
        """
        Get a training (input & label) sample from the dataset.

        Note! 'index' normally indexes the dataset, but not here since the
        dataset is generated. The index is however used, for determining
        which PlaceCells object basis to convert to.

        Note2! Other quantities (e.g. euclidean positions) can be
        access by for example - all as np.arrays!:
        dataset[0]
        dataset.agent.positions
        """
        # select agent/place cell basis (environment) uniformly across mini-batch
        # samples chosen depending on 'index'
        agent = self.agents[index % self.num_environments]
        place_cells = self.place_cell_ensembles[index % self.num_environments]

        agent.reset()
        for _ in range(self.seq_len):
            agent.step()

        # Cast positions (in pc-basis) and velocities to tf.tensors
        velocities = torch.tensor(agent.velocities[1:], dtype=torch.float32)
        positions = torch.tensor(agent.positions, dtype=torch.float32)
        pc_positions = place_cells.softmax_response(positions)
        init_pc_positions, labels = pc_positions[0], pc_positions[1:]
        return [[velocities, init_pc_positions], labels]


def compute_ratemaps(
    model,
    dataset,
    num_trajectories=1000,
    res=np.array([20, 20]),
    idxs=slice(0, 64, 1),
    **kwargs,
):
    activities, x, y = [], [], []
    for _ in range(num_trajectories):
        activities.append(model(dataset[0][0])[0, :, idxs].detach().cpu().numpy())
        x.append(dataset.agents[0].positions[1:, 0])
        y.append(dataset.agents[0].positions[1:, 1])
    activities = np.array(activities)
    return scipy.stats.binned_statistic_2d(
        np.ravel(x),
        np.ravel(y),
        activities.reshape(-1, activities.shape[-1]).T,
        bins=res,
        **kwargs,
    )


def multicontourf(xx, yy, zz, axs=None):
    """plot multiple contourf plots on a grid"""
    if axs is None:
        ncells = int(np.sqrt(zz.shape[0]))
        fig, axs = plt.subplots(
            figsize=(10, 10), nrows=ncells, ncols=ncells, squeeze=False
        )
    else:
        fig = None
        ncells = axs.shape[0]

    # plot response maps using contourf
    for k in range(zz.shape[0]):
        axs[k // ncells, k % ncells].axis("off")
        # ax[int(k / ncells), k % ncells].set_aspect('equal')
        axs[k // ncells, k % ncells].contourf(xx, yy, zz[k], cmap="jet")

    return fig, axs


def multiimshow(zz, axs=None):
    """plot multiple imshow plots on a grid"""
    if axs is None:
        ncells = int(np.sqrt(zz.shape[0]))
        fig, axs = plt.subplots(
            figsize=(10, 10), nrows=ncells, ncols=ncells, squeeze=False
        )
    else:
        fig = None
        ncells = axs.shape[0]

    # plot response maps using contourf
    for k in range(zz.shape[0]):
        axs[k // ncells, k % ncells].axis("off")
        # ax[int(k / ncells), k % ncells].set_aspect('equal')
        axs[k // ncells, k % ncells].imshow(zz[k], cmap="jet")

    return fig, axs


def interpolate_missing_pixels(
    image: np.ndarray, mask: np.ndarray, method: str = "cubic", fill_value: int = 0
):
    """
    Taken from:
    https://stackoverflow.com/questions/37662180/interpolate-missing-values-2d-python

    :param image: a 2D image
    :param mask: a 2D boolean image, True indicates missing values
    :param method: interpolation method, one of
        'nearest', 'linear', 'cubic'.
    :param fill_value: which value to use for filling up data outside the
        convex hull of known pixel values.
        Default is 0, Has no effect for 'nearest'.
    :return: the image with missing values interpolated
    """
    h, w = image.shape[:2]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))

    known_x = xx[~mask]
    known_y = yy[~mask]
    known_v = image[~mask]
    missing_x = xx[mask]
    missing_y = yy[mask]

    interp_values = interpolate.griddata(
        (known_x, known_y),
        known_v,
        (missing_x, missing_y),
        method=method,
        fill_value=fill_value,
    )

    interp_image = image.copy()
    interp_image[missing_y, missing_x] = interp_values

    return interp_image


def _find_inner_circle_dist(sac, topk=10):
    """
    Helper function for grid_score()
    Args:
        sac: autocorrellogram
        topk: number of peak transitions to look for (in any direction)
    Returns:
        robust_dist: robust (median) l2-dist to transition peaks from center
        idxs: the idxs of the peaks used to calculate the (robust) dist to transition peaks
    """
    center_x = sac.shape[0] / 2
    center_y = sac.shape[1] / 2
    # smooth sac
    smooth_sac = signal.correlate2d(
        sac, np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]), mode="same"
    )
    # differential filtering (could experiment with different filter widths for stability)
    dx_sac = signal.correlate2d(
        smooth_sac, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]), mode="same"
    )
    dy_sac = signal.correlate2d(
        smooth_sac, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]), mode="same"
    )
    # double differentiantial filtering
    dx2_sac = dx_sac = signal.correlate2d(
        dx_sac, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]), mode="same"
    )
    dy2_sac = signal.correlate2d(
        dy_sac, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]), mode="same"
    )

    # find peak transitions from center mass
    idxs = np.argpartition((dx2_sac + dy2_sac).flatten(), -topk)[-topk:]
    idxs = np.unravel_index(idxs, sac.shape)
    idxs = np.stack(idxs, axis=-1)  # vectorize idxs
    """
    # plot points in transition plot (double-differentiated sac)
    fig,ax = plt.subplots()
    ax.imshow(dx2_sac + dy2_sac)
    for idx in idxs:
        ax.add_patch(plt.Circle(idx, 0.2, color='r'))
    """
    # calculate distances
    center = np.array([center_x, center_y])
    l2squared_to_center = np.sum((idxs - center) ** 2, axis=-1)
    # choose distance
    robust_dist = np.median(l2squared_to_center)
    return robust_dist, idxs


def _get_annulus_mask(sac):
    """
    Helper function for grid_score()
    """
    # get center coords and create ogrid
    center_x = sac.shape[0] / 2
    center_y = sac.shape[1] / 2
    x, y = np.ogrid[0 : sac.shape[0], 0 : sac.shape[1]]
    # create annulus mask (outer circle fills the square autocorrellogram)
    outer_mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= min(
        center_x ** 2, center_y ** 2
    )
    inner_circle_dist, _ = _find_inner_circle_dist(sac)  # automatic inner circle mask
    inner_mask = (x - center_x) ** 2 + (y - center_y) ** 2 >= inner_circle_dist
    return inner_mask * outer_mask


def grid_score(rate_map):
    """
    Self-made grid-score function. Grid score is not standardized and thus,
    different grid_score functions give varying results.
    """
    # autocorrelate
    sac = signal.correlate2d(rate_map, rate_map, mode="full")
    annulus_mask = _get_annulus_mask(sac)
    masked_sac = sac[annulus_mask]

    # correlate with rotated sacs
    angles = np.arange(30, 180, 30)
    distributed_image_transform = lambda angle: np.corrcoef(
        scipy.ndimage.rotate(sac, angle=angle, reshape=False)[annulus_mask].flatten(),
        masked_sac.flatten(),
    )[0, 1]
    masked_rot_sacs = np.array([*map(distributed_image_transform, angles)])
    phase60 = masked_rot_sacs[1::2]
    phase30 = masked_rot_sacs[::2]
    return np.min(phase60) - np.max(phase30)
