import torch
import numpy as np
import matplotlib.pyplot as plt

from ratsimulator import trajectory_generator


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        place_cells,
        batch_size=64,
        seq_len=20,
        nsteps=100,
        return_cartesian=False,
        **kwargs,
    ):
        self.place_cells = place_cells
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.nsteps = nsteps
        self.return_cartesian = return_cartesian

        self.tg = trajectory_generator(
            environment=environment, seq_len=seq_len, **kwargs
        )

    def set_start_pose(self, angle0=None, p0=None, **kwargs):
        """
        Set a fixed start POSE for the generator - used for rate map calculations
        s.t agent is placed deterministically within every area of an arena.
        """
        self.tg = trajectory_generator(
            environment=self.environment,
            seq_len=self.seq_len,
            angle0=angle0,
            p0=p0,
            **kwargs,
        )

    def __len__(self):
        """
        Length of dataset. But since generated, set
        to a large fixed value.
        """
        return self.nsteps * self.batch_size

    def __getitem__(self, index):
        """Note, 'index' is ignored. Assumes only getting single elements"""
        pos, vel = next(self.tg)[:2]
        pos = torch.tensor(pos, dtype=torch.float32)
        vel = torch.tensor(vel, dtype=torch.float32)
        pc_pos = self.place_cells.softmax_response(pos)
        init_pos, labels = pc_pos[0], pc_pos[1:]
        vel = vel[1:]  # first velocity is a dummy velocity

        if self.return_cartesian:
            # for prediction phase (NOT training)
            return [[vel, init_pos], labels, pos]
        return [[vel, init_pos], labels]


def rate_map(
    model,
    environment,
    dataset,
    seq_len,
    res=np.array([20, 20]),
    idxs=slice(0, 64, 1),
    num_samples=1,
    *args,
    **kwargs,
):
    dataset.return_cartesian = True
    board = environment.get_board(res)
    num_response_maps = (idxs.stop - idxs.start) // idxs.step
    response_maps = np.zeros((num_response_maps, *res))
    count_maps = np.zeros((num_response_maps, *res))
    dres = (environment.boxsize - environment.origo) / res

    # Calculate grid cell responses
    for n in range(num_samples):
        # sample 'same' positions multiple times at different head directions
        angle0 = None  # None implies random sampled head direction
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                p0 = np.random.uniform(
                    low=np.maximum(board[i, j] - dres / 2, environment.origo),
                    high=np.minimum(board[i, j] + dres / 2, environment.boxsize),
                )
                p0 = p0[None]  # p0 needs shape=(1,2)
                # reinitialise pytorch dataset generator
                dataset.set_start_pose(angle0=angle0, p0=p0, **kwargs)
                # sample data from generator
                inputs, labels, pos = dataset[0]
                pos = pos.numpy()
                # assign activity to space (from environment coordinates to pixel-coordinates)
                pos_idxs = np.around(
                    (res - 1) * pos / np.array(environment.boxsize)
                ).astype(int)
                if model == "labels":
                    model_cell_response = labels.numpy()[None]  # add empty-batch dim
                else:
                    model_cell_response = model(inputs).detach().cpu().numpy()
                response_maps[
                    :, pos_idxs[:-1, 0], pos_idxs[:-1, 1]
                ] += model_cell_response[0, :, idxs].T
                count_maps[:, pos_idxs[:-1, 0], pos_idxs[:-1, 1]] += 1

    # reset dataset-object
    dataset.set_start_pose(angle0=None, p0=None, **kwargs)
    dataset.return_cartesian = False

    rate_maps = response_maps / count_maps
    return board, rate_maps, response_maps, count_maps


def multicontourf(xx, yy, zz):
    """plot multiple contourf plots on a grid"""
    ncells = int(np.sqrt(zz.shape[0]))
    fig, ax = plt.subplots(figsize=(10, 10), nrows=ncells, ncols=ncells, squeeze=False)

    # plot response maps using contourf
    for k in range(zz.shape[0]):
        ax[k // ncells, k % ncells].axis("off")
        # ax[int(k / ncells), k % ncells].set_aspect('equal')
        ax[k // ncells, k % ncells].contourf(xx, yy, zz[k], cmap="jet")

    return fig, ax


def multiimshow(zz):
    """plot multiple imshow plots on a grid"""
    ncells = int(np.sqrt(zz.shape[0]))
    fig, ax = plt.subplots(figsize=(10, 10), nrows=ncells, ncols=ncells, squeeze=False)

    # plot response maps using contourf
    for k in range(zz.shape[0]):
        ax[k // ncells, k % ncells].axis("off")
        # ax[int(k / ncells), k % ncells].set_aspect('equal')
        ax[k // ncells, k % ncells].imshow(zz[k], cmap="jet")

    return fig, ax


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
    inner_circle_dist, _ = find_inner_circle_dist(sac)  # automatic inner circle mask
    inner_mask = (x - center_x) ** 2 + (y - center_y) ** 2 >= inner_circle_dist
    return inner_mask * outer_mask


def grid_score(rate_map):
    """
    Self-made grid-score function. Grid score is not standardized and thus,
    different grid_score functions give varying results.
    """
    # autocorrelate
    sac = signal.correlate2d(rate_map, rate_map, mode="full")
    annulus_mask = get_annulus_mask(sac)
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
