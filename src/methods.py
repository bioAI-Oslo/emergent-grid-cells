import torch
import numpy as np
import matplotlib.pyplot as plt

from ratsimulator import trajectory_generator


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        environment,
        place_cells,
        batch_size=64,
        seq_len=20,
        nsteps=100,
        return_cartesian=False,
        **kwargs,
    ):
        self.environment = environment
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
