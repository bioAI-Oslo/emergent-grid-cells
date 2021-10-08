import torch
import numpy as np
import matplotlib.pyplot as plt
import copy

from ratsimulator import trajectory_generator
import torch.utils.data as tdata
from Brain import *


def generic_train_loop(model, trainloader, optimizer, criterion, nepochs):
    loss_history = []
    # loop over the dataset many times with progressbar from tqdm
    for epoch in tqdm.trange(nepochs):

        # generic torch training loop
        running_loss = 0.0
        for data in trainloader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss = loss.item()
        loss_history.append(running_loss / len(trainloader))

    return model, loss_history


class Dataset(torch.utils.data.Dataset):
    def __init__(self, brain, batch_size=64, nsteps=100, return_cartesian=False, *args, **kwargs):
        self.tg = trajectory_generator(*args, **kwargs)
        self.brain = brain
        self.batch_size = batch_size
        self.nsteps = nsteps
        self.return_cartesian = return_cartesian

    def __len__(self):
        """
        Length of dataset. But since generated, set
        to a large fixed value.
        """
        return self.nsteps * self.batch_size

    def __getitem__(self, index):
        """Note, 'index' is ignored. Assumes only getting single elements"""
        """
        # FOR DEBUGGING: print worker IDs
        worker_info = tdata.get_worker_info()
        worker_id = 0
        if worker_info:
                worker_id = worker_info.id
        
        print(f"{worker_id=}, {index=}")
        """
        pos, vel = next(self.tg)[:2]
        # OBS! data is not set to device here, since allocating gpu-memory in parallel is
        # non-trivial. Data should be put on correct device in training loop instead.
        pos = torch.tensor(pos, dtype=torch.float32)
        vel = torch.tensor(vel, dtype=torch.float32)
        pc_pos = self.brain.softmax_response(pos)
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
    bckp_tg = dataset.tg 
    board = environment.get_board(res)
    num_response_maps = (idxs.stop - idxs.start) // idxs.step
    response_maps = np.zeros((num_response_maps, *res))
    count_maps = np.zeros((num_response_maps, *res))
    dres = (environment.boxsize - environment.origo) / res

    # Calculate grid cell responses
    for n in range(num_samples := 1):
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
                tg = trajectory_generator(
                    environment=environment, seq_len=seq_len, angle0=angle0, p0=p0
                )
                dataset.tg = tg
                # sample data from generator
                inputs, labels = dataset[0]
                # reintegrate positions from initial (cartesian) position and velocities
                pos = np.cumsum(
                    np.concatenate([p0, inputs[0].detach().numpy()]), axis=0
                )
                # assign activity to space (from environment coordinates to pixel-coordinates) 
                pos_idxs = np.around((res-1) * pos / np.array(environment.boxsize)).astype(int)
                if model == 'labels':
                    model_cell_response = labels.numpy()[None] # add empty-batch dim
                else:
                    model_cell_response = model(inputs).detach().cpu().numpy()
                response_maps[
                    :, pos_idxs[:-1, 0], pos_idxs[:-1, 1]
                ] += model_cell_response[0, :, idxs].T
                count_maps[:, pos_idxs[:-1, 0], pos_idxs[:-1, 1]] += 1

    dataset.tg = bckp_tg
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
        ax[k // ncells, k % ncells].contourf(
            xx, yy, zz[k], cmap="jet"
        )
    
    return fig, ax

def multiimshow(zz):
    """plot multiple imshow plots on a grid"""
    ncells = int(np.sqrt(zz.shape[0]))
    fig, ax = plt.subplots(figsize=(10, 10), nrows=ncells, ncols=ncells, squeeze=False)

    # plot response maps using contourf
    for k in range(zz.shape[0]):
        ax[k // ncells, k % ncells].axis("off")
        # ax[int(k / ncells), k % ncells].set_aspect('equal')
        ax[k // ncells, k % ncells].imshow(zz[k], cmap='jet')
    
    return fig, ax 
