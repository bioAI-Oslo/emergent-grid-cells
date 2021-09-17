import torch
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
    def __init__(self, brain, batch_size=64, nsteps = 100, *args, **kwargs):
        self.tg = trajectory_generator(*args, **kwargs)
        self.brain = brain
        self.batch_size = batch_size
        self.nsteps = nsteps

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
        pos = self.brain.softmax_response(pos)
        init_pos, labels = pos[0], pos[1:]
        vel = vel[1:] # first velocity is a dummy velocity
        # OBS! data is not set to device here, since allocating gpu-memory in parallel is
        # non-trivial. Data should be put on correct device in training loop instead.
        init_pos = torch.tensor(init_pos, dtype=torch.float32)
        vel = torch.tensor(vel, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)

        return [[vel, init_pos], labels]


