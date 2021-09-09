import torch
from ratsimulator import batch_trajectory_generator
import torch.utils.data as tdata

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
    def __init__(self, num_workers = 1, *args, **kwargs):
        btgs = [batch_trajectory_generator(*args, **kwargs) for _ in range(num_workers)]
        
    def __len__(self):
        """
        Length of dataset. But since generated, set
        to a large fixed value.
        """
        return 1000 # 

    def __getitem__(self, index):
        worker_info = tdata.get_worker_info()
        worker_id = 0
        if worker_info:
                worker_id = worker_info.id
        
        pos, vel = next(btgs[worker_id])
        

        # Insert code similar to "Data generator" in notebook
        # however, do it so that we do not need to load with seq_len=2, but rather seq_len=1...
        # this will proly save some loading time!! 

        return None


