import torch

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