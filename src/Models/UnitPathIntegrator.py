import torch


class UnitPathIntegrator(torch.nn.Module):
    """
    Specification of the SorscherRNN model -
    stripping the model down to the bone in an effort to probe
    which parts of the model is necessary to do path integration.
    """

    def __init__(self, Ng=4096, Np=512, weight_decay=1e-4, nonlinearity="relu", **kwargs):
        super(UnitPathIntegrator, self).__init__(**kwargs)
        # define network architecture
        self.velocity_encoder = torch.nn.Linear(2, Ng, bias=False)
        self.init_position_encoder = torch.nn.Linear(Np, Ng, bias=False)
        self.recurrence = torch.nn.Linear(Ng, Ng, bias=False)
        self.decoder = torch.nn.Linear(Ng, Np, bias=False)
        nonlinearity = torch.nn.ReLU()

    def g(self, inputs):
        """
        One recurrence step from p0 to p0 + v
        """
        v, p0 = inputs
        v = self.velocity_encoder(v)
        p0 = self.init_position_encoder(p0)
        return torch.nn.functional.relu(self.recurrence(p0) + v)

    def call(self, inputs, softmax=False):
        place_preds = self.decoder(self.g(inputs))
        return torch.nn.functional.softmax(place_preds) if softmax else place_preds

    def loss_fn(self, predictions, labels):
        """
        Args:
            inputs ((B, S, 2), (B, Np)): velocity and position in pc-basis
            labels (B, Np): ground truth pc population activations
        """
        # Actual cross entropy between two distributions p(x) and q(x),
        # rather than classic CE implementations assuming one-hot p(x).
        cross_entropy = torch.sum(- labels * torch.log(predictions),axis=-1)
        return torch.mean(cross_entropy)

    def train(self, trainloader, optimizer, nepochs, nsteps):
        """
        Modified generic train loop for sorscher rnn. Data is arbitrary
        large since it is generated and not a fixed set.
        """
        loss_history = []
        pbar = tqdm.tqdm(range(nepochs))
        for epoch in pbar:
            # generic torch training loop
            running_loss = 0.0
            for _ in range(nsteps):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = next(trainloader)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                predictions = self(inputs, softmax=True)
                loss = self.loss_fn(predictions, labels)
                loss.backward()
                optimizer.step()

                running_loss = loss.item()

            loss_history.append(running_loss / nsteps)
            pbar.set_description(f"Epoch={epoch}, loss={loss_history[-1]}")


