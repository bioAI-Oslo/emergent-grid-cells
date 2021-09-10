import torch
import tqdm

class SorscherRNN(torch.nn.Module):
    """
    Model based on:
    https://github.com/ganguli-lab/grid-pattern-formation/blob/master/model.py
    """

    def __init__(
        self,
        Ng=4096,
        Np=512,
        weight_decay=1e-4,
        nonlinearity="relu",
        **kwargs
    ):
        super(SorscherRNN, self).__init__(**kwargs)
        self.Ng, self.Np = Ng, Np

        # define network architecture
        self.encoder = torch.nn.Linear(Np, Ng, bias=False)
        self.RNN = torch.nn.RNN(
            input_size=2,
            hidden_size=Ng,
            num_layers=1,
            nonlinearity=nonlinearity,
            bias=False,
            batch_first=True,
        )
        # Linear read-out weights
        self.decoder = torch.nn.Linear(Ng, Np, bias=False)

    def g(self, inputs):
        v, p0 = inputs
        init_state = self.encoder(p0)
        # return only final prediction in sequence
        return self.RNN(v, init_state[None])[-1][0]

    def forward(self, inputs, softmax=False):
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






