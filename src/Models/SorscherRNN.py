import torch


class SorscherRNN(torch.nn.Module):
    """
    Model based on:
    https://github.com/ganguli-lab/grid-pattern-formation/blob/master/model.py
    """

    def __init__(
        self,
        input_size=2,
        Ng=4096,
        Np=512,
        weight_decay=1e-4,
        nonlinearity="relu",
        **kwargs
    ):
        super(SorscherRNN, self).__init__(**kwargs)
        self.input_size = input_size
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
        return self.RNN(v, p0)[-1][0]

    def forward(self, inputs, softmax=False):
        place_preds = self.decoder(self.g(inputs))
        return torch.nn.functional.softmax(place_preds) if softmax else place_preds

    def loss_fn(self, inputs, labels):
        """
        Args:
            inputs ((B, S, 2), (B, Np)): velocity and position in pc-basis
            labels (B, Np): ground truth pc population activations
        """
        prediction = self(inputs, softmax=True)
        # Actual cross entropy between two distributions p(x) and q(x),
        # rather than classic CE implementations assuming one-hot p(x).
        cross_entropy = torch.sum(labels * torch.log(prediction),axis=-1)
        return torch.mean(cross_entropy)








