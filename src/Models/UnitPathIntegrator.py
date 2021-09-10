import torch
import tqdm

class UnitPathIntegrator(torch.nn.Module):
    """
    Specification of the SorscherRNN model -
    stripping the model down to the bone in an effort to probe
    which parts of the model is necessary to do path integration.
    """

    def __init__(self, Ng=4096, Np=512, **kwargs):
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

    def forward(self, inputs, softmax=False):
        place_preds = self.decoder(self.g(inputs))
        return torch.nn.functional.softmax(place_preds, dim=-1) if softmax else place_preds

    def loss_fn(self, predictions, labels, weight_decay):
        """
        Args:
            inputs ((B, S, 2), (B, Np)): velocity and position in pc-basis
            labels (B, Np): ground truth pc population activations
        """
        # Actual cross entropy between two distributions p(x) and q(x),
        # rather than classic CE implementations assuming one-hot p(x).
        cross_entropy = torch.sum(- labels * torch.log(predictions),axis=-1)
        l2_regularization = weight_decay + torch.sum(self.recurrence.weight**2)
        return torch.mean(cross_entropy) + l2_regularization

    def train(self, trainloader, optimizer, weight_decay, nepochs, device, dtype=torch.float32):
        """
        Modified generic train loop for sorscher rnn. Data is arbitrary
        large since it is generated and not a fixed set.
        """
        loss_history = []
        pbar = tqdm.tqdm(range(1,nepochs+1))
        for epoch in pbar:
            # generic torch training loop
            running_loss = 0.0
            for inputs, labels in trainloader:
                inputs[0] = inputs[0].to(device,dtype=dtype)
                inputs[1] = inputs[1].to(device,dtype=dtype)
                labels = labels.to(device,dtype=dtype)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                predictions = self(inputs, softmax=True)
                loss = self.loss_fn(predictions, labels, weight_decay)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            loss_history.append(running_loss / len(trainloader))
            pbar.set_description(f"Epoch={epoch}, loss={loss_history[-1]}")


