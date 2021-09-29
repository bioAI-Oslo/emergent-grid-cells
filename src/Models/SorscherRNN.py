import torch
import tqdm

from ratsimulator import trajectory_generator


class SorscherRNN(torch.nn.Module):
    """
    Model based on:
    https://github.com/ganguli-lab/grid-pattern-formation/blob/master/model.py
    """

    def __init__(self, Ng=4096, Np=512, nonlinearity="relu", **kwargs):
        super(SorscherRNN, self).__init__(**kwargs)
        self.Ng, self.Np = Ng, Np

        # define network architecture
        self.init_position_encoder = torch.nn.Linear(Np, Ng, bias=False)
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

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def g(self, inputs):
        v, p0 = inputs
        if len(v.shape) != 3:
            # model requires tensor of degree 3 (B,S,N)
            # assume inputs missing empty batch-dim
            v, p0 = v[None], p0[None]

        if self.device != v.device:
            # put v and p0 on same device as model
            v = v.to(self.device, dtype=self.dtype)
            p0 = p0.to(self.device, dtype=self.dtype)

        p0 = self.init_position_encoder(p0)
        # return only final prediction in sequence
        out = self.RNN(v, p0[None])[0]
        return out

    def forward(self, inputs, log_softmax=False):
        place_preds = self.decoder(self.g(inputs))
        return (
            torch.nn.functional.log_softmax(place_preds, dim=-1)
            if log_softmax
            else place_preds
        )

    def loss_fn(self, predictions, labels, weight_decay):
        """
        Args:
            inputs ((B, S, 2), (B, Np)): velocity and position in pc-basis
            labels (B, Np): ground truth pc population activations
        """
        # Actual cross entropy between two distributions p(x) and q(x),
        # rather than classic CE implementations assuming one-hot p(x).
        if labels.device != self.device:
            labels = labels.to(self.device, dtype=self.dtype)
        cross_entropy = torch.sum(-labels * predictions, axis=-1)
        l2_regularization = weight_decay + torch.sum(self.RNN.weight_hh_l0 ** 2)
        return torch.mean(cross_entropy) + l2_regularization

    def save(self, optimizer, loss_history, params, tag, path="../checkpoints/"):
        model_name = type(self).__name__
        params["optimizer_state_dict"] = optimizer.state_dict()
        params["loss_history"] = loss_history
        params["model_state_dict"] = self.state_dict()
        torch.save(params, path + model_name + "_" + tag)

    def train(
        self,
        trainloader,
        optimizer,
        weight_decay,
        nepochs,
        loaded_model=False,
        save_model=False,
        save_freq=1,
        loss_history=[],
        *args,
        **kwargs,
    ):
        """
        Modified generic train loop for sorscher rnn. Data is arbitrary
        large since it is generated and not a fixed set.
        """
        start_epoch = 1
        if loaded_model:
            start_epoch = save_freq * len(loss_history) + 1
        pbar = tqdm.tqdm(range(start_epoch, nepochs + 1))
        for epoch in pbar:
            # generic torch training loop
            running_loss = 0.0
            for inputs, labels in trainloader:
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                predictions = self(inputs, log_softmax=True)
                loss = self.loss_fn(predictions, labels, weight_decay)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            loss_history.append(running_loss / len(trainloader))
            pbar.set_description(f"Epoch={epoch}/{nepochs}, loss={loss_history[-1]}")

            if not (epoch % save_freq):
                self.save(optimizer, loss_history, *args, **kwargs)
        return loss_history

