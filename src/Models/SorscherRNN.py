import torch
import numpy as np
import tqdm

from ratsimulator import trajectory_generator


class SorscherRNN(torch.nn.Module):
    """
    Model based on:
    https://github.com/ganguli-lab/grid-pattern-formation/blob/master/model.py
    """

    def __init__(
        self, place_cell_ensembles, Ng=4096, Np=512, nonlinearity="relu", **kwargs
    ):
        super(SorscherRNN, self).__init__(**kwargs)
        self.place_cell_ensembles = place_cell_ensembles
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

        # initialise model weights
        for param in self.parameters():
            torch.nn.init.xavier_uniform_(param.data, gain=1.0)

        # pruning
        self.original_weights = []
        self._rnnh_prune_mask = torch.ones((Ng, Ng), device=self.device)
        self._rnni_prune_mask = torch.ones((Ng, 2), device=self.device)
        self._prune_mask_idxs = []

    def to(self, device=None, *args, **kwargs):
        """Overwrites: To also add place_cells on same device"""
        # copy place cells before putting on gpu avoid conflict with dataloader
        import copy

        self.place_cell_ensembles = [
            copy.copy(place_cells) for place_cells in self.place_cell_ensembles
        ]
        if device is not None:
            for i, place_cells in enumerate(self.place_cell_ensembles):
                self.place_cell_ensembles[i].pcs = place_cells.pcs.to(device)
        return super(SorscherRNN, self).to(device, *args, **kwargs)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def prune_mask(self):
        return [self._rnni_prune_mask, self._rnnh_prune_mask]

    @property
    def prune_mask_idxs(self):
        return self._prune_mask_idxs

    @prune_mask.setter
    def prune_mask(self, idxs):
        """
        Set outgoing weights corresponding to output nodes at
        idxs to zero in recurrent matrix - I.e. prune grid cells.
        """
        if not self.original_weights:
            # save weights
            self.original_weights = [
                self.RNN.weight_ih_l0.clone(),
                self.RNN.weight_hh_l0.clone(),
            ]

        # reset weights
        self.RNN.weight_ih_l0, self.RNN.weight_hh_l0 = map(
            lambda x: torch.nn.parameter.Parameter(x.clone()), self.original_weights
        )
        self._rnni_prune_mask, self._rnnh_prune_mask = (
            torch.ones((self.Ng, 2), device=self.device),
            torch.ones((self.Ng, self.Ng), device=self.device),
        )

        maski = torch.ones((self.Ng, 2), device=self.device)
        maskh = torch.ones((self.Ng, self.Ng), device=self.device)
        maski[idxs] = torch.zeros(2, device=self.device)
        maskh[idxs] = torch.zeros(self.Ng, device=self.device)
        self.RNN.weight_ih_l0 = torch.nn.parameter.Parameter(
            self.RNN.weight_ih_l0 * maski
        )
        self.RNN.weight_hh_l0 = torch.nn.parameter.Parameter(
            self.RNN.weight_hh_l0 * maskh
        )
        self._rnni_prune_mask = maski
        self._rnnh_prune_mask = maskh
        self._prune_mask_idxs = idxs

    def g(self, inputs):
        v, p0 = inputs
        if len(v.shape) == 2 and len(p0.shape) == 1:
            # model requires tensor of degree 3 (B,S,N)
            # assume inputs missing empty batch-dim
            v, p0 = v[None], p0[None]

        if self.device != v.device:
            # put v and p0 on same device as model
            v = v.to(self.device, dtype=self.dtype)
            p0 = p0.to(self.device, dtype=self.dtype)

        p0 = self.init_position_encoder(p0)
        p0 = p0[None]  # add dummy (unit) dim for number of stacked rnns (D)
        # output of torch.RNN is a 2d-tuple. First element =>
        # return_sequences=True (in tensorflow). Last element => False.
        out, _ = self.RNN(v, p0)
        return out

    def p(self, g_inputs, log_softmax=False):
        place_preds = self.decoder(g_inputs)
        return (
            torch.nn.functional.log_softmax(place_preds, dim=-1)
            if log_softmax
            else place_preds
        )

    def forward(self, inputs, log_softmax=False):
        gs = self.g(inputs)
        return self.p(gs, log_softmax)

    def CE(self, log_predictions, labels):
        return torch.mean(-torch.sum(labels * log_predictions, axis=-1))

    def entropy(self, labels):
        """
        The entropy. Note that entropy is a positive measure.
        Hence the negation at the start. When used to calculate KL:
        KL = CE - entropy
        """
        return torch.mean(
            -torch.sum(torch.nan_to_num(labels * torch.log(labels)), axis=-1)
        )

    def KL(self, cross_entropy_value, entropy_value):
        return cross_entropy_value - entropy_value

    def l2_reg(self, weight_decay):
        return weight_decay * torch.sum(self.RNN.weight_hh_l0 ** 2)

    def loss_fn(self, log_predictions, labels, weight_decay):
        """
        Args:
            inputs ((B, S, 2), (B, Np)): velocity and position in pc-basis
            labels (B, Np): ground truth pc population activations
        """
        # Actual cross entropy between two distributions p(x) and q(x),
        # rather than classic CE implementations assuming one-hot p(x).
        if labels.device != self.device:
            labels = labels.to(self.device, dtype=self.dtype)
        return self.CE(log_predictions, labels) + self.l2_reg(weight_decay)

    def position_error(self, preds, labels, positions, indices):
        batch_size = labels.shape[0]
        pred_error, true_error = 0, 0
        for env_i, place_cells in enumerate(self.place_cell_ensembles):
            mask_i = np.array(indices) == env_i
            decoded_pred_pos = place_cells.to_euclid(preds[mask_i])
            decoded_true_pos = place_cells.to_euclid(labels[mask_i])
            # sum over patchy-batch dim (mean after loop), mean over seq_len dim,
            # "sum" (euclidean) over spatial dim
            pred_error = torch.sum(
                torch.mean(
                    torch.sqrt(
                        torch.sum(
                            (decoded_pred_pos - positions[mask_i, 1:]) ** 2, axis=-1
                        )
                    ),
                    axis=-1,
                )
            )
            true_error = torch.sum(
                torch.mean(
                    torch.sqrt(
                        torch.sum(
                            (decoded_true_pos - positions[mask_i, 1:]) ** 2, axis=-1
                        )
                    ),
                    axis=-1,
                )
            )
        pred_error /= batch_size
        true_error /= batch_size
        return pred_error, true_error

    def train(
        self,
        trainloader,
        optimizer,
        weight_decay,
        nepochs,
        checkpoint_path,
        params,
        loss_history=[],
        training_metrics={},
    ):
        start_epoch = 1
        if loss_history:
            start_epoch = len(loss_history)
        else:
            training_metrics["CE"] = []
            training_metrics["entropy"] = []
            training_metrics["KL"] = []
            training_metrics["l2_reg"] = []
            training_metrics["pred_error"] = []
            training_metrics["true_error"] = []

        save_iter = 0
        pbar = tqdm.tqdm(range(start_epoch, nepochs + 1))
        for epoch in pbar:
            # generic torch training loop
            running_loss = 0.0
            running_ce, running_entropy, running_KL, running_l2_reg = 0, 0, 0, 0
            running_pred_error, running_true_error = 0, 0
            for inputs, labels, positions, indices in trainloader:
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                log_predictions = self(inputs, log_softmax=True)
                loss = self.loss_fn(log_predictions, labels, weight_decay)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                # update training metrics
                labels = labels.to(self.device, dtype=self.dtype)
                positions = positions.to(self.device, dtype=self.dtype)
                pred_error, true_error = self.position_error(
                    log_predictions, labels, positions, indices
                )
                cross_entropy_value = self.CE(log_predictions, labels).item()
                entropy_value = self.entropy(labels).item()
                running_ce += cross_entropy_value
                running_entropy += entropy_value
                running_KL += self.KL(cross_entropy_value, entropy_value)
                running_l2_reg += self.l2_reg(weight_decay).item()
                running_pred_error += pred_error.item()
                running_true_error += true_error.item()

            # add training metrics to training history
            loss_history.append(running_loss / len(trainloader))
            training_metrics["CE"].append(running_ce / len(trainloader))
            training_metrics["entropy"].append(running_entropy / len(trainloader))
            training_metrics["KL"].append(running_KL / len(trainloader))
            training_metrics["l2_reg"].append(running_l2_reg / len(trainloader))
            training_metrics["pred_error"].append(running_pred_error / len(trainloader))
            training_metrics["true_error"].append(running_true_error / len(trainloader))

            save_iter += 1
            # save model training dynamics (model weight history)
            if save_iter >= int(np.sqrt(epoch)) or epoch == nepochs:
                # save frequency gets sparser (sqrt) with training
                params["optimizer_state_dict"] = optimizer.state_dict()
                params["loss_history"] = loss_history
                params["training_metrics"] = training_metrics
                params["model_state_dict"] = self.state_dict()
                torch.save(params, checkpoint_path / f"{epoch:04d}")
                save_iter = 0

            # update tqdm training-bar description
            pbar.set_description(
                f"Epoch={epoch}/{nepochs}, loss={loss_history[-1]}, decoding_error(pred/true)={training_metrics['pred_error'][-1]}/{training_metrics['true_error'][-1]}"
            )
        return loss_history

    def train_old(
        self,
        trainloader,
        optimizer,
        weight_decay,
        nepochs,
        checkpoint_path,
        params,
        save_freq1=1,
        save_freq2=10,
        epoch_to_change_save_freq=100,
        loss_history=[],
        training_metrics={},
    ):
        """
        Modified generic train loop for sorscher rnn. Data is arbitrary
        large since it is generated and not a fixed set.
        """
        start_epoch = 1
        if loss_history:
            start_epoch = (
                save_freq1 * min(len(loss_history), epoch_to_change_save_freq)
                + save_freq2
                * (
                    max(len(loss_history), epoch_to_change_save_freq)
                    - epoch_to_change_save_freq
                )
                + 1
            )
        else:
            training_metrics["CE"] = []
            training_metrics["entropy"] = []
            training_metrics["KL"] = []
            training_metrics["l2_reg"] = []
        pbar = tqdm.tqdm(range(start_epoch, nepochs + 1))
        for epoch in pbar:
            # generic torch training loop
            running_loss = 0.0
            running_ce, running_entropy, running_KL, running_l2_reg = 0, 0, 0, 0
            for inputs, labels in trainloader:
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                log_predictions = self(inputs, log_softmax=True)
                loss = self.loss_fn(log_predictions, labels, weight_decay)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                # update training metrics
                if labels.device != self.device:
                    labels = labels.to(self.device, dtype=self.dtype)
                cross_entropy_value = self.CE(log_predictions, labels).item()
                entropy_value = self.entropy(labels).item()
                running_ce += cross_entropy_value
                running_entropy += entropy_value
                running_KL += self.KL(cross_entropy_value, entropy_value)
                running_l2_reg += self.l2_reg(weight_decay).item()

            # add training metrics to training history
            loss_history.append(running_loss / len(trainloader))
            training_metrics["CE"].append(running_ce / len(trainloader))
            training_metrics["entropy"].append(running_entropy / len(trainloader))
            training_metrics["KL"].append(running_KL / len(trainloader))
            training_metrics["l2_reg"].append(running_l2_reg / len(trainloader))

            # save model training dynamics (model weight history)
            if ((not (epoch % save_freq1)) and epoch < epoch_to_change_save_freq) or (
                (not (epoch % save_freq2)) and epoch > epoch_to_change_save_freq
            ):
                params["optimizer_state_dict"] = optimizer.state_dict()
                params["loss_history"] = loss_history
                params["training_metrics"] = training_metrics
                params["model_state_dict"] = self.state_dict()
                torch.save(params, checkpoint_path / f"{epoch:04d}")

            # update tqdm training-bar description
            pbar.set_description(f"Epoch={epoch}/{nepochs}, loss={loss_history[-1]}")
        return loss_history
