import torch


def CE(log_predictions, labels):
    """
    Cross entropy. Method assumes predictions are already log()-transformed.
    """
    return torch.mean(-torch.sum(labels * log_predictions, axis=-1))


def entropy(labels):
    """
    Information entropy
    """
    return torch.mean(-torch.sum(torch.nan_to_num(labels * torch.log(labels)), axis=-1))


def KL(cross_entropy_value, entropy_value):
    """
    The Kullback Leibler Divergence
    """
    return cross_entropy_value - entropy_value


def l2_reg(weight, weight_decay):
    """
    L2 regularization
    """
    return weight_decay * torch.sum(weight ** 2)


class Logger:
    def __init__(self, num_batches_in_epoch, loss_history=[], training_metrics={}):
        self.num_batches_in_epoch = num_batches_in_epoch
        self.training_metrics = training_metrics
        self.loss_history = loss_history

        if not training_metrics:
            self.training_metrics["CE"] = []
            self.training_metrics["entropy"] = []
            self.training_metrics["KL"] = []
            self.training_metrics["l2_reg"] = []
            self.training_metrics["pred_error"] = []
            self.training_metrics["true_error"] = []

    def new_epoch(self):
        for key in self.training_metrics:
            self.training_metrics[key].append(0)

        self.loss_history.append(0)

    def update(self, loss, log_predictions, labels, l2_reg, pred_error, true_error):
        cross_entropy_value = CE(log_predictions, labels).item()
        entropy_value = entropy(labels).item()

        self.training_metrics["CE"][-1] += cross_entropy_value / self.num_batches_in_epoch
        self.training_metrics["entropy"][-1] += entropy_value / self.num_batches_in_epoch

        self.training_metrics["KL"][-1] += (
            self.KL(cross_entropy_value, entropy_value) / self.num_batches_in_epoch
        )
        self.training_metrics["l2_reg"][-1] += l2_reg / self.num_batches_in_epoch
        self.training_metrics["pred_error"][-1] += pred_error / self.num_batches_in_epoch
        self.training_metrics["true_error"][-1] += true_error / self.num_batches_in_epoch

        self.loss_history[-1] += loss / self.num_batches_in_epoch












