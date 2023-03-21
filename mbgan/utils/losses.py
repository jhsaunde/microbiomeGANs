from torch.nn import MSELoss, L1Loss
from scipy.stats import wasserstein_distance
import torch

def wasserstein_loss():
    pass


def custom_loss():
    pass

def kl_divergence(y_pred, y_true):
    y_true = y_true + 1e-7  # Add a small value to avoid division by zero
    y_pred = y_pred + 1e-7  # Add a small value to avoid log of zero
    kl_div = y_true * torch.log(y_true / y_pred)
    loss = torch.sum(kl_div, dim=1).mean()  # Sum over species and average over samples
    return loss

def swd_loss(y_pred, y_true):
    batch_size, num_points = y_true.shape

    swd_values = []

    for i in range(batch_size):
        swd = wasserstein_distance(y_true[i].detach().cpu().numpy(),
                                   y_pred[i].detach().cpu().numpy())
        swd_values.append(swd)

    swd_loss = torch.tensor(swd_values, device=y_true.device, requires_grad=True).mean()
    return swd_loss