from torch.nn import MSELoss, L1Loss
import torch


mse_loss = MSELoss()

l1_loss = L1Loss()

def custom_loss():
    pass

def kl_loss(y_pred, y_true):
    y_true = y_true + 1e-7  # Add a small value to avoid division by zero
    y_pred = y_pred + 1e-7  # Add a small value to avoid log of zero
    kl_div = y_true * torch.log(y_true / y_pred)
    loss = torch.sum(kl_div, dim=1).mean()  # Sum over species and average over samples
    return loss

def wasserstein_loss(y_pred, y_true):
    return y_true.mean() - y_pred.mean()
