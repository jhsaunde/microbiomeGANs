import torch
from dataset.dataloader import MBDataloader, MBDataset
import pandas as pd
import numpy as np
import os
from utils.losses import mse_loss, l1_loss, kl_loss, wasserstein_loss

LOSSES = {"mse": mse_loss, "l1": l1_loss, "kl": kl_loss, "wasserstein": wasserstein_loss}

def test(model, config):
    print('TESTING')
    dataset = MBDataset(config=config, train=False)
    data_loader = MBDataloader(dataset=dataset, config=config, train=False)
    loss = LOSSES[config.generator.loss]
    losses = []

    model.eval()  # Set the model to evaluation mode


    with torch.no_grad():
        results = []
        for data_16s, data_wgs in data_loader:
            predicted_wgs = model(data_16s)
            loss_vals = loss(predicted_wgs, data_wgs)   # Compute the loss using predicted_outputs
            losses.append(loss_vals.item())             # Store the loss value in the losses list
            results.append(predicted_wgs)               # Append predicted outputs to results list

        results = np.concatenate(results)
        df = pd.DataFrame(results)
        csv_filename = f"{config.exp.name}_results.csv"
        df.to_csv(os.path.join(config.logdir, csv_filename))

