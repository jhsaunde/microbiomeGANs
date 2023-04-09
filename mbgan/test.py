import torch
from dataset.dataloader import MBDataloader, MBTestDataset
import pandas as pd
import numpy as np
import os


def test(model, config):
    print("TESTING")
    dataset = MBTestDataset(config=config)
    data_loader = MBDataloader(dataset=dataset, config=config, shuffle=False, train=False)
    model.eval()


    with torch.no_grad():
        results = []
        for i, data in enumerate(data_loader):
            inputs = data
            predicted_outputs = model(inputs)
            predicted_outputs = predicted_outputs.detach().numpy().tolist()
            results.append(predicted_outputs)

        results = np.concatenate(results)
        df = pd.DataFrame(results)
        csv_filename = f"{config.exp.name}_results.csv"
        df.to_csv(os.path.join(config.logdir, csv_filename))


