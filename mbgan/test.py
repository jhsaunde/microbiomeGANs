import argparse
import torch
import yaml
from utils.utils import process_config
from mbgan.models import ae_models
from dataset.dataloader import MBDataloader, MBTestDataset
import pandas as pd
import os

def main(config_file: str, exp_name: str):
    config = process_config(config_file, exp_name=exp_name)

    dataset = MBTestDataset(config=config)
    data_loader = MBDataloader(dataset=dataset, config=config, shuffle=True)

    model = ae_models.Autoencoder(config.data.input_size, config.data.output_size)
    model.load_state_dict(torch.load(os.path.join(config.exp.experiment_dir, "autoencoder_exp.py")))
    model.eval()

    with torch.no_grad():
        results = []
        for data in data_loader:
            inputs = data
            predicted_outputs = model(inputs)
            predicted_outputs = predicted_outputs.detach().numpy().tolist()
            results.append(predicted_outputs)


        df = pd.DataFrame({"RA" : results})
        csv_filename = f"{config.exp.name}_results.csv"
        df.to_csv(os.path.join(config.exp.experiment_dir, csv_filename))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_file", type=str, default="config/autoencoder.yml", help="config path to use")
    ap.add_argument("--exp.name", type=str, default="nonamed_autoencoder_exp")
    args = vars(ap.parse_args())

    main(config_file=args["config_file"], exp_name=args["exp.name"])
