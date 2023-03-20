import argparse
import torch
import yaml
from utils.utils import process_config
from mbgan.models import ae_models
from dataset.dataloader import MBDataloader, MBTestDataset

def main(config_file: str, exp_name: str):
    config = process_config(config_file, exp_name=exp_name)

    dataset = MBTestDataset(config=config)
    data_loader = MBDataloader(dataset=dataset, config=config, shuffle=True)

    model = ae_models.Autoencoder(config.data.input_size, config.data.output_size)
    model.load_state_dict(torch.load("/Users/jamessaunders/dev/microbiomeGANs/mbgan/logs/tester.py"))
    model.eval()

    with torch.no_grad():
        for data in dataloader_16s:
            inputs = data
            predicted_outputs = model(inputs)
