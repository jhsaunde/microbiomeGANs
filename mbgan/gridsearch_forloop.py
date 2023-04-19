import argparse
import torch
import yaml
import os
from utils.utils import process_config
from models import ae_models
from dataset.dataloader import MBDataloader, MBDataset
from build_network_and_trainer import build_generator, build_discriminator, build_trainer
from test import test

config_file = "config/autoencoder.yml"
exp_name = "grid_search_test"
kwargs = {}


def run():
    for bs in [32, 128]:
        for nodes in [64, 128]:
            for lr in [0.1, 0.01, 0.001, 0.0001]:
                for loss in ["mse", "l1", "kl", "wasserstein"]:
                    for models in ['autoencoder3', 'autoencoder5']:

                        config = process_config(config_file, exp_name=exp_name)
                        dataset = MBDataset(config=config)
                        data_loader = MBDataloader(dataset=dataset, config=config)

                        config.exp.model_spec = models
                        config.exp.name = f"AE_BS{bs}_LR{lr}_LS{loss}_N{nodes}_M{models}"
                        config.trainer.batch_size = bs
                        config.generator.lr = lr
                        config.generator.loss = loss
                        config.generator.nodes = nodes

                        generator = build_generator(config)
                        kwargs["generator"] = generator

                        # Train the model
                        trainer = build_trainer(config=config, data_loader=data_loader, networks=kwargs)
                        trainer.train()

                        # Test the model
                        test(model=trainer.generator, config=config)


if __name__ == '__main__':
    run()