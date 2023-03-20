import argparse
import torch
import yaml

from utils.utils import process_config
from dataset.dataloader import MBDataloader, MBDataset

from build_network_and_trainer import build_generator, build_discriminator, build_trainer

torch.manual_seed(42)

def print_file(filename):
    file = open(filename)
    for line in file:
        print(line)


def main(config_file: str, exp_name: str):
    config = process_config(config_file, exp_name=exp_name)

    dataset = MBDataset(config=config)
    data_loader = MBDataloader(dataset=dataset, config=config, shuffle=True)

    kwargs = {}
    if config.exp.type == "autoencoder":
        generator = build_generator(config)
        kwargs["generator"] = generator

    elif config.exp.type == "simplegan":
        generator = build_generator(config)
        discriminator = build_discriminator(config)
        kwargs["generator"] = generator
        kwargs["discriminator"] = discriminator

    elif config.exp.type == "cyclegan":
        g_ab, g_ba = build_generator(config)
        d_a, d_b = build_discriminator(config)
        kwargs["g_ab"] = g_ab
        kwargs["g_ba"] = g_ba
        kwargs["d_a"] = d_a
        kwargs["d_b"] = d_b

    trainer = build_trainer(config=config, data_loader=data_loader, networks=kwargs)

    print_file(config_file)

    print("#########################################################")
    print(f"Experiment: {exp_name} has started!")
    print("#########################################################")

    trainer.train()

    # Common pytorch convention to save file as .py or .pth
    torch.save(trainer.generator.state_dict(), config.exp.experiment_dir+config.exp.name+".py")

    return


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_file", type=str, default="config/autoencoder.yml", help="config path to use")
    ap.add_argument("--exp.name", type=str, default="autoencoder_exptt1")
    args = vars(ap.parse_args())

    main(config_file=args["config_file"], exp_name=args["exp.name"])
