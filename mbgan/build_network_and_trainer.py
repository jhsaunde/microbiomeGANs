from models import ae_models, cyclegan_models
from trainer import trainer


def build_generator(config):
    # because the objective of this network is to go from 16s to wgs, the input is 16s, and output is wgs
    input_size = config.data.input_size  # 16s
    output_size = config.data.output_size  # wgs

    if config.exp.model_spec == "autoencoder":
        return ae_models.Autoencoder(input_size=input_size, output_size=output_size)
    elif config.exp.model_spec == "autoencoder1":
        return ae_models.Autoencoder1(input_size=input_size, output_size=output_size)
    elif config.exp.type == "simplegan":
        return ae_models.model_spec(input_size=input_size, output_size=output_size)
    elif config.exp.type == "cyclegan":
        # two generators for cyclegan
        return cyclegan_models.Generator(input_size=input_size, output_size=output_size), cyclegan_models.Generator(
            input_size=output_size, output_size=input_size)

    else:
        raise ValueError("exp type should be either autoencoder, cyclegan, or simplegan")


def build_discriminator(config):
    input_size = config.data.input_size  # 16s
    output_size = config.data.output_size  # wgs

    if config.exp.type == "autoencoder":
        raise ValueError("build discriminator should not be called with just autoencoder")
    elif config.exp.type == "simplegan":
        return cyclegan_models.Discriminator(input_size=input_size)
    elif config.exp.type == "cyclegan":
        # two generators for cyclegan
        return cyclegan_models.Discriminator(input_size=input_size), cyclegan_models.Discriminator(
            input_size=output_size)
    else:
        raise ValueError("exp type should be either autoencoder, cyclegan, or simplegan")


def build_trainer(config, data_loader, networks):
    if "autoencoder" in config.exp.type:
        return trainer.AETrainer(config=config, data_loader=data_loader, generator=networks["generator"])
    elif config.exp.type == "simplegan":
        return trainer.SimpleGANTrainer(config=config, data_loader=data_loader, generator=networks["generator"],
                                        discriminator=networks["discriminator"])
    elif config.exp.type == "cyclegan":
        return trainer.CycleGANTrainer(config=config, data_loader=data_loader, g_ab=networks["g_ab"],
                                       g_ba=networks["g_ba"], d_a=networks["d_a"], d_b=networks["d_b"])
    else:
        raise ValueError("exp type should be either autoencoder, cyclegan, or simplegan")
