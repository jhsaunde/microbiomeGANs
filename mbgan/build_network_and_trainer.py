from models import ae_models, cyclegan_models
from trainer import trainer


def build_generator(config):
    # Network objective is to transform 16S to WGS data, unless specified otherwise (CycleGAN)
    input_size = config.data.input_size  # 16s
    output_size = config.data.output_size  # wgs
    nodes = config.generator.nodes

    if config.exp.model_spec == "autoencoder3":
        return ae_models.Autoencoder_3L(input_size=input_size, output_size=output_size, nnodes=nodes)
    elif config.exp.model_spec == "autoencoder5":
        return ae_models.Autoencoder_5L(input_size=input_size, output_size=output_size, nnodes=nodes)
    elif config.exp.model_spec == "feedforwarddecoder2":
        return ae_models.Feedforward_Decoder2L(input_size=input_size, output_size=output_size, nnodes=nodes)
    elif config.exp.model_spec == "feedforwarddecoder3":
        return ae_models.Feedforward_Decoder3L(input_size=input_size, output_size=output_size, nnodes=nodes)
    elif config.exp.model_spec == "feedforward1":
        return ae_models.Feedforward_1L(input_size=input_size, output_size=output_size, nnodes=nodes)
    elif config.exp.model_spec == "feedforward2":
        return ae_models.Feedforward_2L(input_size=input_size, output_size=output_size, nnodes=nodes)
    elif config.exp.model_spec == "feedforward3":
        return ae_models.Feedforward_3L(input_size=input_size, output_size=output_size, nnodes=nodes)

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
