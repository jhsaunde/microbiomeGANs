from utils.utils import process_config

config = process_config(config_file, exp_name=exp_name)

for bs in [32,64,128]:
    for lr in [0.1, 0.01, 0.001, 0.0001]:
        for loss in ["mse", "l1", "kl", "wasserstein"]:
            config.trainer.batch_size = bs
            config.generator.lr = lr
            config.generator.loss = loss
            trainer = build_trainer(config=config)
            trainer.train()
            test(model=trainer.generator, config=config)


