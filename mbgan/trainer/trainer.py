from utils.losses import MSELoss, L1Loss
from torch import optim
import itertools
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

from collections import defaultdict
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AETrainer():
    def __init__(self, config, data_loader, generator):
        self.config = config
        self.generator = generator
        self.data_loader = data_loader
        self.loss = MSELoss()
        self.optimizer = optim.Adam(self.generator.parameters(), lr=self.config.generator.hyperparameters.lr,
                                    betas=(self.config.generator.hyperparameters.beta1,
                                           self.config.generator.hyperparameters.beta2))
        self.writer = SummaryWriter(log_dir=self.config.logdir)

    def train_on_batch(self, sixteens, wgs):
        self.optimizer.zero_grad()
        sixteens = sixteens.to(DEVICE)
        wgs = wgs.to(DEVICE)

        # Generate
        wgs_generated = self.generator(sixteens)

        loss = self.loss(wgs_generated, wgs)

        loss.backward()
        self.optimizer.step()
        return loss


    def train(self):
        self.generator.train()
        avg_loss = []
        for epoch in range(self.config.trainer.num_epochs):
            metrics = defaultdict()
            losses = []

            for real_A, real_B in self.data_loader:
                loss = self.train_on_batch(real_A, real_B)
                loss = loss.detach().numpy().tolist()
                losses.append(loss)


            print(f"Epoch [{epoch}/{self.config.trainer.num_epochs}]\t" 
                  f"Average MSE reconstruction loss: {sum(losses) / len(losses):.4f}\t")

            epoch_avg = sum(losses)/len(losses)
            avg_loss.append(epoch_avg)
            metrics["train/mse_loss"] = epoch_avg
            self.writer.add_scalar(tag='holder', scalar_value=epoch_avg, global_step=epoch)


        df = pd.DataFrame({"Average MSE" : avg_loss})
        csv_filename = f"{self.config.exp.name}.csv"
        df.to_csv(os.path.join(self.config.exp.experiment_dir, csv_filename))


class SimpleGANTrainer():
    def __init__(self, config, data_loader, generator, discriminator):
        self.config = config
        self.generator = generator
        self.discriminator = discriminator
        self.data_loader = data_loader
        self.loss = MSELoss()
        self.optimizer = optim.Adam(self.generator.parameters(),
                                    lr=self.config.generator.hyperparameters.lr,
                                    betas=(
                                        self.config.generator.hyperparameters.beta1,
                                        self.config.generator.hyperparameters.beta2)
                                    )
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(),
                                                  lr=self.config.discriminator.hyperparameters.lr,
                                                  betas=(
                                                      self.config.discriminator.hyperparameters.beta1,
                                                      self.config.discriminator.hyperparameters.beta2
                                                         )
                                                  )
        self.writer = SummaryWriter()

    def train_on_batch(self, sixteens, wgs):
        self.optimizer.zero_grad()
        sixteens = sixteens.to(DEVICE)
        wgs = wgs.to(DEVICE)

        # Generate
        wgs_generated = self.generator(sixteens)

        loss = self.loss(wgs_generated, wgs)

        loss.backward()
        self.optimizer.step()
        return loss

    def train(self):
        for epoch in range(self.config.trainer.num_epochs):
            losses = []
            for real_A, real_B in self.data_loader:
                loss = self.train_on_batch(real_A, real_B)
                losses.append(loss)

            print(f"Epoch [{epoch}/{self.config.trainer.num_epochs}]\t"
                  f"Average MSE reconstruction loss: {sum(losses) / len(losses):.4f}\t")
            self.writer.add_scalar

        df = pd.DataFrame({"Average MSE" : losses})
        csv_filename = f"{self.config.exp.name}.csv"
        df.to_csv(os.path.join(self.config.exp.experiment_dir, csv_filename))


class CycleGANTrainer():
    def __init__(self, config, data_loader, g_ab, g_ba, d_a, d_b):
        self.config = config
        self.G_AB = g_ab
        self.G_BA = g_ba
        self.D_A = d_a
        self.D_B = d_b
        self.data_loader = data_loader

        self.G_optimizer = optim.Adam(itertools.chain(self.G_AB.parameters(), self.G_BA.parameters()),
                                      lr=self.config.generator.hyperparameters.lr,
                                      betas=(self.config.generator.hyperparameters.beta1,
                                             self.config.generator.hyperparameters.beta2))

        self.D_A_optimizer = optim.Adam(self.D_A.parameters(),
                                        lr=self.config.discriminator.hyperparameters.lr,
                                        betas=(self.config.discriminator.hyperparameters.beta1,
                                               self.config.discriminator.hyperparameters.beta2
                                               )
                                        )
        self.D_B_optimizer = optim.Adam(self.D_B.parameters(),
                                        lr=self.config.discriminator.hyperparameters.lr,
                                        betas=(
            self.config.discriminator.hyperparameters.beta1, self.config.discriminator.hyperparameters.beta2))

        self.MSE_loss = MSELoss()
        self.L1_loss = L1Loss()

        self.D_A_losses = []
        self.D_B_losses = []
        self.G_losses = []
        self.writer = SummaryWriter()

    def generate_A_B(self, real_A):
        fake_B = self.G_AB(real_A)
        return fake_B

    def generate_B_A(self, real_B):
        fake_A = self.G_BA(real_B)
        return fake_A

    def backward_D_A(self, real_A, fake_A):
        # Real A
        self.D_A_optimizer.zero_grad()
        D_real_A = self.D_A(real_A)
        D_real_A_loss = self.MSE_loss(D_real_A, torch.ones_like(D_real_A))

        # Fake A
        D_fake_A = self.D_A(fake_A.detach())
        D_fake_A_loss = self.MSE_loss(D_fake_A, torch.zeros_like(D_fake_A))

        # Backward and optimize
        D_A_loss = (D_real_A_loss + D_fake_A_loss) * 0.5

        D_A_loss.backward()
        self.D_A_optimizer.step()

        # Append loss to list
        self.D_A_losses.append(D_A_loss.item())

    def backward_D_B(self, real_B, fake_B):
        # Real B
        self.D_B_optimizer.zero_grad()
        D_real_B = self.D_B(real_B)
        D_real_B_loss = self.MSE_loss(D_real_B, torch.ones_like(D_real_B))

        # Fake B
        D_fake_B = self.D_B(fake_B.detach())
        D_fake_B_loss = self.MSE_loss(D_fake_B, torch.zeros_like(D_fake_B))

        # Backward and optimize
        D_B_loss = (D_real_B_loss + D_fake_B_loss) * 0.5

        D_B_loss.backward()
        self.D_B_optimizer.step()

        # Append loss to list
        self.D_B_losses.append(D_B_loss.item())

    def backward_G(self, real_A, real_B, fake_A, fake_B):
        # Adversarial loss

        self.G_optimizer.zero_grad()

        D_fake_A = self.D_A(fake_A)
        G_AB_adv_loss = self.MSE_loss(D_fake_A, torch.ones_like(D_fake_A))

        D_fake_B = self.D_B(fake_B)
        G_BA_adv_loss = self.MSE_loss(D_fake_B, torch.ones_like(D_fake_B))

        # Cycle consistency loss
        reconstructed_A = self.G_BA(fake_B)
        G_cycle_loss_A = self.L1_loss(reconstructed_A, real_A)

        reconstructed_B = self.G_AB(fake_A)
        G_cycle_loss_B = self.L1_loss(reconstructed_B, real_B)

        # Total generator loss
        G_AB_loss = G_AB_adv_loss + G_cycle_loss_A
        G_BA_loss = G_BA_adv_loss + G_cycle_loss_B

        # Backward and optimize

        G_loss = G_AB_loss + G_BA_loss

        G_loss.backward()

        self.G_optimizer.step()

        # Append loss to list
        self.G_losses.append(G_loss.item())

    def train_on_batch(self, real_A, real_B):
        real_A = real_A.to(DEVICE)
        real_B = real_B.to(DEVICE)

        # Generate fake images
        fake_B = self.generate_A_B(real_A=real_A)
        fake_A = self.generate_B_A(real_B=real_B)

        # Train discriminators
        self.backward_D_A(real_A=real_A, fake_A=fake_A)
        self.backward_D_B(real_B=real_B, fake_B=fake_B)
        # Train generators
        self.backward_G(real_A=real_A, real_B=real_B, fake_A=fake_A, fake_B=fake_B)

    def train(self):
        for epoch in range(self.config.trainer.num_epochs):

            for real_A, real_B in self.data_loader:
                self.train_on_batch(real_A, real_B)

            print(f"Epoch [{epoch}/{self.config.trainer.num_epochs}]\t"
                  f"Loss_D_A: {self.D_A_losses[-1]:.4f}\t"
                  f"Loss_D_B: {self.D_B_losses[-1]:.4f}\t"
                  f"Loss_G: {self.G_losses[-1]:.4f}")

        df = pd.DataFrame({"D_A_losses" : self.D_A_losses, "D_B_losses" : self.D_B_losses, "G_losses" : self.G_losses})
        df.to_csv(f"C:/MSc/dev/mbgan/logs/{self.config.exp.name}.csv")



