from torch import nn
import torch

class LogTransformLayer(nn.Module):
    def __init__(self):
        super(LogTransformLayer, self).__init__()

    def forward(self, x):
        return torch.log((1 + 1000 * x) / (1 + x))

#  Create Generator
class Autoencoder(nn.Module):
    def __init__(self, input_size, output_size, nnodes=128):
        super().__init__()

        self.log_transform = LogTransformLayer()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, nnodes),
            nn.ReLU(),
            nn.Linear(nnodes, nnodes // 2),
            nn.ReLU(),
            nn.Linear(nnodes // 2, nnodes // 4),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(nnodes // 4, nnodes // 2),
            nn.ReLU(),
            nn.Linear(nnodes // 2, nnodes),
            nn.ReLU(),
            nn.Linear(nnodes, output_size),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        #x = self.log_transform(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return x
