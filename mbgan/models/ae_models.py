from torch import nn

#  Create Generator
class Autoencoder(nn.Module):
    def __init__(self, input_size, output_size, nnodes=64):
        super().__init__()

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
            nn.Linear(nnodes, nnodes * 2),
            nn.ReLU(),
            nn.Linear(nnodes * 2, nnodes * 4),
            nn.ReLU(),
            nn.Linear(nnodes * 4, output_size),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
