from torch import nn
import torch

class LogTransformLayer(nn.Module):
    def __init__(self):
        super(LogTransformLayer, self).__init__()

    def forward(self, x):
        return torch.log((1 + 1000 * x) / (1 + x))

# Create Generators

# Autoencoder_3L
# 128-64-128 or 64-32-64
class Autoencoder_3L(nn.Module):
    def __init__(self, input_size, output_size, nnodes=128, use_log_transform=True, use_batch_norm=True):
        super().__init__()

        self.use_log_transform = use_log_transform
        self.log_transform = LogTransformLayer()

        encoder_layers = [
            nn.Linear(input_size, nnodes),
            nn.BatchNorm1d(nnodes) if use_batch_norm else nn.Identity(),
            nn.ReLU(),

            nn.Linear(nnodes, nnodes // 2),
            nn.BatchNorm1d(nnodes // 2) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
        ]

        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = [
            nn.Linear(nnodes // 2, nnodes),
            nn.BatchNorm1d(nnodes) if use_batch_norm else nn.Identity(),
            nn.ReLU(),

            nn.Linear(nnodes, output_size),
            nn.Softmax(dim=1),
        ]

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        if self.use_log_transform:
            x = self.log_transform(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return x



# Autoencoder_5L
# 128-64-32-64-128 or 64-32-16-32-64
class Autoencoder_5L(nn.Module):
    def __init__(self, input_size, output_size, nnodes=128, use_log_transform=True, use_batch_norm=True):
        super().__init__()

        self.use_log_transform = use_log_transform
        self.log_transform = LogTransformLayer()

        encoder_layers = [
            nn.Linear(input_size, nnodes),
            nn.BatchNorm1d(nnodes) if use_batch_norm else nn.Identity(),
            nn.ReLU(),

            nn.Linear(nnodes, nnodes // 2),
            nn.BatchNorm1d(nnodes // 2) if use_batch_norm else nn.Identity(),
            nn.ReLU(),

            nn.Linear(nnodes // 2, nnodes // 4),
            nn.BatchNorm1d(nnodes // 4) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
        ]

        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = [
            nn.Linear(nnodes // 4, nnodes // 2),
            nn.BatchNorm1d(nnodes // 2) if use_batch_norm else nn.Identity(),
            nn.ReLU(),

            nn.Linear(nnodes // 2, nnodes),
            nn.BatchNorm1d(nnodes) if use_batch_norm else nn.Identity(),
            nn.ReLU(),

            nn.Linear(nnodes, output_size),
            nn.Softmax(dim=1),
        ]

        self.decoder = nn.Sequential(*decoder_layers)


    def forward(self, x):
        if self.use_log_transform:
            x = self.log_transform(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Feedforward-Decoder
# 64-64 or 128-128
class Feedforward_Decoder2L(nn.Module):
    def __init__(self, input_size, output_size, nnodes=128, use_log_transform=True, use_batch_norm=False):
        super().__init__()

        self.use_log_transform = use_log_transform
        self.log_transform = LogTransformLayer()

        decoder_layers = [

            nn.Linear(input_size, nnodes),
            nn.BatchNorm1d(nnodes) if use_batch_norm else nn.Identity(),
            nn.ReLU(),

            nn.Linear(nnodes, nnodes * 2),
            nn.BatchNorm1d(nnodes*2) if use_batch_norm else nn.Identity(),
            nn.ReLU(),

            nn.Linear(nnodes * 2, output_size),
            nn.Softmax(dim=1),

        ]

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        if self.use_log_transform:
            x = self.log_transform(x)
        x = self.decoder(x)
        return x



# Feedforward-Decoder
# 64-64 or 128-128
class Feedforward_Decoder3L(nn.Module):
    def __init__(self, input_size, output_size, nnodes=128, use_log_transform=True, use_batch_norm=False):
        super().__init__()

        self.use_log_transform = use_log_transform
        self.log_transform = LogTransformLayer()

        decoder_layers = [

            nn.Linear(input_size, nnodes),
            nn.BatchNorm1d(nnodes) if use_batch_norm else nn.Identity(),
            nn.ReLU(),

            nn.Linear(nnodes, nnodes * 2),
            nn.BatchNorm1d(nnodes * 2) if use_batch_norm else nn.Identity(),
            nn.ReLU(),

            nn.Linear(nnodes * 2, nnodes * 4),
            nn.BatchNorm1d(nnodes * 4) if use_batch_norm else nn.Identity(),
            nn.ReLU(),

            nn.Linear(nnodes * 4, output_size),
            nn.Softmax(dim=1),

        ]

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        if self.use_log_transform:
            x = self.log_transform(x)
        x = self.decoder(x)
        return x


class Feedforward_1L(nn.Module):
    def __init__(self, input_size, output_size, nnodes=128, use_log_transform=True, use_batch_norm=False):
        super().__init__()

        self.use_log_transform = use_log_transform
        self.log_transform = LogTransformLayer()

        decoder_layers = [

            nn.Linear(input_size, nnodes),
            nn.BatchNorm1d(nnodes) if use_batch_norm else nn.Identity(),
            nn.ReLU(),

            nn.Linear(nnodes, output_size),
            nn.Softmax(dim=1),

        ]

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        if self.use_log_transform:
            x = self.log_transform(x)
        x = self.decoder(x)
        return x



class Feedforward_2L(nn.Module):
    def __init__(self, input_size, output_size, nnodes=128, use_log_transform=True, use_batch_norm=False):
        super().__init__()

        self.use_log_transform = use_log_transform
        self.log_transform = LogTransformLayer()

        decoder_layers = [

            nn.Linear(input_size, nnodes),
            nn.BatchNorm1d(nnodes) if use_batch_norm else nn.Identity(),
            nn.ReLU(),

            nn.Linear(nnodes, nnodes),
            nn.BatchNorm1d(nnodes) if use_batch_norm else nn.Identity(),
            nn.ReLU(),

            nn.Linear(nnodes, output_size),
            nn.Softmax(dim=1),

        ]

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        if self.use_log_transform:
            x = self.log_transform(x)
        x = self.decoder(x)
        return x


class Feedforward_3L(nn.Module):
    def __init__(self, input_size, output_size, nnodes=128, use_log_transform=True, use_batch_norm=False):
        super().__init__()

        self.use_log_transform = use_log_transform
        self.log_transform = LogTransformLayer()

        decoder_layers = [
            nn.Linear(input_size, nnodes),
            nn.BatchNorm1d(nnodes) if use_batch_norm else nn.Identity(),
            nn.ReLU(),

            nn.Linear(nnodes, nnodes),
            nn.BatchNorm1d(nnodes) if use_batch_norm else nn.Identity(),
            nn.ReLU(),

            nn.Linear(nnodes, nnodes),
            nn.BatchNorm1d(nnodes) if use_batch_norm else nn.Identity(),
            nn.ReLU(),

            nn.Linear(nnodes, output_size),
            nn.Softmax(dim=1),
        ]

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        if self.use_log_transform:
            x = self.log_transform(x)
        x = self.decoder(x)
        return x
