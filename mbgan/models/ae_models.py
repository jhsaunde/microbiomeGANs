from torch import nn


#  Create Generator
class Autoencoder(nn.Module):
    def __init__(self, input_size, output_size, nnodes=64):
        super().__init__()

        self.fc1 = nn.Linear(input_size, nnodes)
        self.bn1 = nn.BatchNorm1d(nnodes, momentum=0.8)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(nnodes, nnodes)
        self.bn2 = nn.BatchNorm1d(nnodes, momentum=0.8)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(nnodes, nnodes)
        self.bn3 = nn.BatchNorm1d(nnodes, momentum=0.8)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(nnodes, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.softmax(x)
        return x
