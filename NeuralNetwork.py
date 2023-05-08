from torch import nn


class MyFlatten(nn.Module):
    def forward(self, x):
        """x: [b, h, w] -> [b, h*w]"""
        return x.reshape(x.shape[0], -1)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = MyFlatten()
        self.fc = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x).float()
        logits = self.fc(x)
        return logits
