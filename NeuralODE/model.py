import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.utils.data
import matplotlib.pyplot as plt
import torch.optim as optim
# from torch.autograd.functional import broyden
import torch.autograd as autograd
from torchdyn.models import NeuralODE


class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(28 * 28, 64),  # Flatten MNIST images to 784 dimensions
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 28 * 28)  # We will reshape back to 28x28 in the model
        )

    def forward(self, t, x, *args, **kwargs):
        return self.net(x)


class ODEModel(nn.Module):
    def __init__(self):
        super(ODEModel, self).__init__()
        self.odefunc = ODEFunc()
        self.neural_ode = NeuralODE(self.odefunc, solver='dopri5', sensitivity='adjoint')
        self.fc = nn.Linear(784, 10)  # Final layer to map to 10 class outputs

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten MNIST images
        x = self.neural_ode(x)[1][-1, :]
        x = self.fc(x)  # Map from 784 to 10 dimensions for classification
        return x