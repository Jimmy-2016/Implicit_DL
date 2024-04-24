
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from model import *
import torch.utils.data
import matplotlib.pyplot as plt
import torch.optim as optim
# from torch.autograd.functional import broyden



class DEQLayer(nn.Module):
    def __init__(self, in_features, hidden_units, out_features):
        super(DEQLayer, self).__init__()
        self.transformation = nn.Sequential(
            nn.Linear(in_features, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, out_features),
            nn.Tanh()
        )

    def forward(self, x):
        z = torch.zeros_like(x)
        for _ in range(50):  # Simple fixed-point iteration loop
            z = self.transformation(z + x)


        # def func(z):
        #     return z - self.transformation(z)
        #
        # # Finding the fixed point
        # equilibrium_point, _ = broyden(func, torch.zeros_like(x), tol=1e-4, max_iter=50)
        return z


class DEQModel(nn.Module):
    def __init__(self):
        super(DEQModel, self).__init__()
        self.flatten = nn.Flatten()
        self.deq_layer = DEQLayer(784, 256, 784)  # MNIST images are 28x28
        self.fc = nn.Linear(784, 10)  # Output layer for 10 classes

    def forward(self, x):
        x_input = self.flatten(x)
        x_deq = self.deq_layer(x_input)
        out = self.fc(x_deq)
        return out