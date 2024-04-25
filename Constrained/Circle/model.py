import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.utils.data
import matplotlib.pyplot as plt
import torch.optim as optim
from cvxpylayers.torch import CvxpyLayer
import cvxpy as cp
import torch.autograd as autograd
from torch.utils.data import Dataset, DataLoader
from matplotlib.patches import Circle


class CircleProjectionLayer(nn.Module):
    def __init__(self, radius, center):
        super().__init__()
        self.radius = radius

        # Create optimization variables.
        z = cp.Variable(2)

        # Set up the optimization problem: minimize ||z - f||^2 subject to z being on the circle.
        f = cp.Parameter(2)
        center = cp.Parameter(2)

        objective = cp.Minimize(cp.sum_squares(z - f))
        constraints = [cp.sum_squares(z - center) <= radius ** 2]

        # Create the CVXPY problem
        problem = cp.Problem(objective, constraints)

        # Convert the CVXPY problem to a CVXPY Layer.
        self.cvxpylayer = CvxpyLayer(problem, parameters=[f, center], variables=[z])

    def forward(self, x, center):
        # Solve the optimization problem.
        z, = self.cvxpylayer(x, center)
        return z


class MyNN(nn.Module):
    def __init__(self, input_size, radius, center):
        super().__init__()
        self.linear = nn.Linear(input_size, 2)  # Now we output 2D points.
        self.circle_projection = CircleProjectionLayer(radius, center)
        self.out = nn.Linear(2, 1)
        self.relu = nn.ReLU()

    def forward(self, x, center):
        # A simple linear transformation
        x = self.linear(x)

        # The output is projected onto the circle
        x = self.circle_projection(x, center)
        return self.out(x)



class CustomDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def plt_circle(radius, center_x, center_y):
    theta = torch.linspace(0, 2 * torch.pi, 100)

    x = center_x + radius * torch.cos(theta)
    y = center_y + radius * torch.sin(theta)

    return x, y