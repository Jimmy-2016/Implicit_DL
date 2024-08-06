
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.utils.data
import matplotlib.pyplot as plt
import torch.optim as optim
# from torch.autograd.functional import broyden
from solvers import anderson
import torch.autograd as autograd



class ResNetLayer(nn.Module):
    def __init__(self, n_channels, n_inner_channels, kernel_size=3, num_groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, n_inner_channels, kernel_size, padding=kernel_size // 2, bias=False)
        self.conv2 = nn.Conv2d(n_inner_channels, n_channels, kernel_size, padding=kernel_size // 2, bias=False)
        self.norm1 = nn.GroupNorm(num_groups, n_inner_channels)
        self.norm2 = nn.GroupNorm(num_groups, n_channels)
        self.norm3 = nn.GroupNorm(num_groups, n_channels)
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)

    def forward(self, z, x):
        y = self.norm1(F.relu(self.conv1(z)))
        return self.norm3(F.relu(z + self.norm2(x + self.conv2(y))))



def forward_iteration(f, x0, max_iter=50, tol=1e-2):
    f0 = f(x0)
    res = []
    for k in range(max_iter):
        x = f0
        f0 = f(x)
        res.append((f0 - x).norm().item() / (1e-5 + f0.norm().item()))
        if (res[-1] < tol):
            break
    return f0, res


class DEQFixedPoint(nn.Module):
    def __init__(self, f, solver, **kwargs):
        super().__init__()
        self.f = f
        self.solver = solver
        self.kwargs = kwargs

    def forward(self, x):
        # compute forward pass and re-engage autograd tape
        with torch.no_grad():
            z, self.forward_res = self.solver(lambda z: self.f(z, x), torch.zeros_like(x), **self.kwargs)
        z = self.f(z, x)

        # set up Jacobian vector product (without additional forward calls)
        z0 = z.clone().detach().requires_grad_()
        f0 = self.f(z0, x)

        def backward_hook(grad):
            g, self.backward_res = self.solver(lambda y: autograd.grad(f0, z0, y, retain_graph=True)[0] + grad,
                                               grad, **self.kwargs)
            return g

        z.register_hook(backward_hook)
        return z



class DEQModel(nn.Module):
    def __init__(self):
        super(DEQModel, self).__init__()
        self.flatten = nn.Flatten()
        self.f = ResNetLayer(48, 64, kernel_size=3)
        # self.deq_layer = DEQFixedPoint(self.f, anderson, tol=1e-2, max_iter=25, m=5),  # MNIST images are 28x28
        self.fc = nn.Sequential(nn.Conv2d(1, 48, kernel_size=3, bias=True, padding=1),
                      nn.BatchNorm2d(48),
                      DEQFixedPoint(self.f, anderson, tol=1e-2, max_iter=25, m=5),
                      nn.Linear(200, 10)# Output layer for 10 classes
                                )

    def forward(self, x):
        # x_input = self.flatten(x)
        # x_deq = self.deq_layer(x)
        out = self.fc(x)
        return out

