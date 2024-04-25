
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import torch
from cvxpylayers.torch import CvxpyLayer


n = 7

# Define variables & parameters
x = cp.Variable()
y = cp.Parameter(n)

# Define objective and constraints
objective = cp.sum_squares(y - x)
constraints = []

# Synthesize problem
prob = cp.Problem(cp.Minimize(objective), constraints)

# Set parameter values
y.value = np.random.randn(n)

# Solve problem in one line
prob.solve(requires_grad=True)
print("solution:", "%.3f" % x.value)
print("analytical solution:", "%.3f" % np.mean(y.value))

# Set gradient wrt x
x.gradient = np.array([1.])

# Differentiate in one line
prob.backward()
print("gradient:", y.gradient)
print("analytical gradient:", np.ones(y.size) / n)

