import torch
import math

#  D set 

N_POINTS = 25
DOMAIN = torch.linspace(0, 1, N_POINTS).view(-1, 1) # Reshape for GPyTorch

def ground_truth(x):
    if x.ndim == 0: x = x.view(-1)
    return torch.sin(x * (2 * math.pi)) + torch.randn(x.size()) * 0.2

# h
SAFETY_THRESHOLD = 0.2

INITIAL_SAFE_INDICES = torch.tensor([5, 4]) # Corresponds to x=0.25, x=0.29
INITIAL_X = DOMAIN[INITIAL_SAFE_INDICES]
INITIAL_Y = ground_truth(INITIAL_X).flatten()

LIPSCHITZ_CONSTANT = 7.0

BETA = 2.0

N_ITERATIONS = 15