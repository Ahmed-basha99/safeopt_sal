import torch
import math

#  D set 

N_POINTS = 200
DOMAIN = torch.linspace(0, 1, N_POINTS).view(-1, 1) 

def ground_truth(x):
    if x.ndim == 0: x = x.view(-1)
    return torch.sin(x * (6 * math.pi) + 0.5) 
# h
SAFETY_THRESHOLD = -0.2

INITIAL_SAFE_INDICES = torch.tensor([ 1,2]) 
INITIAL_X = DOMAIN[INITIAL_SAFE_INDICES]
INITIAL_Y = ground_truth(INITIAL_X).flatten()

LIPSCHITZ_CONSTANT = 6*math.pi # max derivative 

BETA = 4.0

# domain discritaztion 

N_ITERATIONS = 15