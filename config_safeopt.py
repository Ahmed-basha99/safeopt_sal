import torch
import math

#  D set 

N_POINTS = 50
DOMAIN = torch.linspace(0, 1, N_POINTS).view(-1, 1) 

def ground_truth(x):
    if x.ndim == 0: x = x.view(-1)
    return torch.sin(x * (2 * math.pi)) 
# h
SAFETY_THRESHOLD = -0.2

INITIAL_SAFE_INDICES = torch.tensor([ 4,25, 20 , 49]) 
INITIAL_X = DOMAIN[INITIAL_SAFE_INDICES]
INITIAL_Y = ground_truth(INITIAL_X).flatten()

LIPSCHITZ_CONSTANT =2*math.pi # max derivative 

BETA = 2.0

# domain discritaztion 

N_ITERATIONS = 15