import torch
import math

#  D set 

N_POINTS = 200
DOMAIN = torch.linspace(0, 1, N_POINTS).view(-1, 1) 

# N_POINTS_X = 50
# N_POINTS_Y = 50
# N_POINTS = N_POINTS_X * N_POINTS_Y

# # Create 1D linspaces for each dimension
# x_lin = torch.linspace(0, 1, N_POINTS_X)
# y_lin = torch.linspace(0, 1, N_POINTS_Y)

# # Create a 2D grid
# grid_x, grid_y = torch.meshgrid(x_lin, y_lin, indexing='ij')

# Flatten the grid to create the domain (N_POINTS_TOTAL, 2)
# DOMAIN = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

def ground_truth_2d(x) : 
    if x.ndim == 1:
        x = x.unsqueeze(0)
        
    x_0 = x[:, 0]
    x_1 = x[:, 1]
    
    fun = torch.sin(x_0 * (6 * math.pi) + 0.5) + torch.cos(x_1 * (6 * math.pi) + 0.5)
    
    return fun.flatten()

def ground_truth_1d(x):
    if x.ndim == 0: x = x.view(-1)
    return torch.sin(x * (6 * math.pi) + 0.5) 

ground_truth = ground_truth_1d
SAFETY_THRESHOLD = -0.2

# INITIAL_SAFE_INDICES = torch.tensor([ 1,2]) 
# INITIAL_X = DOMAIN[INITIAL_SAFE_INDICES]
# INITIAL_Y = ground_truth(INITIAL_X).flatten()

# idx_1 = 0 * N_POINTS_Y + 10
# idx_2 = 20 * N_POINTS_Y + 20
# INITIAL_SAFE_INDICES = torch.tensor([idx_1, idx_2]) 
INITIAL_SAFE_INDICES = torch.tensor([20,66,80])

INITIAL_X = DOMAIN[INITIAL_SAFE_INDICES]
INITIAL_Y = ground_truth(INITIAL_X).flatten()

LIPSCHITZ_CONSTANT = 6*math.pi # + math.sqrt(2.0) # max derivative 

BETA = 100

# domain discritaztion 

N_ITERATIONS = 15