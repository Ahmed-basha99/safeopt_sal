import math
import torch
import gpytorch
from matplotlib import pyplot as plt
from GPR import GPR

# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

train_x = torch.linspace(0, 1, 100)
train_y = torch.sin(train_x * (2*math.pi)) + torch.randn(train_x.size()) * math.sqrt(0.04)
    
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = GPR(train_x, train_y, likelihood)

model.train()
likelihood.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range (50) : 
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f   noise: %.3f   L_scale: %.3f' % (
        i + 1, 50, loss.item(),
        model.likelihood.noise.item(),
        model.covar_module.base_kernel.lengthscale.item()))
    optimizer.step()

# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()

# Test points are regularly spaced along [0,1]
# Make predictions by feeding model through likelihood
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_x = torch.linspace(0, 1, 51)
    observed_pred = likelihood(model(test_x))
    print(observed_pred)

