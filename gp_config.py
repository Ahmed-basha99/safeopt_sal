import gpytorch, torch, math

# training : 
epochs = 25
lr = 0.2
optimizer_class = torch.optim.Adam
# model : 
mean = gpytorch.means.ConstantMean()
kernel = gpytorch.kernels.RBFKernel()
likelihood = gpytorch.likelihoods.GaussianLikelihood()

# data :
n_train = 100 
n_test = 51 
train_x = torch.linspace(0,1, n_train)
train_y = torch.sin(train_x * (2*math.pi)) + torch.randn(train_x.size()) * math.sqrt(0.04)
test_x = torch.linspace(0,1, n_test)
test_y = torch.sin(test_x * (2*math.pi)) + torch.randn(test_x.size()) * math.sqrt(0.04)