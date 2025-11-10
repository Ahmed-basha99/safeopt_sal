import math
import torch
import gpytorch
from matplotlib import pyplot as plt
from GPR import GPR
import gp_config as config 

class GPTrainer () : 
    def __init__(self, cfg, train_x, train_y) : 
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = cfg.test_x
        self.test_y = cfg.test_y
        self.epochs = cfg.epochs
        self.lr = cfg.lr
        self.likelihood = cfg.likelihood
        self.model = GPR(self.train_x, self.train_y, self.likelihood)
        self.optimizer = cfg.optimizer_class(self.model.parameters(), lr=self.lr)

    def update_training_data(self, train_x, train_y) : 
        self.train_x = train_x
        self.train_y = train_y
        self.model.set_train_data(self.train_x, self.train_y, strict = False)

    def train(self) :
        self.model.train()
        self.likelihood.train()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        
        for i in range (config.epochs) : # check this later
            self.optimizer.zero_grad()
            output = self.model(self.train_x)
            loss = -mll(output, self.train_y)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f   noise: %.3f   Length_scale: %.3f   output_scale: %.3f' % (
                i + 1, config.epochs, loss.item(),
                self.model.likelihood.noise.item(),
                self.model.covar_module.base_kernel.lengthscale.item(),
                self.model.covar_module.outputscale.item()))
            
            self.optimizer.step()

    def plot(self) :
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(self.test_x))
            f,ax = plt.subplots(1,1, figsize=(8,6))
            lower, upper = observed_pred.confidence_region()
            ax.plot(self.train_x.numpy(), self.train_y.numpy(), 'k*')
            ax.plot(self.test_x.numpy(), observed_pred.mean.numpy(), 'b')
            ax.fill_between(self.test_x.numpy(), lower.numpy(), upper.numpy(),alpha = 0.5)
            ax.set_ylim([-3,3])
            ax.legend(["observed data", "posterier mean", "uncertainty bounds"])
            plt.show() 
    
    def evaluate(self) : 
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(self.test_x))
            mean = observed_pred.mean
            mse = torch.mean((mean - self.test_y)**2)
            print(f'Mean Squared Error : {mse.item()}')
            return mse.item()
        
    def get_posterier(self,x) : 
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = self.likelihood(self.model(x))
            mean = posterior.mean
            std_dev = posterior.stddev 
            return mean, std_dev
    def save_model(self, path) :
        torch.save(self.model.state_dict(), path)



# trainer = GPTrainer(config)
# trainer.train()
# trainer.plot()
# trainer.evaluate()