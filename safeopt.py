import torch 
import gpytorch
import config_safeopt as cfg
from GPTrainer import GPTrainer
import math
import gp_config as gp_cfg

def safeopt_algo():

    train_x = cfg.INITIAL_X
    train_y = cfg.INITIAL_Y

    gp_trainer = GPTrainer(gp_cfg, train_x, train_y)

    C_lower = torch.full_like(cfg.DOMAIN.flatten(), -float('inf'))
    C_upper = torch.full_like(cfg.DOMAIN.flatten(), float('inf'))
    C_lower[cfg.INITIAL_SAFE_INDICES] = cfg.SAFETY_THRESHOLD
    
    S_mask = torch.zeros(cfg.N_POINTS, dtype=torch.bool)
    S_mask[cfg.INITIAL_SAFE_INDICES] = True
    
    # Pre-compute pairwise distances for the discrete domain D
    distance_matrix = torch.cdist(cfg.DOMAIN, cfg.DOMAIN)

    for i in range(cfg.N_ITERATIONS):

        gp_trainer = GPTrainer(gp_cfg, train_x, train_y)
        gp_trainer.train()
        
        mean, std_dev = gp_trainer.get_posterier(cfg.DOMAIN)

        Q_lower = mean - math.sqrt(cfg.BETA) * std_dev
        Q_upper = mean + math.sqrt(cfg.BETA) * std_dev

        C_lower = torch.max(C_lower, Q_lower) 
        C_upper = torch.min(C_upper,Q_upper)

        S_prev_indices = torch.where(S_mask)[0]
        new_S_mask = S_mask.clone()
        for s_idx in S_prev_indices:
            # set of indicies for all points that are lipschitz safe relative to x[s_idx]
            lipschitz_safe = C_lower[s_idx] - cfg.LIPSCHITZ_CONSTANT * distance_matrix[s_idx] >= cfg.SAFETY_THRESHOLD
            new_S_mask = torch.logical_or(new_S_mask, lipschitz_safe)
        S_mask = new_S_mask

        if not torch.any(S_mask):
            print(f"Iteration {i+1}: Safe set is empty. Stopping.")
            
        wt_D = C_upper - C_lower
        wt_S = torch.full_like(wt_D, -1e9)
        wt_S[S_mask] = wt_D[S_mask]

        xt = cfg.DOMAIN[torch.argmax(wt_S)]
        yt = cfg.ground_truth(xt) + torch.randn(1) * 0.2 # 

        print (train_y.shape, yt.unsqueeze(0).shape , yt.shape)
        train_x = torch.cat([train_x, xt.unsqueeze(0)])
        train_y = torch.cat([train_y, yt])


print(f"Final train set size : {len(train_x)} \n Final safe set = {torch.where(S_mask)[0].tolist()}")

if __name__ == '__main__':
    safeopt_algo()
