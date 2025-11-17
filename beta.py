import math
import numpy as np

def gamma_bound_srinivas09 (t,D_size,sigma,kxx) : 
    inside_log = 1+ sigma**-2 * kxx * t * D_size
    return D_size * math.log(inside_log) 

def beta_srinivas09(t,D_size,sigma,kxx, delta = 0.1, B = 0) : 
    return 2*B + 300 * (math.log(t/delta))**3 * gamma_bound_srinivas09(t,D_size,sigma,kxx) 

def beta_fiedler21( R, lieklihood_noise,K,delta = 0.05 ,B=0) :
    K = K.detach().numpy()
    lam_bar = max(1,lieklihood_noise)
    inside_det = K + lam_bar * np.eye(K.shape[0]) 
    inside_log = np.linalg.det(inside_det)
    inside_sqr = math.log(inside_log)-2*math.log(delta)
    return B+R*math.sqrt(inside_sqr)

