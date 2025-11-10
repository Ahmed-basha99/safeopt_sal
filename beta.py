import math

def gamma_bound_srinivas09 (t,D_size,sigma,kxx) : 
    inside_log = 1+ sigma**-2 * kxx * t * D_size
    return D_size * math.log(inside_log) 

def beta_srinivas09(t,D_size,sigma,kxx, delta = 0.1, B = 1) : 
    return 2*B + 300 * (math.log(t/delta))**3 * gamma_bound_srinivas09(t,D_size,sigma,kxx)