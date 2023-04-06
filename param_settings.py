"""
This scripts defines the parameters in the simulation and algorithms.
"""
import numpy as np


# random seed settings
seed = 7767
rng = np.random.default_rng(seed)


# environment settings
ws_len = 10
total_type = 5
type_pdf = [0.2, 0.3, 0.1, 0.3, 0.1]
# obstacles, [xc,yc,rc, norm, x_scale,y_scale], norm=-1 is inf norm
obs = [[2.5,2.8,1, -1,0.5,1.2], [7,2,0.8, 2,1,1], [2,7,0.8, 2,1,1], [6,8,1, 1,1,1]]
pd = [9, 9]     # destination


# system parameters
dimxa, dimxb = 2, 3
dimx = dimxa + dimxb
dimua, dimub = 2, 2
dt = 0.2
T = 2
dimT = int(T / dt)


# meta-learning parameters
alp = 1e-4      # inner loop learning rate
mom = 0.8       # momentum for sgd
beta = 1e-4     # outer loop (meta) learning rate
kappa = 2       # D_rand_num / D_obs_num ratio


# leader's cost parameters, {|x-xd|, |pa-pb|, |ua|}
q1 = 5*np.diag(np.ones(dimx))
q2 = 10*np.diag(np.ones(dimxa))
q3 = np.diag(np.ones(dimua))
qf1 = 5 * q1
qf2 = 5 * q2


# follower's cost parameters, {|pb-pd|, |pa-pb|, |ub|, obs}
def get_follower_param(theta):
    alp = []
    if theta == 0:
        alp = [1, 8, 1, 0.8] #[1, 5, 10, 0.8]
    elif theta == 1:
        alp = [1, 10, 2, 0.7] #[0.5, 10, 20, 0.6]
    elif theta == 2:
        alp = [1, 10, 2, 0.6] #[0.3, 10, 10, 0.5]
    elif theta == 3:
        alp = [1, 5, 0.5, 1] #[1.5, 1, 0, 1]
    elif theta == 4:
        alp = [1, 5, 0.3, 1.2] #[1.5, 1, 5, 1.2]
    else:
        raise Exception("Wrong type for the follower.")
    return alp


# different starting points
x0_list = [
    [2,1, 0,0, 0.78],
    [2,1, 1,1, 0],
    [2,1, 3,0, 1.57],
    [5,1, 5,0, 1.57],
    [5,1, 6,0, 3],
    [1,4, 0,5, 0],
    [1,4, 1,5, 0.1],
    [1,8, 0,8, 0.78]
]

def print_key_parameters():
    print('Parameters:')
    print('- type_pdf:', type_pdf)
    
    print('- alp in meta:', alp)
    print('- beta in meta:', beta)
    
    print('- prediction horizon:', T)
    print('- discretization time:', dt)

    print('- follower\'s coefficients: (|pb-pd|, |pa-pb|, |ub|, obs)')
    print('  - type 0:', get_follower_param(0))
    print('  - type 1:', get_follower_param(1))
    print('  - type 2:', get_follower_param(2))
    print('  - type 3:', get_follower_param(3))
    print('  - type 4:', get_follower_param(4))

    print('')
print_key_parameters()