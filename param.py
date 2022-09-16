"""
This script defines global parameters.
- Vector parameters should be converted to numpy type for computation.
- Scalar parameters (alp, beta) do not need conversion.
"""
# environment settings 
ws_len = 10         # working space edge length (working space is a square)
total_type = 4      # number of types of the follower, type starting from 0
type_pdf = [0.1, 0.2, 0.3, 0.4]
dist_leader = 3     # safe distance for leader's guidance
kappa = 3           # D_traj_num / D_random_num ratio
dima = 2            # dimension for leader's action
dimb = 2            # dimension for follower's action
dimx = 4            # dimension for state
dimxA = 2           # dimension for leader's state
dimxB = 2           # dimension for follower's state

# meta learning parameter
mu = 0.9            # momentum in SGD
lr = 0.001          # learning rate (step size) in SGD for each task
lr_meta = 0.001     # learning rate (step size) in SGD for updating the entire model

# BVP parameters
dt = 0.1            # time step for discritization
Tf = 1             # time horizon in continuous setting


def get_param_from_type(theta):
    """
    This function returns the parameter given the follower's type information.
    - obs: obstacle position and safe distance, [x,y,d]
    - pf: final destination, [x,y]
    - p0A: leader's starting position, [x,y]
    - p0B: follower's starting position, [x,y]
    - alp: leader's obj parameter, [alp1, alp2, ...]
    - beta: follower's obj parameter, [beta1, beta2, ...]
    """
    if theta == 0:
        obs = [[1,1,1], [2,2,1], [3,3,1]]
        pf = [5,5]
        p0A = [0,0]
        p0B = [0,0]
        alp = [1,1,1,1,1]
        beta = [1,1,1, 1]
    elif theta == 1:
        obs = [[1,1,1], [2,2,1], [3,3,1]]
        pf = [5,5]
        p0A = [0,0]
        p0B = [0,0]
        alp = [1,1,1,1,1]
        beta = [1,1,1, 1]
    elif theta == 2:
        obs = [[1,1,1], [2,2,1], [3,3,1]]
        pf = [5,5]
        p0A = [0,0]
        p0B = [0,0]
        alp = [1,1,1,1,1]
        beta = [1,1,1,1]
    else:
        obs = [[1,1,1], [2,2,1], [3,3,1]]
        pf = [5,5]
        p0A = [0,0]
        p0B = [0,0]
        alp = [1,1,1,1,1]
        beta = [1,1,1,1]
    
    return obs, p0A, p0B, pf, alp, beta