"""
This script defines global parameters.
- Vector parameters should be converted to numpy type for computation.
- Scalar parameters (alp, beta) do not need conversion.
"""
# environment settings 
ws_len = 10         # working space edge length (working space is a square)
total_type = 4      # number of types of the follower, type starting from 0
type_pdf = [0.3, 0.2, 0.3, 0.2]
kappa = 3           # D_traj_num / D_random_num ratio
dima = 2            # dimension for leader's action
dimb = 2            # dimension for follower's action
dimx = 5            # dimension for state
dimxA = 2           # dimension for leader's state
dimxB = 3           # dimension for follower's state

# meta learning parameter
gam = 5             # weight in meta cost function
momentum = 0.8      # momentum in SGD
lr = 0.005          # learning rate (step size) in SGD for each task
lr_meta = 0.005     # learning rate (step size) in SGD for updating the entire model

# discritization parameters
dt = 0.5            # time step for discritization
Tf = 8             # time horizon in continuous setting

# soft constraint parameters
mu = 50
nu = 50

"""
    Define parameters given scenario and type information.
    - obs:      obstacle position and safe distance, [x,y,d]
    - pf:       final destination, [x,y]
    - x0A:      leader's starting position, [x,y]
    - x0B:      follower's starting position and angle, [x,y,phi]
    - q, qf:    leader's objective parameters, order aligns with the paper
    - w:        follower's objective parameter, order aligns with the paper
"""
def scenario0(theta):
    obs = []    # can also be different for different theta
    pf = [8., 8.]
    if theta == 0:
        x0A = [0., 0.5]
        x0B = [0.5, 0.5, 0.]
        q = [1.,1.,2.,2.,0.], [10.,10.], 1., [.1,.1]        # q4 is for a @ q4 @ a
        qf = [2., 2., 5., 5., 0.], [10., 10.]
        w = [ [0.,0., 1.,1.,0.], [3., 3.], 3., [.1, .1] ]      # w4 is for b @ w4 @ b
    elif theta == 1:        
        x0A = [0., 7.]
        x0B = [0.5, 8., 0.]
        q = [1.,1.,2.,2.,0.], [10.,10.], 1., [.1,.1]
        qf = [2., 2., 5., 5., 0.], [10., 10.]
        w = [ [0.,0., 2.,2.,0.], [1., 1.], 1., [.1, .1] ]
    elif theta == 2:
        x0A = [1., 4.]
        x0B = [2., 5., 0.]
        q = [1.,1.,2.,2.,0.], [10.,10.], 1., [.1,.1]
        qf = [2., 2., 5., 5., 0.], [10., 10.]
        w = [ [0.,0., .1,.1,0.], [3., 3.], 3., [.0, .0] ]
    else:
        x0A = [0., 5.5]
        x0B = [0., 6., 1.]
        q = [1.,1.,2.,2.,0.], [10.,10.], 1., [.1,.1]
        qf = [2., 2., 5., 5., 0.], [10., 10.]
        w = [ [0.,0., 2.,2.,0.], [3., 3.], 3., [.5, .5] ]
    return obs, x0A, x0B, pf, q, qf, w


def scenario1(theta):
    obs = [[3.5,6,1], [6,3,1]]
    pf = [8., 8.]
    if theta == 0:
        x0A = [0., 0.5]
        x0B = [0.5, 0.5, 0.]
        q = [1.,1.,2.,2.,0.], [10.,10.], 1., [.1,.1]        # q4 is for a @ q4 @ a
        qf = [2., 2., 5., 5., 0.], [10., 10.]
        w = [ [0.,0., 1.,1.,0.], [3., 3.], 3., [.1, .1] ]      # w4 is for b @ w4 @ b
    elif theta == 1:        
        x0A = [0., 7.]
        x0B = [0.5, 8., 0.]
        q = [1.,1.,2.,2.,0.], [10.,10.], 1., [.1,.1]
        qf = [2., 2., 5., 5., 0.], [10., 10.]
        w = [ [0.,0., 2.,2.,0.], [1., 1.], 1., [.1, .1] ]
    elif theta == 2:
        x0A = [1., 4.]
        x0B = [2., 5., 0.]
        q = [1.,1.,2.,2.,0.], [10.,10.], 1., [.1,.1]
        qf = [2., 2., 5., 5., 0.], [10., 10.]
        w = [ [0.,0., 1,1,0.], [1., 1.], 1., [.0, .0] ]
    else:
        x0A = [0., 5.5]
        x0B = [0., 6., 0.]
        q = [1.,1.,2.,2.,0.], [10.,10.], 1., [.1,.1]
        qf = [2., 2., 5., 5., 0.], [10., 10.]
        w = [ [0.,0., 2.,2.,0.], [3., 3.], 3., [.5, .5] ]
    return obs, x0A, x0B, pf, q, qf, w


def scenario2(theta):
    obs = [[2,7,1], [3,3.5,1.], [7,5,1]]
    pf = [8., 8.]
    if theta == 0:
        x0A = [0., 0.5]
        x0B = [0.5, 0.5, 0.]
        q = [1.,1.,2.,2.,0.], [10.,10.], 1., [.1,.1]        # q4 is for a @ q4 @ a
        qf = [2., 2., 5., 5., 0.], [10., 10.]
        w = [ [0.,0., 1.,1.,0.], [.5, .5], .5, [.1, .1] ]      # w4 is for b @ w4 @ b
    elif theta == 1:        
        x0A = [0., 7.5]
        x0B = [0., 8., 1.]
        q = [1.,1.,2.,2.,0.], [10.,10.], 1., [.1,.1]
        qf = [2., 2., 5., 5., 0.], [10., 10.]
        w = [ [0.,0., 2.,2.,0.], [2., 2.], 2., [.1, .1] ]
    elif theta == 2:
        x0A = [1., 4.]
        x0B = [2., 5., 0.]
        q = [1.,1.,2.,2.,0.], [10.,10.], 1., [.1,.1]
        qf = [2., 2., 5., 5., 0.], [10., 10.]
        w = [ [0.,0., .1,.1,0.], [.1, .1], .1, [.0, .0] ]
    else:
        x0A = [0.5, 6]
        x0B = [0., 6., 0.]
        q = [1.,1.,2.,2.,0.], [10.,10.], 1., [.1,.1]
        qf = [2., 2., 5., 5., 0.], [10., 10.]
        w = [ [0.,0., 2.,2.,0.], [1., 1.], 1., [.5, .5] ]
    return obs, x0A, x0B, pf, q, qf, w


def get_param(scn, theta):
    if scn == 0:
        return scenario0(theta)
    elif scn == 1:
        return scenario1(theta)
    else:
        return scenario2(theta)

def print_key_parameters():
    print('Parameters:')
    print('- type_pdf:', type_pdf)
    print('- kappa:', kappa)
    print('- gamma:', gam)

    print('- alp, inner GD :', lr)
    print('- beta, meta GD:', lr_meta)
    print('- SGD momentum:', momentum)

    print('- total time horizon:', int(Tf/dt))
    print('- discretization time:', dt)
    print('- soft safety constraint penalty coeff:', nu)
    print('- soft dynamic constraint penalty coeff:', mu)

    print('')
print_key_parameters()