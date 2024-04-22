import os
import numpy as np
import torch
from sg_meta.agent import Leader, Follower
from sg_meta.model import BRNet
from sg_meta.utils import save_results
from sg_meta import train_data_dir


def receding_horizon(theta, x0=None):
    """
    This function implements the receding horizon planning to guide type theta follower.
    """
    fname = os.path.join(train_data_dir, 'meta_data', f'adapt{theta}.pth')
    #fname = os.path.join(train_data_dir, 'indiv_data', f'br{theta}.pth')       # or use individual learned model for testing
    if not os.path.exists(fname):
        raise Exception(f'No BR model found for type {theta}. Run adaptation or other learning algorithms first.')

    leader = Leader()
    follower = Follower(theta)
    br = BRNet()
    br.load_state_dict(torch.load(fname))
    
    xd = np.array(leader.pd + follower.pd + [0])
    iter, ITER = 0, 10
    eps = 4e-1
    x_traj, ua_traj, ub_traj = [], [], []
    x_traj.append(x0)
    x_t = x0

    # get initial trajectory
    x_init, ua_init = leader.init2(x0)
    for iter in range(ITER):
        print(f'receding horizon planning  {iter+1}:')
        x_opt, a_opt = leader.compute_opt_traj(br.state_dict(), x_init, ua_init)
        ua_t = a_opt[0, :]
        
        # follower's response
        ub_t = follower.get_br(x_t, ua_t)
        x_tp1 = follower.dynamics(x_t, ua_t, ub_t)

        x_traj.append(x_tp1)
        ua_traj.append(ua_t)
        ub_traj.append(ub_t)
        #print(x_t, ua_t, ub_t)

        if np.linalg.norm(x_tp1[leader.dimxa: leader.dimx-1]-follower.pd) <= eps:
            break
        
        # reformulate initial conditions
        x_t = x_tp1
        x_init, ua_init = leader.init2(x_t)
        '''
        # or use the following simplified initialization
        x_init[:-1] = x_init[1:]
        x_init[-1] = xd
        ua_init[:-1] = ua_init[1:]
        ua_init[-1] = np.zeros(leader.dimua)
        '''
    
    save_flag = True
    if save_flag:
        dname = os.path.join(train_data_dir, 'rh_traj')
        with np.printoptions(precision=1):
            res = {
                f'x_type{theta}_{x0[0:2]}.npy': np.array(x_traj),
                f'ua_type{theta}_{x0[0:2]}.npy': np.array(ua_traj),
                f'ub_type{theta}_{x0[0:2]}.npy': np.array(ub_traj),
            }
        save_results(dname, res)



if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)

    theta = 1   # [0, ..., 4]
    
    #x0 = np.array([1, 8, 0, 8, 0.5])        # [x0_leader, x0_follower]
    #x0 = np.array([5, 1, 6, 0, 3.])
    x0 = np.array([0, 4.5, 0, 4, 0.])
    
    receding_horizon(theta, x0)