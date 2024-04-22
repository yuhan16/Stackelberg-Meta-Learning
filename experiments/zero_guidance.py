import os
import numpy as np
from sg_meta.agent import Follower
from sg_meta.utils import save_results
from sg_meta import train_data_dir


def noguide(theta, x0):
    '''Compute follower's trajectory without the leader's guidance.'''
    ff = Follower(theta)
    ff.coeff[1] = 0     # set zero guidance cost

    ua = np.zeros(ff.dimua)
    pa = np.zeros(ff.dimxa)
    ITER = 200
    ub_traj = []
    xb_traj = []
    xb_traj.append(x0[ff.dimxa:])
    
    for i in range(ITER):
        x = np.concatenate( (pa, xb_traj[i]) )
        ub_traj.append( ff.get_br(x, ua) )
        
        if xb_traj[i][0] > 10 or xb_traj[i][0] < -0.1 or xb_traj[i][1] > 10 or xb_traj[i][1] < -0.1:
            print('follower goes outside the working space.')
            break

        x_tp1 = ff.dynamics(x, ua, ub_traj[i])
        xb_traj.append( x_tp1[ff.dimxa:] )
        print(xb_traj[i], ub_traj[i])
    
    with np.printoptions(precision=1):
        dname = os.path.join(train_data_dir, 'no_guide')
        res = {
            f'f{theta}_{x0[ff.dimxa:]}.npy': np.array(xb_traj),
        }
        #fname = os.path.join(train_data_dir, 'no_guide', f'f{theta}_{x0[ff.dimxa:]}.npy')
        #np.save(fname, np.array(xb_traj))
        save_results(dname, res)
    #print(xb_traj)
    #print(ub_traj)
    return


if __name__ == '__main__':
    theta = 4
    x0_list = [
        [0,0, 0,4,0.0],
        [0,0, 0,8,0.5],
        [0,0, 6,0,3.0],
    ]   # add other initial states
    x0 = np.array(x0_list[0])

    noguide(theta, x0)