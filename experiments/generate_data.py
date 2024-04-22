import os
import numpy as np
from sg_meta.agent import Leader, Follower
from sg_meta.meta import Meta
from sg_meta.utils import save_results
from sg_meta import train_data_dir


def main():
    """
    Generates br data using randomly generated x_traj and ua_traj.
    The data is stored and used for data set in meta-learning.
    Each type of follower has a separate data set.
    """
    leader = Leader()
    leader.meta = Meta()
    
    for theta in range(leader.meta.total_type):
        N_data = 15000

        print(f'Generating simulation data for follower {theta} (random)...')
        data_theta_rand = leader.meta.sample_task_theta_random(theta, [], N=N_data)
        dname = os.path.join(train_data_dir, 'br_data')
        res = {
            f'f{theta}_rand.npy': data_theta_rand,
        }
        save_results(dname, res)
        
        print(f'Generating simulation data for follower {theta} (near obs)...')
        data_theta_obs = leader.meta.sample_task_theta_obs(theta, [], N=N_data)
        dname = os.path.join(train_data_dir, 'br_data')
        res = {
            f'f{theta}_obs.npy': data_theta_obs,
        }
        save_results(dname, res)
        

if __name__ == '__main__':
    main()