"""
This is the main script for testing and main exectuation.
"""

import torch
import os
import numpy as np
import param
from utils_meta import BRNet, Leader, Follower, Meta


def generate_random_br_data():
    """
    This function generates randomly sampled BR data for all scenarios.
    """
    TOTAL_SCE = 3
    TOTAL_TYPE = param.total_type
    meta = Meta()
    for scn in range(TOTAL_SCE):
        dir_name = 'data/br_data/scenario' + str(scn) + '/'             # check if directory exists
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        
        for theta in range(TOTAL_TYPE):
            fname = dir_name + 'task' + str(theta) + '_uniform.npy'     # check if data file exists
            if not os.path.exists(fname):
                ff = Follower(scn, theta)
                task_data = meta.sample_task_theta_uniform(ff, N=10000)
                np.save(fname, task_data) 
                print('Generated randomly sampled BR data for scenario {} task {}.'.format(scn, theta))
            else:
                print('Sampled BR data for scenario {} task {} already exist.'.format(scn, theta))


def no_guide():
    """
    This function implements no-guidance control for the follower and record the corresponding trajectory.
    """
    TOTAL_SCE = 3
    TOTAL_TYPE = param.total_type
    
    for scn in range(TOTAL_SCE):
        for theta in range(TOTAL_TYPE):
            ff = Follower(scn, theta)
            x_traj, b_traj = ff.no_guidance()
            
            dir_name = 'data/data_noguide/scenario' + str(scn) + '/'
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)
            
            fname = dir_name + 'task' + str(theta) + '_xtraj.npy'
            np.save(fname, x_traj)
            fname = dir_name + 'task' + str(theta) + '_btraj.npy'
            np.save(fname, b_traj)


def main():
    meta = Meta()
    brnet = BRNet()
    leader = Leader()
    iter, ITER_MAX = 0, 1e4
    br_list, D2_list = [], []
    while iter <= ITER_MAX:
        task_sample = meta.sample_tasks(20)
        for theta in task_sample:
            leader.read_task_info(theta)
            follower_theta = Follower(0, theta)

            # sample D1_theta and update model
            x_traj, a_traj = leader.solve_oc(brnet) # ??? to be tested
            D1_theta = meta.sample_task_theta(follower_theta, x_traj, a_traj, N=100)
            br_list.append( meta.update_model_theta(leader, brnet, D1_theta, a_traj) ) 
            """br_list has the same (W,b) as brnet but updated gradient (dW,db) for different tasks."""

            # sample D2_theta for future update
            x_traj, a_traj = leader.solve_oc(brnet)
            D2_list.append( meta.sample_task_theta(follower_theta, x_traj, a_traj, N=100) )
        
        # update the entire BR model
        brnet = meta.update_model(leader, brnet, task_sample, br_list, D2_list)
        iter += 1


if __name__ == '__main__':
    #main()
    generate_random_br_data()       # pre-sample BR data for future use.
    no_guide()                      # implement no-guidance control for the follower.