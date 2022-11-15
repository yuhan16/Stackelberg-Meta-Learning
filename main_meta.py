"""
This is the main script for testing and main exectuation.
"""

import torch
import numpy as np
from utils_meta import BRNet, Leader, Follower, Meta

def test():
    print(1)

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
            follower_theta = Follower(theta)

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
    test()