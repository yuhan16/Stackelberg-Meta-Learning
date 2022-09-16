"""
This is the main script for testing and main exectuation.
"""

import torch
import numpy as np
from utils import BRNet, Leader, Follower, Meta


def test():     # base function test
    f1 = Follower(0)
    x = np.array([5,5,5,5])
    a = np.array([0.1, 0.1])
    b_opt = f1.get_br(x, a)
    #print(b_opt)
    #print(f1.compute_obj(x, a, b_opt))
    
    meta = Meta()
    print(meta.sample_tasks(100))
    d1 = meta.sample_task_theta_uniform(f1, 10)
    print(len(d1))

    brnet = BRNet()
    print(brnet(torch.tensor([[1.,2.,3.,4.,5.,6.], [1.,2.,3.,4.,5.,6.]])))


def test2():
    meta = Meta()
    brnet = BRNet()
    leader = Leader()
    leader.read_task_info(0)

    x_traj, a_traj = leader.solve_oc(brnet)
    oc_cost = leader.obj_oc(x_traj, a_traj)
    print(oc_cost)

    follower = Follower(0)
    task_data = meta.sample_task_theta(follower, x_traj, a_traj, N=100)
    br_cost = leader.obj_br(brnet, task_data)
    print(br_cost)

    oc_grad = leader.grad_obj_oc(brnet, a_traj)
    print(oc_grad.get_data())
    br_grad = leader.grad_obj_br(brnet, task_data)
    print(br_grad.get_grad())

    b2 = meta.update_model_theta(leader, brnet, task_data, a_traj)
    np.save('taskdata.npy', list2array(task_data))
    np.save('xtraj.npy', x_traj)
    np.save('atraj.npy', a_traj)

def test3():
    meta = Meta()
    brnet = BRNet()
    leader = Leader()
    leader.read_task_info(0)

    task_data = np.load('taskdata.npy')
    task_data = array2list(task_data)
    x_traj = np.load('xtraj.npy')
    a_traj = np.load('atraj.npy')

    oc_cost = leader.obj_oc(x_traj, a_traj)
    print(oc_cost)
    br_cost = leader.obj_br(brnet, task_data)
    print(br_cost)
    br_grad = leader.grad_obj_br(brnet, task_data)
    print(br_grad.get_grad())

def test4():
    brnet = BRNet()
    x, a = torch.ones(4), torch.ones(2)
    gx, ga = brnet.compute_grad(x, a)
    print(gx, ga)

def test5():
    brnet = BRNet()
    leader = Leader()
    leader.read_task_info(0)
    leader.solve_oc1(brnet)


def list2array(data):
    N = len(data)
    x = np.zeros((N, 4+2+2))
    for i in range(N):
        di = data[i]
        x[i, :] = np.concatenate((di[0], di[1], di[2]))
    return x

def array2list(x):
    N = x.shape[0]
    data = []
    for i in range(N):
        di = [ x[i, 0:4], x[i, 4:6], x[i, 6:] ]
        data.append(di)
    return data
        

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
    #test()
    #test2()
    #test3()
    #test4()
    test5()