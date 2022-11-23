"""
This scripts performs miscellaneous tests. 
Encapusulate each test into a test function and run the test.
"""

import torch
import numpy as np
from utils_meta import BRNet, Leader, Follower, Meta, Auxiliary
from os.path import exists


def test1():     # basic function test
    f1 = Follower(0, 0)
    x = np.array([3,3, 2.2,2.2,0.])
    a = np.array([0.1, 0.1])
    b_opt = f1.get_br(x, a)
    #print(b_opt)
    #print(f1.compute_obj(x, a, b_opt))
    
    x = np.array([ 9.68049253,  8.19268296,  0.40908305,  4.6376705 , -0.14172506])
    a = np.array([-0.51520282, -0.50625648])
    b_opt = f1.get_br(x, a)
    x = np.array([6.16562293, 3.6280085 , 6.08480784, 9.70409036, 0.38566065])
    a = np.array([-0.05890506,  0.25706389])
    b_opt = f1.get_br(x, a)
    x = np.array([ 6.49518801,  8.60204918,  7.06759417,  4.3650341 , -0.77067148])
    a = np.array([-0.7588574 , -0.65125681])
    b_opt = f1.get_br(x, a)

    meta = Meta()
    print(meta.sample_tasks(100))
    d1 = meta.sample_task_theta_uniform(f1, 10)
    print(len(d1))

    brnet = BRNet()
    #print(brnet(torch.tensor([[1.,2.,3.,4.,5.,6.], [1.,2.,3.,4.,5.,6.]])))
    x, a = torch.ones(5), torch.ones(2)
    gx, ga = brnet.compute_input_jac(x, a)
    print(gx, ga)

def test2():     # leader's cost function and OC computation
    scn, theta = 0, 0
    meta = Meta()
    brnet = BRNet()
    leader = Leader()
    leader.read_task_info(scn, theta)

    x_traj, a_traj = leader.solve_oc(brnet)
    oc_cost = leader.obj_oc(x_traj, a_traj)
    print(oc_cost)

    follower = Follower(scn, theta)
    task_data = meta.sample_task_theta(follower, x_traj, a_traj, N=100)
    br_cost = leader.obj_br(brnet, task_data)
    print(br_cost)

    oc_grad = leader.grad_obj_oc(brnet, a_traj)
    print(oc_grad.get_data())
    br_grad = leader.grad_obj_br(brnet, task_data)
    print(br_grad.get_grad())

    b2 = meta.update_model_theta(leader, brnet, task_data, a_traj)
    np.save('taskdata.npy', task_data)
    np.save('xtraj.npy', x_traj)
    np.save('atraj.npy', a_traj)

def test3():
    meta = Meta()
    brnet = BRNet()
    leader = Leader()
    leader.read_task_info(0)

    task_data = np.load('taskdata.npy')
    x_traj = np.load('xtraj.npy')
    a_traj = np.load('atraj.npy')

    oc_cost = leader.obj_oc(x_traj, a_traj)
    print(oc_cost)
    br_cost = leader.obj_br(brnet, task_data)
    print(br_cost)
    br_grad = leader.grad_obj_br(brnet, task_data)
    print(br_grad.get_grad())

def test4():    # brnet test
    brnet1 = BRNet()
    x1, a1 = torch.tensor([1,0,0,1,0.]), torch.tensor([1,1.])
    x2, a2 = torch.tensor([0,1,0,1,1.]), torch.tensor([0,1.])
    b1 = torch.tensor([0,-1.])
    y1 = brnet1(x1, a1)
    cost_fn = torch.nn.MSELoss(reduction='sum')
    cost1 = cost_fn(y1, b1)
    brnet1.zero_grad()
    cost1.backward()
    # monitor output of each layer
    y = []  # y[i] is a 2d array
    def forward_hook(model, input, output):
        y.append( output.detach() )
    h1 = brnet1.linear1.register_forward_hook(forward_hook)
    h2 = brnet1.linear2.register_forward_hook(forward_hook)
    h3 = brnet1.linear3.register_forward_hook(forward_hook)
    #h4 = brnet1.linear4.register_forward_hook(forward_hook)
    _ = brnet1.forward(x1, a1)
    h1.remove()
    h2.remove()
    h3.remove()
    #h4.remove()
    print(y)
    p1 = brnet1.get_data_dict(); dp1 = brnet1.get_grad_dict()
    for n, _ in brnet1.named_parameters():
        print(n, p1[n])
        print(dp1[n])

    # test stack
    brnet = BRNet()
    x, a = torch.stack((x1, x2, x1)), torch.stack((a1, a2, a1))
    y = brnet(x, a)
    b = torch.stack((b1,2*b1, 3*b1))
    cost = cost_fn(y, b)
    brnet.zero_grad()
    cost.backward()
    p = brnet.get_data_dict(); dp = brnet.get_grad_dict()
    for n, _ in brnet.named_parameters():
        print(n,p[n])
        print(dp[n])

def test5():    # sampling test
    scn, theta = 0, 0
    leader = Leader()
    meta = Meta()
    ff = Follower(scn, theta)
    leader.read_task_info(scn, theta)
    task_data = meta.sample_task_theta_uniform(ff, N=10)
    print(task_data)

    x_traj = np.random.rand(leader.dimT+1, leader.dimxA+leader.dimxB)
    a_traj = np.random.rand(leader.dimT, leader.dima)
    task_data = meta.sample_task_theta(ff, x_traj, a_traj, N=10)
    print(task_data)
    

def test6():    # opt solver test for single agent 0/1
    aux = Auxiliary()
    meta = Meta()
    brnet = BRNet()
    leader = Leader()
    leader.read_task_info(1)

    task_data = np.load('data/task1_uniform.npy')
    if exists('data/brnet1_init.pth'):
        brnet = torch.load('data/brnet1_init.pth')
    else:
        brnet = meta.train_brnet(brnet, task_data, N=1000)
        torch.save(brnet, 'data/brnet1_init.pth')
    x_traj, a_traj = leader.solve_oc(brnet)
    aux.plot_trajectory(x_traj)
    #f1 = Follower(1)
    #x_gd, b_gd = f1.get_interactive_traj(x_traj[0,:], a_traj)
    #aux.plot_trajectory(x_traj, real_pB=x_gd[:, 2:4])
    #print(leader.obj_oc(x_gd, a_traj))
    
    dif = aux.check_dynamics(x_traj, a_traj, brnet)
    print( np.abs(np.max(dif)) )
    print( leader.get_b_traj(brnet, x_traj, a_traj) )

def test7():    # opt solver test for single agent, multiple times
    aux = Auxiliary()
    meta = Meta()
    brnet = BRNet()
    leader = Leader()
    leader.read_task_info(1)
    f1 = Follower(0, 1)

    task_data = np.load('data/task1_uniform.npy')
    if exists('data/brnet1_init.pth'):
        brnet = torch.load('data/brnet1_init.pth')
    else:
        brnet = meta.train_brnet(brnet, task_data, N=50)
        torch.save(brnet, 'data/brnet1_init.pth')

    for iter in range(10):
        x_traj, a_traj = leader.solve_oc(brnet)
        aux.plot_trajectory(x_traj, real_pB=None)
        x_gd, b_gd = f1.get_interactive_traj(x_traj[0,:], a_traj)
        aux.plot_trajectory(x_traj, real_pB=x_gd[:, 2:4])
        print('iter {}, guess = {:.3f}, real = {:.3f}'.format(iter, leader.obj_oc(x_traj, a_traj), leader.obj_oc(x_gd, a_traj)) )

        # update brnet
        traj_data = meta.list2array( meta.sample_task_theta_traj(f1, x_traj, a_traj, N=100) )
        brnet = meta.train_brnet(brnet, traj_data)
        tmp = 1
    
def test8():    # test solve_oc1 function, test content is the same as test7()
    aux = Auxiliary()
    meta = Meta()
    brnet = BRNet()
    leader = Leader()
    leader.read_task_info(1)   # theta = -1 for testing
    ff = Follower(0, theta=1)

    fname = 'data/task_test_uniform.npy'
    fname = 'data/task1_uniform.npy' 
    if exists(fname):
        task_data = np.load(fname)
        task_data = meta.array2list(task_data)
    else:
        task_data = meta.sample_task_theta_uniform(ff, N=10000)
        np.save(fname, meta.list2array(task_data))
    fname = 'data/brnet_meta_test_s0v2.pth'
    if exists(fname):
        brnet = torch.load(fname)
    else:
        brnet = meta.train_brnet(brnet, task_data, N=100)
        torch.save(brnet, 'data/brnet_test_init.pth')

    x_traj_list, a_traj_list = [], []
    for iter in range(10):
        x_traj_list, a_traj_list = leader.solve_oc1(brnet, x_traj_list, a_traj_list, traj_num=1)
        
        # find the optimal solution from candidate trajectories.
        cost_list = []
        for i in range(len(x_traj_list)):
            cost_list.append( leader.obj_oc(x_traj_list[i], a_traj_list[i]) )
        idx = np.argmin(np.array(cost_list))
        x_traj_opt, a_traj_opt = x_traj_list[idx], a_traj_list[idx]
        # cost_opt = cost_list[idx]
        print('Optimal trjaectory: %d\n' %(idx))

        # plot things
        for i in range(len(x_traj_list)):
            x_gd, b_gd = ff.get_interactive_traj(x_traj_list[i][0,:], a_traj_list[i])
            x_sm, b_sm = leader.shooting_simulate_traj(brnet, a_traj_list[i])
            #pltutil.plot_trajectory(x_traj_list[i], real_pB=x_gd[:, 2:4], sim_pB=None)

            print('traj {}: iter {},  guess = {:.3f}, real = {:.3f}'.format(i, iter, leader.obj_oc(x_traj_list[i], a_traj_list[i]), leader.obj_oc(x_gd, a_traj_list[i])) )    
            aux.check_input_constraint(brnet, x_traj_list[i], a_traj_list[i])
            aux.check_dynamics_constraint(brnet, x_traj_list[i], a_traj_list[i])
            tmp = 1

        # update brnet
        #new_data = meta.sample_task_theta_traj(ff, x_traj_opt, a_traj_opt, N=100)   # use trajectory data
        #new_data = task_data                                                        # use uniform data
        new_data = meta.sample_task_theta(ff, x_traj_opt, a_traj_opt, N=100)         # use mix data
        brnet = meta.train_brnet(brnet, new_data, N=50)
        tmp = 1

def test9():    # no guidance test
    scn, theta = 0, 0
    ff = Follower(scn, theta)
    x_traj, b_traj = ff.no_guidance()
    print(x_traj, b_traj)

        
def adaption(theta):    # for follower 1 for now
    aux = Auxiliary()
    meta = Meta()
    brnet = BRNet()
    leader = Leader()
    leader.read_task_info(1)   # theta = -1 for testing
    ff = Follower(0, theta=1)

    fname = 'data/task_test_uniform.npy'
    fname = 'data/task1_uniform.npy' 
    if exists(fname):
        task_data = np.load(fname)
        task_data = meta.array2list(task_data)
    else:
        task_data = meta.sample_task_theta_uniform(ff, N=10000)
        np.save(fname, meta.list2array(task_data))
    fname = 'data/brnet_meta_test_s0v2.pth'
    if exists(fname):
        brnet = torch.load(fname)
    else:
        brnet = meta.train_brnet(brnet, task_data, N=100)
        torch.save(brnet, 'data/brnet_test_init.pth')
    
    for iter in range(10):
        x_traj, a_traj = leader.solve_oc2(brnet)
        x_gd, b_gd = ff.get_interactive_traj(x_traj[0,:], a_traj)
        D1_theta = meta.sample_task_theta(ff, x_traj, a_traj, N=200)
        brnet_new = meta.update_model_theta(leader, brnet, D1_theta, a_traj)

        brnet.load_state_dict(brnet_new.state_dict())
        del brnet_new
        
        # plot things
        print('iter {},  guess = {:.3f}, real = {:.3f}'.format(iter, leader.obj_oc(x_traj, a_traj), leader.obj_oc(x_gd, a_traj)) )    
        aux.check_input_constraint(brnet, x_traj, a_traj)
        aux.check_dynamics_constraint(brnet, x_traj, a_traj)
        tmp = 1


def main():
    meta = Meta()
    brnet = BRNet()
    leader = Leader()
    a_traj_list, br_list, D2_list = [], [], []
    iter, ITER_MAX = 0, 100
    while iter <= ITER_MAX:
        print('Meta Iteration: {}'.format(iter))
        task_sample = meta.sample_tasks(5)
        #task_sample = [2, 3]
        for i in range(len(task_sample)):
            theta = task_sample[i]
            leader.read_task_info(theta)
            follower_theta = Follower(0, theta)

            # sample D1_theta and update model
            x_traj, a_traj = leader.solve_oc2(brnet)
            D1_theta = meta.sample_task_theta(follower_theta, x_traj, a_traj, N=200)
            br_list.append( meta.update_model_theta(leader, brnet, D1_theta, a_traj) ) 
            """br_list has the same (W,b) as brnet but updated gradient (dW,db) for different tasks."""

            # sample D2_theta for future update
            x_traj, a_traj = leader.solve_oc2(brnet)
            D2_list.append( meta.sample_task_theta(follower_theta, x_traj, a_traj, N=200) )
            a_traj_list.append( a_traj )
        
        # update the entire BR model
        brnet = meta.update_model(leader, brnet, task_sample, br_list, D2_list, a_traj_list) # ??? to be tested
        iter += 1
        
        # clear intermidiate brnets and D2
        br_list.clear()
        D2_list.clear()
        a_traj_list.clear()
    torch.save(brnet, 'data/brnet_meta_test_s0v2.pth')
    print('task sample:', task_sample)
    meta.print_info()



if __name__ == '__main__':
    #main()
    #adaption(1)
    #test1()    # basic funcion test
    test2()    # leader's cost function and OC computation
    #test3()
    #test4()     # brnet test
    #test5()
    #test6()    # opt and pmp solver test for single agent, single time.
    #test7()    # opt and pmp solver test for single agent, multiple time
    #test8()     # test solve_oc1, test content same as test7()
    #test9()     # no guidance