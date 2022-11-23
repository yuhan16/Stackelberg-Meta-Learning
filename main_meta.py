"""
This is the main script for testing and main exectuation.
"""

import torch
import os, sys, time
import numpy as np
import param
from utils_meta import BRNet, Leader, Follower, Meta


def generate_random_br_data():
    """
    This function generates randomly sampled BR data for all scenarios.
    """
    # generate data directory
    if not os.path.exists('data/'):
        os.mkdir('data')
        os.mkdir('data/br_data/')
        os.mkdir('data/data_meta/')
        os.mkdir('data/data_noguide/')
        os.mkdir('data/data_nometa/')

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


def nometa(scn, theta):     # componenet of individual_learn()
    """
    This function implements non-meta learning. Online update brnet and then planning.
    """
    leader = Leader()
    meta = Meta()
    brnet = BRNet()
    leader.read_task_info(scn, theta)
    ff = Follower(scn, theta)
    key = 't' + str(theta)

    # load initial trajectory
    dir_name = 'data/data_nometa/scenario' + str(scn) + '/'
    x_init_dict = np.load(dir_name + 'x_traj_nometa.npy', allow_pickle=True).flat[0]
    a_init_dict = np.load(dir_name + 'a_traj_nometa.npy', allow_pickle=True).flat[0]

    ITER_MAX = 30       # non-meta-training steps
    xx = []
    for iter in range(ITER_MAX):
        x_init, a_init = x_init_dict[key], a_init_dict[key]     # x_init, a_init can also be random in each meta-training
        x_traj, a_traj = leader.solve_oc(brnet, x_init, a_init)
        x_init_dict[key], a_init_dict[key] = x_traj.copy(), a_traj.copy()     # update initial trajectory dictionary
        
        # some checks
        x_gd, b_gd = ff.get_interactive_traj(x_traj[0,:], a_traj)
        print('iter {},  guess = {:.3f}, real = {:.3f}'.format(iter, leader.obj_oc(x_traj, a_traj), leader.obj_oc(x_gd, a_traj)) )    
        #aux.check_constraint_violation(brnet, x_traj, a_traj)
        #aux.check_safety(leader, x_traj)

        # update brnet
        #new_data = meta.sample_task_theta_traj(ff, x_traj, a_traj, N=50)    # use trajectory data
        #new_data = task_data                                                # use uniform data
        new_data = meta.sample_task_theta(ff, x_traj, a_traj, N=50)         # use mix data
        brnet = meta.train_brnet(brnet, new_data, N=50)

        #pltutil.plot_trajectory(x_traj, real_pB=x_gd[:, 2:4], sim_pB=None)
        #xx = np.hstack( (x_traj[:, :leader.dimxA], x_gd[:, leader.dimxA:]) )

    # save learning results
    np.save(dir_name + 'x_traj_nometa.npy', x_init_dict)
    np.save(dir_name + 'a_traj_nometa.npy', a_init_dict)
    torch.save(brnet, dir_name + 'brnet_nometa_task' + str(theta) + '.pth')


def individual_learn():
    if len(sys.argv) != 2:
        raise Exception("Wrong number of arguments. Correct usage: python ./main_meta <scenario index>")
    else:
        if not sys.argv[1].isdigit() or int(sys.argv[1]) > 2 or int(sys.argv[1]) < 0:
            raise Exception("Scenario index should be an ineteger in \{0,1,2\}.")
        else:
            scn = int(sys.argv[1])      # scenario index
    #scn = 0

    x_init_dict, a_init_dict = {}, {}   # record initial trajectory in each learning iteration
    x_init_dict['t0'], x_init_dict['t1'], x_init_dict['t2'], x_init_dict['t3'] = None, None, None, None
    a_init_dict['t0'], a_init_dict['t1'], a_init_dict['t2'], a_init_dict['t3'] = None, None, None, None
    dir_name = 'data/data_nometa/scenario' + str(scn) + '/'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    np.save(dir_name + '/x_traj_nometa.npy', x_init_dict)
    np.save(dir_name + '/a_traj_nometa.npy', a_init_dict)
    
    print('Individual for scenario:', scn)
    for theta in range(param.total_type):
        print('\nAdapting task:', theta)
        start_t = time.time()
        nometa(scn, theta)
        end_t = time.time()
        print('meta ADAPTATION for secnario {} task {}, time: {} s.'.format(scn, theta, end_t-start_t))


def meta_training(scn):     # componenet of sg_meta_learn()
    leader = Leader()
    meta = Meta()
    brnet = BRNet()

    x_init_dict, a_init_dict = {}, {}               # record initial trajectory in each meta-training iteration
    x_init_dict['t0'], x_init_dict['t1'], x_init_dict['t2'], x_init_dict['t3'] = None, None, None, None
    a_init_dict['t0'], a_init_dict['t1'], a_init_dict['t2'], a_init_dict['t3'] = None, None, None, None
    a_traj_list, br_list, D2_list = [], [], []
    iter, ITER_MAX = 0, 50          # meta training iterations
    while iter <= ITER_MAX:
        print('Meta Iteration: {}'.format(iter))
        task_sample = meta.sample_tasks(5)
        #task_sample = [2, 3]

        for i in range(len(task_sample)):
            theta = task_sample[i]
            leader.read_task_info(scn, theta)
            follower_theta = Follower(scn, theta)
            key = 't' + str(theta)

            # sample D1_theta and update model
            x_init, a_init = x_init_dict[key], a_init_dict[key]     # x_init, a_init can also be random in each meta-training
            x_traj, a_traj = leader.solve_oc(brnet, x_init, a_init)
            D1_theta = meta.sample_task_theta(follower_theta, x_traj, a_traj, N=50)
            br_list.append( meta.update_model_theta(leader, brnet, D1_theta, a_traj) )
            """br_list has the same (W,b) as brnet but updated gradient (dW,db) for different tasks."""
            x_init, a_init = x_traj.copy(), a_traj.copy()           # update initial trajectory
            
            # sample D2_theta for future update
            x_traj, a_traj = leader.solve_oc(brnet, x_init, a_init)
            x_init, a_init = x_traj.copy(), a_traj.copy()           # update initial trajectory
            D2_list.append( meta.sample_task_theta(follower_theta, x_traj, a_traj, N=50) )
            a_traj_list.append( a_traj )
            x_init_dict[key], a_init_dict[key] = x_init, a_init     # update initial trajectory dictionary
        
        # update the entire BR model
        brnet = meta.update_model(leader, brnet, task_sample, br_list, D2_list, a_traj_list)
        iter += 1
        
        # clear intermidiate brnets and D2
        br_list.clear()
        D2_list.clear()
        a_traj_list.clear()
    
    # save meta training results
    dir_name = 'data/data_meta/scenario' + str(scn) + '/'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    fname = dir_name + 'brnet_meta_scn' + str(scn) + '.pth'
    torch.save(brnet, fname)

    # save meta trained trajectories as initial trajectories for future adaptation.
    np.save(dir_name + 'x_traj_meta.npy', x_init_dict)
    np.save(dir_name + 'a_traj_meta.npy', a_init_dict)


def meta_adapt(scn, theta): # componenet of individual_learn()
    leader = Leader()
    meta = Meta()
    brnet = BRNet()

    leader.read_task_info(scn, theta)
    ff = Follower(scn, theta)
    key = 't' + str(theta)

    # load data from meta training
    dir_name = 'data/data_meta/scenario' + str(scn) + '/'
    fname = dir_name + 'brnet_meta_scn' + str(scn) + '.pth'
    if os.path.exists(fname):
        brnet = torch.load(fname)
    else:
        raise Exception("No meta brnet model found. Run meta training first.")
    
    fname = dir_name + 'x_traj_meta.npy'
    if os.path.exists(fname):
        x_init_dict = np.load(dir_name + 'x_traj_meta.npy', allow_pickle=True).flat[0]
        a_init_dict = np.load(dir_name + 'a_traj_meta.npy', allow_pickle=True).flat[0]
    else:
        raise Exception("No initial trajectory dictionary found. Run meta training first.")
    
    ITER_MAX = 15       # adaptation iteration
    for iter in range(ITER_MAX):
        x_init, a_init = x_init_dict[key], a_init_dict[key]
        x_traj, a_traj = leader.solve(brnet, x_init, a_init)
        D1_theta = meta.sample_task_theta(ff, x_traj, a_traj, N=50)
        brnet_new = meta.update_model_theta(leader, brnet, D1_theta, a_traj)

        brnet.load_state_dict(brnet_new.state_dict())                   # copy adapted brnet parameter
        x_init_dict[key], a_init_dict = x_traj.copy(), a_traj.copy()    # update adapted initial trajectory
        #del brnet_new
        
        # some checks
        x_gd, b_gd = ff.get_ground_truth(x_traj[0,:], a_traj)
        print('iter {},  guess = {:.3f}, real = {:.3f}'.format(iter, leader.obj_oc(x_traj, a_traj), leader.obj_oc(x_gd, a_traj)) )    
        #aux.check_constraint_violation(brnet, x_traj, a_traj)
    
    # save meta adaptation results
    dir_name = 'data/data_meta/scenario' + str(scn) + '/'
    fname = dir_name + 'brnet_adapt_task' + str(theta) + '.pth'
    torch.save(brnet, fname)

    # save adapted trajectories for interactive guidance. Save |total_type| times, overwrite previous file.
    np.save(dir_name + 'x_traj_adapt.npy', x_init_dict)
    np.save(dir_name + 'a_traj_adapt.npy', a_init_dict)


def sg_meta_learn():
    if len(sys.argv) != 2:
        raise Exception("Wrong number of arguments. Correct usage: python ./main_meta <scenario index>")
    else:
        if not sys.argv[1].isdigit() or int(sys.argv[1]) > 2 or int(sys.argv[1]) < 0:
            raise Exception("Scenario index should be an ineteger in \{0,1,2\}.")
        else:
            scn = int(sys.argv[1])      # scenario index
    #scn = 0
    
    print('Meta-training for scenario:', scn)
    start_t = time.time()
    meta_training(scn)         # perform meta-training
    end_t = time.time()
    print('meta TRAINING for secnario {} finished, time: {} s.\n'.format(scn, end_t-start_t))
    
    for theta in range(param.total_type):
        print('\nAdapting task:', theta)
        start_t = time.time()
        meta_adapt(scn, theta)
        end_t = time.time()
        print('meta ADAPTATION for secnario {} task {}, time: {} s.'.format(scn, theta, end_t-start_t))


if __name__ == '__main__':
    """
    Before run any functions, first run generate_random_br_data() to obtain and store randomly sampled BR data.
    """
    #generate_random_br_data()       # pre-sample BR data for future use.
    
    """
    Uncomment sg_meta_learn() to run SG meta-learning.
    """
    sg_meta_learn()                # implement Stackelberg meta-learning and adaptation for a specific scenario
    #meta_training(0)               # run meta training for scenario 0.
    #meta_adapt(0, 1)               # run adaptation for scenario 0 task 1.

    """
    Uncomment individual_learn() to run individual learning.
    """
    #individual_learn()              # implement individual learning (a non-meta-learning approach) for a specific scenario
    #nometa(0, 1)                    # run individual learning (a non-meta-learning approach) for scenario 0 task 1

    """
    Uncomment no_guide() to run no guidance control.
    """    
    #no_guide()                      # implement no-guidance control for the follower. (all scenarios)
