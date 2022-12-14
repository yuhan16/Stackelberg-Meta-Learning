"""
This script implements different plot functions using stored data in the data/ directory.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import param
from utils_meta import BRNet, Leader, Follower, Meta


def plot_env(ax, scn, theta):
    """
    This function plots the environment settings. Use it as a block for other plot functions.
    """
    obs, _, _, pf, _, _, _ = param.get_param(scn, theta) 
    obs_num = len(obs)
    obs = np.array(obs)

    # plot destination
    ax.plot(pf[0], pf[1], '*', markersize=20, color='y')

    # plot obstacles
    for j in range(obs_num):
        c = plt.Circle((obs[j][0], obs[j][1]), obs[j][2], ec='k', fc='0.9', fill=True, ls='--', lw=2)
        ax.add_patch(c)
        #ax.add_artist(c)

    # check if plot directory exists
    dir_name = 'data/plots/'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    return ax


def plot_empty_env():
    """
    This function plots the empty environment with robot starting positions for a specific scenario.
    """
    scn = 2
    leader = Leader()
    TOTAL_TYPE = param.total_type
    fig, ax = plt.subplots()
    ax = plot_env(ax, scn, 0)
    ax.set_xlim((-0.5,10.5))
    ax.set_ylim((-0.5,10.5))
    ax.set_aspect(1)

    theta = 0
    leader.read_task_info(scn, theta)
    x0A, x0B = leader.x0[0: leader.dimxA], leader.x0[leader.dimxA: ]
    ax.plot(x0A[0], x0A[1], 'bo', label='type θ=0')
    ax.plot(x0B[0], x0B[1], 'b^')

    theta = 1
    leader.read_task_info(scn, theta)
    x0A, x0B = leader.x0[0: leader.dimxA], leader.x0[leader.dimxA: ]
    ax.plot(x0A[0], x0A[1], 'ro', label='type θ=1')
    ax.plot(x0B[0], x0B[1], 'r^')

    theta = 2
    leader.read_task_info(scn, theta)
    x0A, x0B = leader.x0[0: leader.dimxA], leader.x0[leader.dimxA: ]
    ax.plot(x0A[0], x0A[1], 'go', label='type θ=2')
    ax.plot(x0B[0], x0B[1], 'g^')

    theta = 3
    leader.read_task_info(scn, theta)
    x0A, x0B = leader.x0[0: leader.dimxA], leader.x0[leader.dimxA: ]
    ax.plot(x0A[0], x0A[1], 'yo', label='type θ=3')
    ax.plot(x0B[0], x0B[1], 'y^')

    ax.legend(fontsize='large')
    fig.savefig('data/plots/fig_empty_env.png')


def plot_no_guide():
    """
    This function plots no-guidance vs meta-learning result for a specific scenario and task.
    """
    scn, theta = 0, 2
    # get no guidance trajectory
    x_noguide = np.load('data/data_noguide/scenario' + str(scn) + '/task' + str(theta) + '_xtraj.npy')

    # get meta-learning trajectory
    xx = np.load('data/data_meta/scenario' + str(scn) + '/x_traj_adapt.npy', allow_pickle=True).flat[0]
    aa = np.load('data/data_meta/scenario' + str(scn) + '/a_traj_adapt.npy', allow_pickle=True).flat[0]
    key = 't' + str(theta)
    x_traj, a_traj = xx[key], aa[key]
    ff = Follower(scn, theta)
    x_meta, _ = ff.get_interactive_traj(x_traj[0,:], a_traj)

    fig, ax = plt.subplots()
    #x.set_xlim((0,8))
    #ax.set_ylim((1,9))
    ax.set_aspect(1)
    ax.set_box_aspect(1)
    ax = plot_env(ax, scn, theta)

    ax.plot(x_meta[:, 2], x_meta[:, 3], 'r^', label='meta-learning')    # only plot follower's trajectory
    ax.plot(x_noguide[:, 2], x_noguide[:, 3], 'gs', label='no guide')   # only plot follower's trajectory
    ax.legend(loc=4, fontsize='x-large')

    fig.savefig('data/plots/fig_meta_noguide.png')
    plt.close(fig)


def plot_nometa_scn0():
    scn, theta = 0, 0
    ff = Follower(scn, theta)
    xx = np.load('data/data_nometa/scenario' + str(scn) + '/x_traj_nometa.npy', allow_pickle=True).flat[0]
    aa = np.load('data/data_nometa/scenario' + str(scn) + '/a_traj_nometa.npy', allow_pickle=True).flat[0]
    key = 't' + str(theta)

    x_traj, a_traj = xx[key], aa[key]
    x_gd, _ = ff.get_interactive_traj(x_traj[0, :], a_traj)

    fig, ax = plt.subplots()
    ax.set_xlim((0,10))
    ax.set_ylim((0,10))
    ax.set_aspect(1)
    ax = plot_env(ax, scn, theta)
    ax.plot(x_gd[:,0], x_gd[:,1], 'b^', markersize=5)   # leader's position trajectory
    ax.plot(x_gd[:,2], x_gd[:,3], 'ro', markersize=5)   # follower's position trajectory

    fig.savefig('data/plots/fig_nometa_scn0task0.png')
    plt.close(fig)


def plot_meta_scn0():
    scn = 0
    # get meta trajectory
    xx = np.load('data/data_meta/scenario' + str(scn) + '/x_traj_adapt.npy', allow_pickle=True).flat[0]
    aa = np.load('data/data_meta/scenario' + str(scn) + '/a_traj_adapt.npy', allow_pickle=True).flat[0]

    key = 't' + str(0)
    x_traj, a_traj = xx[key], aa[key]
    ff = Follower(scn, 0)
    x_gd, b_gd = ff.get_interactive_traj(x_traj[0,:], a_traj)
    x_t0 = x_gd

    key = 't' + str(1)
    x_traj, a_traj = xx[key], aa[key]
    ff = Follower(scn, 1)
    x_gd, b_gd = ff.get_interactive_traj(x_traj[0,:], a_traj)
    x_t1 = x_gd

    key = 't' + str(2)
    x_traj, a_traj = xx[key], aa[key]
    ff = Follower(scn, 2)
    x_gd, b_gd = ff.get_interactive_traj(x_traj[0,:], a_traj)
    x_t2 = x_gd

    key = 't' + str(3)
    x_traj, a_traj = xx[key], aa[key]
    ff = Follower(scn, 3)
    x_gd, b_gd = ff.get_interactive_traj(x_traj[0,:], a_traj)
    x_t3 = x_gd

    fig, ax = plt.subplots()
    ax.set_xlim((0,10))
    ax.set_ylim((0,10))
    ax.set_aspect(1)
    ax.set_box_aspect(1)

    ax.plot(x_t0[:, 2], x_t0[:, 3], 'b^', label='follower θ=0')
    ax.plot(x_t1[:, 2], x_t1[:, 3], 'r^', label='follower θ=1')
    ax.plot(x_t2[:, 2], x_t2[:, 3], 'g^', label='follower θ=2')
    ax.plot(x_t3[:, 2], x_t3[:, 3], 'y^', label='follower θ=3')
    ax = plot_env(ax, scn, theta=0)
    ax.legend(loc=4, fontsize='x-large')
    fig.savefig('data/plots/fig_meta_scn0.png')
    plt.close(fig)


def plot_meta_scn1():
    scn = 1
    # get meta trajectory
    xx = np.load('data/data_meta/scenario' + str(scn) + '/x_traj_adapt.npy', allow_pickle=True).flat[0]
    aa = np.load('data/data_meta/scenario' + str(scn) + '/a_traj_adapt.npy', allow_pickle=True).flat[0]

    key = 't' + str(0)
    x_traj, a_traj = xx[key], aa[key]
    ff = Follower(scn, 0)
    x_gd, b_gd = ff.get_interactive_traj(x_traj[0,:], a_traj)
    x_t0 = x_gd

    key = 't' + str(1)
    x_traj, a_traj = xx[key], aa[key]
    ff = Follower(scn, 1)
    x_gd, b_gd = ff.get_interactive_traj(x_traj[0,:], a_traj)
    x_t1 = x_gd

    key = 't' + str(2)
    x_traj, a_traj = xx[key], aa[key]
    ff = Follower(scn, 2)
    x_gd, b_gd = ff.get_interactive_traj(x_traj[0,:], a_traj)
    x_t2 = x_gd

    key = 't' + str(3)
    x_traj, a_traj = xx[key], aa[key]
    ff = Follower(scn, 3)
    x_gd, b_gd = ff.get_interactive_traj(x_traj[0,:], a_traj)
    x_t3 = x_gd

    fig, ax = plt.subplots()
    ax.set_xlim((0,10))
    ax.set_ylim((0,10))
    ax.set_aspect(1)
    ax.set_box_aspect(1)

    ax.plot(x_t0[:, 2], x_t0[:, 3], 'b^', label='follower θ=0')
    ax.plot(x_t1[:, 2], x_t1[:, 3], 'r^', label='follower θ=1')
    ax.plot(x_t2[:, 2], x_t2[:, 3], 'g^', label='follower θ=2')
    ax.plot(x_t3[:10, 2], x_t3[:10, 3], 'y^', label='follower θ=3')     # no need to plot all trajectory because get stuck
    ax = plot_env(ax, scn, theta=0)
    ax.legend(fontsize='x-large')
    fig.savefig('data/plots/fig_meta_scn1.png')
    plt.close(fig)


def plot_meta_scn2():
    scn = 2
    # get meta trajectory
    xx = np.load('data/data_meta/scenario' + str(scn) + '/x_traj_adapt.npy', allow_pickle=True).flat[0]
    aa = np.load('data/data_meta/scenario' + str(scn) + '/a_traj_adapt.npy', allow_pickle=True).flat[0]

    key = 't' + str(0)
    x_traj, a_traj = xx[key], aa[key]
    ff = Follower(scn, 0)
    x_gd, b_gd = ff.get_interactive_traj(x_traj[0,:], a_traj)
    x_t0 = x_gd

    key = 't' + str(1)
    x_traj, a_traj = xx[key], aa[key]
    ff = Follower(scn, 1)
    x_gd, b_gd = ff.get_interactive_traj(x_traj[0,:], a_traj)
    x_t1 = x_gd

    key = 't' + str(2)
    x_traj, a_traj = xx[key], aa[key]
    ff = Follower(scn, 2)
    x_gd, b_gd = ff.get_interactive_traj(x_traj[0,:], a_traj)
    x_t2 = x_gd

    key = 't' + str(3)
    x_traj, a_traj = xx[key], aa[key]
    ff = Follower(scn, 3)
    x_gd, b_gd = ff.get_interactive_traj(x_traj[0,:], a_traj)
    x_t3 = x_gd

    fig, ax = plt.subplots()
    ax.set_xlim((0,10))
    ax.set_ylim((0,10))
    ax.set_aspect(1)
    ax.set_box_aspect(1)

    ax.plot(x_t0[:9, 2], x_t0[:9, 3], 'b^', label='follower θ=0')       # no need to plot all trajectory because get stuck
    ax.plot(x_t1[:, 2], x_t1[:, 3], 'r^', label='follower θ=1')
    ax.plot(x_t2[:, 2], x_t2[:, 3], 'g^', label='follower θ=2')
    ax.plot(x_t3[:, 2], x_t3[:, 3], 'y^', label='follower θ=3')
    ax = plot_env(ax, scn, theta=0)
    ax.legend(fontsize='x-large')
    fig.savefig('data/plots/fig_meta_scn1.png')
    plt.close(fig)


def plot_meta_detail():
    scn, theta = 1, 0
    xx = np.load('data/data_meta/scenario' + str(scn) + '/x_traj_adapt.npy', allow_pickle=True).flat[0]
    aa = np.load('data/data_meta/scenario' + str(scn) + '/a_traj_adapt.npy', allow_pickle=True).flat[0]
    key = 't' + str(theta)
    x_traj, a_traj = xx[key], aa[key]
    ff = Follower(scn, theta)
    x_gd, b_gd = ff.get_interactive_traj(x_traj[0,:], a_traj)
    x_t0 = x_gd

    fig, ax = plt.subplots()
    ax.set_xlim((0,10))
    ax.set_ylim((0,10))
    ax.set_aspect(1)
    ax.plot(x_t0[:, 0], x_t0[:, 1], 'bo', label='leader')
    ax.plot(x_t0[:, 2], x_t0[:, 3], 'r^', label='follower')
    ax.plot(x_traj[:, 2], x_traj[:, 3], 'gx', label='leader\'s conjecture')
    ax = plot_env(ax, scn, theta)
    ax.legend(fontsize='large')
    fig.savefig('fig_meta_detail_1.png')
    plt.close(fig)


    scn, theta = 2, 0
    xx = np.load('data_meta/scenario' + str(scn) + '/x_traj_adapt.npy', allow_pickle=True).flat[0]
    aa = np.load('data_meta/scenario' + str(scn) + '/a_traj_adapt.npy', allow_pickle=True).flat[0]
    key = 't' + str(theta)
    x_traj, a_traj = xx[key], aa[key]
    ff = Follower(scn, theta)
    x_gd, b_gd = ff.get_interactive_traj(x_traj[0,:], a_traj)
    x_t0 = x_gd

    fig, ax = plt.subplots()
    ax.set_xlim((0,10))
    ax.set_ylim((0,10))
    ax.set_aspect(1)
    ax.plot(x_t0[:, 0], x_t0[:, 1], 'bo', label='leader')
    ax.plot(x_t0[:9, 2], x_t0[:9, 3], 'r^', label='follower')
    ax.plot(x_traj[:, 2], x_traj[:, 3], 'gx', label='leader\'s conjecture')
    ax = plot_env(ax, scn, theta)
    ax.legend(fontsize='large', loc=4)
    fig.savefig('data/plots/fig_meta_detail_2.png')
    plt.close(fig)


def plot_animation():
    """
    This function plots a series of figures to generate gif for animation.
    """
    dir_name = 'tmp_animation/'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    scn, theta = 2, 2
    xx = np.load('data/data_meta/scenario' + str(scn) + '/x_traj_adapt.npy', allow_pickle=True).flat[0]
    aa = np.load('data/data_meta/scenario' + str(scn) + '/a_traj_adapt.npy', allow_pickle=True).flat[0]

    key = 't' + str(scn)
    x_traj, a_traj = xx[key], aa[key]
    ff = Follower(scn, theta)
    x_gd, b_gd = ff.get_ground_truth(x_traj[0,:], a_traj)

    for i in range(1, x_traj.shape[0]+1):
        fig, ax = plt.subplots()
        ax.set_xlim((-0.5,10.5))
        ax.set_ylim((-0.5,10.5))
        ax.set_aspect(1)
        ax = plot_env(ax, scn, theta)
        ax.plot(x_gd[0:i, 0], x_gd[0:i, 1], 'bo', label='leader')   # leader's trajectory
        ax.plot(x_gd[0:i, 2], x_gd[0:i, 3], 'r^', label='follower')
        ax.legend(loc=4, fontsize='large')
        
        fname = dir_name + 'animation_s' + str(scn) + 't' + str(theta) + '_' + str(i) + '.png' 
        fig.savefig(fname)
        plt.close(fig)


if __name__ == '__main__':
    plot_empty_env()    # plot empty environment with initial positions for all robots. (single scenario)
    
    plot_meta_scn0()    # plot adaptation result for all robots (scenario 0).
    plot_meta_scn1()    # plot adaptation result for all robots (scenario 1).
    plot_meta_scn2()    # plot adaptation result for all robots (scenario 2).
    plot_meta_detail()  # plot two detailed adaptation result (scenario 1 and scenario 2 task 0)

    plot_no_guide()     # plot no-guidance control result for a specific scenario and task.
    plot_nometa_scn0()  # plot individual learning for scenario 0.
    plot_animation()    # plot series of figures to generate gif. (single scenario single task)
    