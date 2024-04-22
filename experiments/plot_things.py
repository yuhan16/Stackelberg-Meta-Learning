import os
import numpy as np
import matplotlib.pyplot as plt
from sg_meta.agent import Leader
from sg_meta.meta import Meta
from sg_meta.utils import PlotUtils
from sg_meta import train_data_dir, plot_data_dir 


clist = [   # color list
    (0.121,0.467,0.705),    # blue
    (0.992, 0.725, 0.419),  # orange
    (0.925, 0.364, 0.231),  # red
]


def plot_adapt_loss_bar():     
    '''
    plot final adapted loss for all followers
    '''
    leader = Leader()
    leader.meta = Meta()

    # load adaptation data
    param_ave, output_ave, adapt = [], [], []
    for theta in range(leader.meta.total_type):
        param_ave.append( np.load(os.path.join(train_data_dir, 'ave_param', f'adapt{theta}_loss.npy')) )
        output_ave.append( np.load(os.path.join(train_data_dir, 'ave_output', f'adapt{theta}_loss.npy')) )
        adapt.append( np.load(os.path.join(train_data_dir, 'meta_data', f'adapt{theta}_loss.npy')) )
    param_ave = np.vstack(param_ave)
    output_ave = np.vstack(output_ave)
    adapt = np.vstack(adapt)

    fig, ax = plt.subplots()
    x = ['θ=0', 'θ=1', 'θ=2', 'θ=3', 'θ=4']
    x_axis = np.arange(leader.meta.total_type)
    ax.bar(x_axis-0.2, adapt[:,-1], width=0.2, color=clist[0], alpha=0.8, label='Adaptation')
    ax.bar(x_axis, output_ave[:,-1], width=0.2, color=clist[1], alpha=0.8, label='Output-Ave')
    ax.bar(x_axis+0.2, param_ave[:,-1]/2, width=0.2, color=clist[2], alpha=0.8, label='Param-Ave (/2)')
    ax.set_xticks(x_axis)
    ax.set_xticklabels(x)
    ax.xaxis.set_tick_params(labelsize='large')
    ax.yaxis.set_tick_params(labelsize='large')
    ax.set_xlabel('Follower type', fontsize='xx-large')
    ax.set_ylabel('MSE Error', fontsize='xx-large')
    ax.legend(fontsize='x-large')

    fname = os.path.join(plot_data_dir, 'adapt_final.png')
    fig.savefig(fname, dpi=300)
    plt.close(fig)


def plot_adapt_curve_theta(theta):
    '''
    plot adaptation curve for type theta follower.
    '''
    leader = Leader()
    leader.meta = Meta()
    
    # load adaptation data
    param_ave, output_ave, adapt = [], [], []
    for th in range(leader.meta.total_type):
        param_ave.append( np.load(os.path.join(train_data_dir, 'ave_param', f'adapt{th}_loss.npy')) )
        output_ave.append( np.load(os.path.join(train_data_dir, 'ave_output', f'adapt{th}_loss.npy')) )
        adapt.append( np.load(os.path.join(train_data_dir, 'meta_data', f'adapt{th}_loss.npy')) )
    param_ave = np.vstack(param_ave)
    output_ave = np.vstack(output_ave)
    adapt = np.vstack(adapt)

    fig, ax = plt.subplots()
    ax.plot(adapt[theta,:], color=clist[0], alpha=0.8, label='Adaptation')
    ax.plot(output_ave[theta,:], color=clist[1], alpha=0.8, label='Output-Ave')
    ax.plot(param_ave[theta,:]/2, color=clist[2], alpha=0.8, label='Param-Ave (/2)')
    ax.xaxis.set_tick_params(labelsize='large')
    ax.yaxis.set_tick_params(labelsize='large')
    ax.set_xlabel('Iteration', fontsize='xx-large')
    ax.set_ylabel('MSE Error', fontsize='xx-large')
    ax.legend(fontsize='x-large')
    
    fname = os.path.join(plot_data_dir, f'adapt{theta}.png')
    fig.savefig(fname, dpi=300)
    plt.close(fig)


def plot_mpc_traj(theta):
    '''
    plot mpc trajectory for type theta follower
    '''
    leader = Leader()
    pltutil = PlotUtils()
    
    fig, ax = plt.subplots()
    ax = pltutil.plot_env(ax)

    # load trajectory data
    x = np.load( os.path.join(train_data_dir, 'rh_traj', f'x_type{theta}_[1. 8.].npy') )
    pa, pb = x[:, 0:leader.dimxa], x[:, leader.dimxa: leader.dimx-1]
    ax.plot(pa[:,0], pa[:,1], 'o-', markersize=6, linewidth=1, color=clist[0], alpha=0.5, markeredgecolor='none', label='leader')
    ax.plot(pb[:,0], pb[:,1], '^-', markersize=6, linewidth=1, color=clist[2], alpha=0.5, markeredgecolor='none', label='follower')

    x = np.load( os.path.join(train_data_dir, 'rh_traj', f'x_type{theta}_[5. 1.].npy') )
    pa, pb = x[:75, 0:leader.dimxa], x[:75, leader.dimxa: leader.dimx-1]
    ax.plot(pa[:,0], pa[:,1], 'o-', markersize=6, linewidth=2, color=clist[0], alpha=0.5)
    ax.plot(pb[:,0], pb[:,1], '^-', markersize=6, linewidth=2, color=clist[2], alpha=0.5)

    # x = np.load( os.path.join(train_data_dir, 'rh_traj', f'x_type{theta}_[0.  4.5].npy') )    # uncomment to plot other trajectory
    # pa, pb = x[:, 0:leader.dimxa], x[:, leader.dimxa: leader.dimx-1]
    # ax.plot(pa[:,0], pa[:,1], 'o-', markersize=6, linewidth=2, color=clist[0], alpha=0.5)
    # ax.plot(pb[:,0], pb[:,1], '^-', markersize=6, linewidth=2, color=clist[2], alpha=0.5)

    ax.xaxis.set_tick_params(labelsize='x-large')
    ax.yaxis.set_tick_params(labelsize='x-large')
    ax.legend(fontsize='x-large')
    fig.set_figwidth(4.8)

    fname = os.path.join(plot_data_dir, f'mpc{theta}.png')
    fig.savefig(fname)
    plt.close(fig)


def plot_mpc_traj_animation(theta):
    '''
    plot animations of mpc trajectory.
    '''
    if not os.path.exists( os.path.join(plot_data_dir, 'anim', 'mpc') ):
        os.makedirs(os.path.join(plot_data_dir, 'anim', 'mpc'))

    leader = Leader()
    pltutil = PlotUtils()
    
    # load trajectory data    
    fname = os.path.join(train_data_dir, 'rh_traj', f'x_type{theta}_[1. 8.].npy')
    #fname = os.path.join(train_data_dir, 'rh_traj', f'x_type{theta}_[5. 1.].npy')      # uncomment to plot other trajectory
    #fname = os.path.join(train_data_dir, 'rh_traj', f'x_type{theta}_[0.  4.5].npy')
    x = np.load(fname)
    pa, pb = x[:, 0:leader.dimxa], x[:, leader.dimxa: leader.dimx-1]

    for i in range(x.shape[0]):
        fig, ax = plt.subplots()
        ax = pltutil.plot_env(ax)

        ax.plot(pa[:i+1,0], pa[:i+1,1], 'o-', markersize=6, linewidth=1, color=clist[0], alpha=0.5, markeredgecolor='none', label='leader')
        ax.plot(pb[:i+1,0], pb[:i+1,1], '^-', markersize=6, linewidth=1, color=clist[2], alpha=0.5, markeredgecolor='none', label='follower')

        ax.xaxis.set_tick_params(labelsize='x-large')
        ax.yaxis.set_tick_params(labelsize='x-large')
        ax.legend(fontsize='x-large', loc='lower left')
        fig.set_figwidth(4.8)

        fname = os.path.join(plot_data_dir, 'anim', 'mpc', f'x_type{theta}_[1. 8.].png')
        #fname = os.path.join(plot_data_dir, 'anim', 'mpc', f'x_type{theta}_[5. 1.].png')   # uncomment to save other trajectory animation
        #fname = os.path.join(plot_data_dir, 'anim', 'mpc', f'x_type{theta}_[0.  4.5]_{i}.png')
        fig.savefig(fname, dpi=300)
        plt.close(fig)


def plot_no_guide_anim(theta):
    '''
    plot animation for no guide trajectory. plot three no guide trajectory in the same figure.
    '''
    if not os.path.exists( os.path.join(plot_data_dir, 'anim', 'no_guide') ):
        os.makedirs(os.path.join(plot_data_dir, 'anim', 'no_guide'))

    pltutil = PlotUtils()

    # load trajectory data
    fname = os.path.join(train_data_dir, 'no_guide', f'f{theta}_[6. 0. 3.].npy')
    x1 = np.load(fname)
    fname = os.path.join(train_data_dir, 'no_guide', f'f{theta}_[0.  8.  0.5].npy')
    x2 = np.load(fname)
    fname = os.path.join(train_data_dir, 'no_guide', f'f{theta}_[0. 4. 0.].npy')
    x3 = np.load(fname)
    
    for i in range(x1.shape[0]-1):
        fig, ax = plt.subplots()
        ax = pltutil.plot_env(ax)
        ax.plot(x1[:i+1,0], x1[:i+1,1], '^-', markersize=7, linewidth=2, color='tab:red', alpha=0.8, label='follower')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.xaxis.set_tick_params(labelsize='large')
        ax.yaxis.set_tick_params(labelsize='large')
        #ax.set_title('color map for follower '+str(theta))
        ax.legend(fontsize='x-large', loc='lower left')

        fname = os.path.join(plot_data_dir, 'anim', 'no_guide', f'type_{theta}_{i}.png')
        fig.savefig(fname)
        plt.close(fig)
    
    for i in range(x2.shape[0]-1):
        fig, ax = plt.subplots()
        ax = pltutil.plot_env(ax)
        ax.plot(x1[:,0], x1[:,1], '^-', markersize=7, linewidth=2, color='tab:red', alpha=0.8, label='follower')
        ax.plot(x2[:i+1,0], x2[:i+1,1], '^-', markersize=7, linewidth=2, color='tab:red', alpha=0.8)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.xaxis.set_tick_params(labelsize='large')
        ax.yaxis.set_tick_params(labelsize='large')
        #ax.set_title('color map for follower '+str(theta))
        ax.legend(fontsize='x-large', loc='lower left')

        fname = os.path.join(plot_data_dir, 'anim', 'no_guide', f'type_{theta}_{i+x1.shape[0]}.png')
        fig.savefig(fname)
        plt.close(fig)
    
    for i in range(x3.shape[0]-1):
        fig, ax = plt.subplots()
        ax = pltutil.plot_env(ax)
        ax.plot(x1[:,0], x1[:,1], '^-', markersize=7, linewidth=2, color='tab:red', alpha=0.8, label='follower')
        ax.plot(x2[:,0], x2[:,1], '^-', markersize=7, linewidth=2, color='tab:red', alpha=0.8)

        ax.plot(x3[:i+1,0], x3[:i+1,1], '^-', markersize=7, linewidth=2, color='tab:red', alpha=0.8)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.xaxis.set_tick_params(labelsize='large')
        ax.yaxis.set_tick_params(labelsize='large')
        #ax.set_title('color map for follower '+str(theta))
        ax.legend(fontsize='x-large', loc='lower left')

        fname = os.path.join(plot_data_dir, 'anim', 'no_guide', f'type_{theta}_{i+x1.shape[0]+x2.shape[0]}.png')
        fig.savefig(fname)
        plt.close(fig)



if __name__ == "__main__":
    #plot_adapt_loss_bar()
    plot_adapt_curve_theta(theta=2)
    #plot_mpc_traj(theta=1)
    #plot_mpc_traj_animation(theta=1)
    #plot_no_guide_anim(theta=4)
