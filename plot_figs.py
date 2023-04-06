"""
This script plots figures. Define each plot function.
"""
import numpy as np
from utils import Leader, Follower, Meta, PlotUtils, BRNet
import matplotlib.pyplot as plt


def plot_adapt_curve():     # plot final adapted loss for all followers
    leader = Leader()
    leader.meta = Meta()

    import plot_data
    param_ave = np.array(plot_data.param_ave_loss)
    output_ave = np.array(plot_data.output_ave_loss)
    adapt = np.array(plot_data.adapt_loss)

    fig, ax = plt.subplots()
    x = ['θ=0', 'θ=1', 'θ=2', 'θ=3', 'θ=4']
    x_axis = np.arange(leader.meta.total_type)
    ax.bar(x_axis-0.2, adapt[:,-1], width=0.2, color=(0.121,0.467,0.705), alpha=0.8, label='Adaptation')
    ax.bar(x_axis, output_ave[:,-1], width=0.2, color=(0.992, 0.725, 0.419), alpha=0.8, label='Output-Ave')
    ax.bar(x_axis+0.2, param_ave[:,-1]/2, width=0.2, color=(0.925, 0.364, 0.231), alpha=0.8, label='Param-Ave (/2)')
    ax.set_xticks(x_axis)
    ax.set_xticklabels(x)
    ax.xaxis.set_tick_params(labelsize='large')
    ax.yaxis.set_tick_params(labelsize='large')
    ax.set_xlabel('Follower type', fontsize='xx-large')
    ax.set_ylabel('MSE Error', fontsize='xx-large')
    ax.legend(fontsize='x-large')
    fig.savefig('data/plots/adapt_final.png', dpi=300)
    plt.close(fig)


def plot_adapt_theta(theta):    # plot adaptation curve for type theta follower
    leader = Leader()
    leader.meta = Meta()

    import plot_data
    param_ave = np.array(plot_data.param_ave_loss)
    output_ave = np.array(plot_data.output_ave_loss)
    adapt = np.array(plot_data.adapt_loss)

    fig, ax = plt.subplots()
    #x = ['θ=0', 'θ=1', 'θ=2', 'θ=3', 'θ=4']
    x_axis = np.arange(leader.meta.total_type)
    ax.plot(adapt[theta,:], color=(0.121,0.467,0.705), alpha=0.8, label='Adaptation')
    ax.plot(output_ave[theta,:], color=(0.992, 0.725, 0.419), alpha=0.8, label='Output-Ave')
    #ax.bar(x_axis-0.2, adapt[:,-1], width=0.2, color=(0.121,0.467,0.705), alpha=0.8, label='Adaptation')
    #ax.bar(x_axis, output_ave[:,-1], width=0.2, color=(0.992, 0.725, 0.419), alpha=0.8, label='Output-Ave')
    #ax.bar(x_axis+0.2, param_ave[:,-1]/2, width=0.2, color=(0.925, 0.364, 0.231), alpha=0.8, label='Param-Ave (/2)')
    #ax.set_xticks(x_axis)
    #ax.set_xticklabels(x)
    ax.xaxis.set_tick_params(labelsize='large')
    ax.yaxis.set_tick_params(labelsize='large')
    ax.set_xlabel('Iteration', fontsize='xx-large')
    ax.set_ylabel('MSE Error', fontsize='xx-large')
    ax.legend(fontsize='x-large')
    fig.savefig('data/plots/adapt'+str(theta)+'.png', dpi=300)
    plt.close(fig)


def plot_mpc_traj(theta):       # plot mpc trajectory for type theta follower 
    leader = Leader()
    pltutil = PlotUtils()
    fig, ax = plt.subplots()
    ax = pltutil.plot_env(ax)
    color = [(0.121,0.467,0.705), (0.992, 0.725, 0.419), (0.925, 0.364, 0.231)]
    
    x = np.load('data/rc_traj_'+str(theta)+'/x_type'+str(theta)+'_[1. 8.].npy')
    pa, pb = x[:, 0:leader.dimxa], x[:, leader.dimxa: leader.dimx-1]
    ax.plot(pa[:,0], pa[:,1], 'o-', markersize=6, linewidth=1, color=color[0], alpha=0.5, markeredgecolor='none', label='leader')
    ax.plot(pb[:,0], pb[:,1], '^-', markersize=6, linewidth=1, color=color[2], alpha=0.5, markeredgecolor='none', label='follower')

    x = np.load('data/rc_traj_'+str(theta)+'/x_type'+str(theta)+'_[5. 1.].npy')
    pa, pb = x[:75, 0:leader.dimxa], x[:75, leader.dimxa: leader.dimx-1]
    ax.plot(pa[:,0], pa[:,1], 'o-', markersize=6, linewidth=2, color=color[0], alpha=0.5)
    ax.plot(pb[:,0], pb[:,1], '^-', markersize=6, linewidth=2, color=color[2], alpha=0.5)

    #x = np.load('data/rc_traj_'+str(theta)+'/x_type'+str(theta)+'_[0. 4.5].npy')
    #pa, pb = x[:, 0:leader.dimxa], x[:, leader.dimxa: leader.dimx-1]
    #ax.plot(pa[:,0], pa[:,1], 'o-', markersize=6, linewidth=2, color=color[0], alpha=0.5)
    #ax.plot(pb[:,0], pb[:,1], '^-', markersize=6, linewidth=2, color=color[2], alpha=0.5)

    ax.xaxis.set_tick_params(labelsize='x-large')
    ax.yaxis.set_tick_params(labelsize='x-large')
    ax.legend(fontsize='x-large')
    fig.set_figwidth(4.8)
    fig.savefig('data/plots/mpc'+str(theta)+'.png', dpi=300)
    plt.close(fig)


def plot_traj_animation(theta):
    leader = Leader()
    pltutil = PlotUtils()
    color = [(0.121,0.467,0.705), (0.992, 0.725, 0.419), (0.925, 0.364, 0.231)]
    
    x = np.load('data/rc_traj_'+str(theta)+'/x_type'+str(theta)+'_[1. 8.].npy')
    #x = np.load('data/rc_traj_'+str(theta)+'/x_type'+str(theta)+'_[5. 1.].npy')
    x = np.load('data/rc_traj_'+str(theta)+'/x_type'+str(theta)+'_[0. 4.5].npy')
    pa, pb = x[:, 0:leader.dimxa], x[:, leader.dimxa: leader.dimx-1]
    for i in range(x.shape[0]):
        fig, ax = plt.subplots()
        ax = pltutil.plot_env(ax)

        ax.plot(pa[:i+1,0], pa[:i+1,1], 'o-', markersize=6, linewidth=1, color=color[0], alpha=0.5, markeredgecolor='none', label='leader')
        ax.plot(pb[:i+1,0], pb[:i+1,1], '^-', markersize=6, linewidth=1, color=color[2], alpha=0.5, markeredgecolor='none', label='follower')

        ax.xaxis.set_tick_params(labelsize='x-large')
        ax.yaxis.set_tick_params(labelsize='x-large')
        ax.legend(fontsize='x-large', loc='lower left')
        fig.set_figwidth(4.8)
        fig.savefig('anim/x_'+str(theta)+'_[0. 4]_'+str(i)+'.png', dpi=300)
        plt.close(fig)


def plot_no_guide_anim(theta):
    leader = Leader()
    pltutil = PlotUtils()
    x1 = np.load('f2x1.npy')[:30, :]
    x2 = np.load('f2x2.npy')[:40, :]
    x3 = np.load('f2x3.npy')[:10, :]
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
        fig.savefig('anim/noguide_type_'+str(theta)+'_'+str(i)+'.png')
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
        fig.savefig('anim/noguide_type_'+str(theta)+'_'+str(i+x1.shape[0])+'.png')
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
        fig.savefig('anim/noguide_type_'+str(theta)+'_'+str(i+x1.shape[0]+x2.shape[0])+'.png')
        plt.close(fig)


def plot_adapt_compare():   # print ave and adapt result
    import main_ave_largenn, main, main_ave_br
    leader = Leader()
    leader.meta = Meta()
    fig, ax = plt.subplots(2, 3)

    ave_loss = main_ave_br.main()
    for i in range(leader.meta.total_type):
        nrow, ncol = i//3, i%3
        adapt_loss = main.sg_adapt(i)
        nn_loss = main_ave_largenn.adapt(i)
        ax[nrow, ncol].plot(adapt_loss, label='adapt')
        ax[nrow, ncol].plot(nn_loss, label='nn')
        #ax[nrow, ncol].plot(ave_loss, label='ave')
        ax[nrow, ncol].set_title('type '+str(i))
        ax[nrow, ncol].legend()
    fig.savefig('data/plots/adapt_compare.png')
    plt.close(fig)


if __name__ == "__main__":
    #plot_adapt_curve()
    plot_adapt_theta(4)
    #plot_mpc_traj(4)
    #plot_traj_animation(2)
    #plot_no_guide_anim(2)
    #plot_adapt_compare()   # print adaptation results for three methods