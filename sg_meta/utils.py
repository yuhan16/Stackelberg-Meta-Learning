import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from . import config_data_dir, tmp_data_dir


def save_results(dname, res):
    '''Save the provided results.'''
    if not os.path.exists(dname):
        os.makedirs(dname)
    
    for key in res:
        if '.pt' in key:
            torch.save(res[key], f'{dname}/{key}')
        else:
            np.save(f'{dname}/{key}', res[key])


class PlotUtils:
    """
    This class defines plotting related functions.
    """
    def __init__(self) -> None:
        param = json.load(open(os.path.join(config_data_dir, 'parameters.json')))

        self.ws_len = param['ws_length']
        self.obs = param['obstacle_settings']
        self.pd = param['destination']
    

    def plot_env(self, ax):
        """
        This function plots the simulation environments.
        """
        ax = self.plot_obs(ax)
        im = plt.imread(os.path.join(config_data_dir, 'dest-logo.png'))
        ax.imshow(im, extent=(8.4,9.6,8.4,9.6), zorder=-1)
        
        ax.set_xlim(0, 10.1)
        ax.set_ylim(0, 10.1)
        ax.set_aspect(1)
        ax.xaxis.set_tick_params(labelsize='large')
        ax.yaxis.set_tick_params(labelsize='large')
        return ax
    

    def plot_obs(self, ax):
        """
        This function plots the obstacles.
        """
        def plot_l1_norm(ax, obs):      # plot l1 norm ball, a diamond
            xc, yc, rc = obs[0], obs[1], obs[2]
            x_scale, y_scale = obs[4], obs[5]
            # compute half width and height, construct vertex array
            width, height = rc*x_scale, rc*y_scale
            xy = [[xc, yc+height], [xc-width, yc], [xc, yc-height], [xc+width, yc]]
            p = mpatches.Polygon(np.array(xy), closed=True, ec='k', fc='0.9', fill=True, ls='--', lw=2)
            ax.add_patch(p)
            return ax
        
        def plot_l2_norm(ax, obs):      # plot l2 norm ball, an ellipse
            xc, yc, rc = obs[0], obs[1], obs[2]
            x_scale, y_scale = obs[4], obs[5]
            # compute width and height
            width, height = 2*rc*x_scale, 2*rc*y_scale
            p = mpatches.Ellipse((xc, yc), width, height, ec='k', fc='0.9', fill=True, ls='--', lw=2)
            ax.add_patch(p)
            return ax
        
        def plot_linf_norm(ax, obs):    # plot linf norm ball, a rectangle
            xc, yc, rc = obs[0], obs[1], obs[2]
            x_scale, y_scale = obs[4], obs[5]
            # compute x0, y0, width, and height
            width, height = 2*rc*x_scale, 2*rc*y_scale
            x0, y0 = xc-width/2, yc-height/2
            p = mpatches.Rectangle((x0,y0), width, height, ec='k', fc='0.9', fill=True, ls='--', lw=2)
            ax.add_patch(p)
            return ax            

        obs_num = len(self.obs)
        for i in range(obs_num):
            obs_i = self.obs[i]
            if obs_i[3] == 1:
                ax = plot_l1_norm(ax, obs_i)
            elif obs_i[3] == 2:
                ax = plot_l2_norm(ax, obs_i)
            else:
                ax = plot_linf_norm(ax, obs_i)
        return ax
    

    def plot_traj(self, x_traj, fname=None):
        """
        This function plots the trajectory.
        """
        if fname is None:
            fname = os.path.join(tmp_data_dir, 'x_traj.png')
        fig, ax = plt.subplots()
        ax = self.plot_env(ax)
        if x_traj.shape[1] <= 2:
            pa = x_traj[:, 0:2]
            ax.plot(pa[:,0], pa[:,1], 'o')
        else:
            pa = x_traj[:, 0:2]
            pb = x_traj[:, 2:4]
            ax.plot(pa[:,0], pa[:,1], 'o')
            ax.plot(pb[:,0], pb[:,1], 's')
        
        fig.savefig(fname, dpi=100)
        plt.close(fig)
    