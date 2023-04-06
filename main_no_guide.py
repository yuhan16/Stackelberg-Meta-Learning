"""
Implement no guidance control. zero guidance cost
"""
import numpy as np
from utils import Leader, Follower, Meta, PlotUtils
import matplotlib.pyplot as plt

def noguide(theta, x0):
    leader = Leader()
    leader.meta = Meta()
    ff = Follower(theta)
    ff.coeff[1] = 0     # set zero guidance cost
    #ff.coeff[2] = 1

    ua = np.zeros(leader.dimua)
    pa = np.zeros(leader.dimxa)
    ITER = 200
    ub_traj = []
    xb_traj = []
    xb_traj.append(x0[leader.dimxa:])
    for i in range(ITER):
        x = np.concatenate( (pa, xb_traj[i]) )
        ub_traj.append( ff.get_br(x, ua) )
        
        if xb_traj[i][0] > 10 or xb_traj[i][0] < -0.1 or xb_traj[i][1] > 10 or xb_traj[i][1] < -0.1:
            break

        x_tp1 = ff.dynamics(x, ua, ub_traj[i])
        xb_traj.append( x_tp1[leader.dimxa:] )
        print(xb_traj[i], ub_traj[i])
        
    #print(xb_traj)
    #print(ub_traj)
    return np.array(xb_traj)


def plot_color_map():    # plot colormap
    leader = Leader()
    leader.meta = Meta()
    theta = 2
    ff = Follower(theta)
    def fz(x, y):    # compute follower's cost function
        cost = 0
        p = np.array([x, y])
        #cost += ff.coeff[0] * (p-ff.pd) @ (p-ff.pd)
        J_obs = 0
        for i in range(len(ff.obs)):
            obs_i = ff.obs[i]
            xc, yc, rc = obs_i[0], obs_i[1], obs_i[2]
            x_scale, y_scale = obs_i[4], obs_i[5]
            f1 = np.diag([1/x_scale, 1/y_scale]) @ (p-np.array([xc,yc]))
            if obs_i[3] == 1:
                f2 = np.linalg.norm(f1, ord=1)
            elif obs_i[3] == 2:
                f2 = np.linalg.norm(f1, ord=2)
            else:
                f2 = np.linalg.norm(f1, ord=np.inf)
            # check safety margin
            if ff.coeff[3]*(f2-rc) <= 0.1:
                J_obs -= np.log(0.1)
                #return np.inf
            elif ff.coeff[3]*(f2-rc) <= 1:
                J_obs -= np.log(ff.coeff[3]*(f2-rc))
            else:
                pass
        cost += J_obs*10
        return cost
    
    x, y = np.arange(0, 10.1, 0.1), np.arange(0, 10.1, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((X.shape[0]-1, Y.shape[0]-1))
    for i in range(X.shape[0]-1):
        for j in range(X.shape[1]-1):
            Z[i, j] = fz(X[i,j], Y[i,j])
    fig, ax = plt.subplots()
    plot = ax.pcolormesh(X, Y, Z, cmap='Blues')
    ax.set_aspect(1)

    # run main script to generate no guide trajectory first
    x1 = np.load('data/no_guide/f'+str(theta)+'x1.npy')
    x2 = np.load('data/no_guide/f'+str(theta)+'x2.npy')
    x3 = np.load('data/no_guide/f'+str(theta)+'x3.npy')
    ax.plot(x1[:,0], x1[:,1], '^-', markersize=7, linewidth=2, color='tab:red', alpha=0.8, label='follower')
    ax.plot(x2[:,0], x2[:,1], '^-', markersize=7, linewidth=2, color='tab:red', alpha=0.8)
    ax.plot(x3[:,0], x3[:,1], '^-', markersize=7, linewidth=2, color='tab:red', alpha=0.8)

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.xaxis.set_tick_params(labelsize='large')
    ax.yaxis.set_tick_params(labelsize='large')
    #ax.set_title('color map for follower '+str(theta))
    ax.legend(fontsize='x-large', loc=7)
    fig.colorbar(plot)
    fig.savefig('data/plots/noguide_'+str(theta)+'.png', dpi=300)
    plt.close(fig)



def main():
    pltutil = PlotUtils()
    fig, ax = plt.subplots()
    ax = pltutil.plot_env(ax)
    color = [(0.121,0.467,0.705), (0.992, 0.725, 0.419), (0.925, 0.364, 0.231)]

    theta = 2
    x0 = np.array([0,0, 0,4,0])
    x1 = noguide(theta, x0)
    ax.plot(x1[:,0], x1[:,1], '^-', markersize=4, linewidth=2, color=color[2], alpha=0.8, label='follower')

    x0 = np.array([0,0, 0,8,0.5])
    x2 = noguide(theta, x0)
    ax.plot(x2[:,0], x2[:,1], '^-', markersize=4, linewidth=2, color=color[2], alpha=0.8)

    x0 = np.array([0,0, 6,0,3.])
    x3= noguide(theta, x0)
    ax.plot(x3[:,0], x3[:,1], '^-', markersize=4, linewidth=2, color=color[2], alpha=0.8)

    #np.save('f2x1.npy', x1)
    #np.save('f2x2.npy', x2)
    #np.save('f2x3.npy', x3)
    x1 = np.save('data/no_guide/f'+str(theta)+'x1.npy', x1)
    x2 = np.save('data/no_guide/f'+str(theta)+'x2.npy', x2)
    x3 = np.dave('data/no_guide/f'+str(theta)+'x3.npy', x3)
    
    # ax.legend(fontsize='x-large')
    # fig.savefig('tmp/tmp13.png')
    # plt.close(fig)

if __name__ == '__main__':
    import os
    if not os.path.exists('data/no_guide'):
        os.mkdir('data/no_guide')
    
    plot_color_map()
    #main()