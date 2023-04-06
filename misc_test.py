"""
This script performs miscellaneous tests encapsulated in a function.
"""
import numpy as np
from utils import Leader, Follower, Meta, PlotUtils, BRNet
import matplotlib.pyplot as plt
import torch


def test1():    # test environment plot
    pltutil = PlotUtils()
    fig, ax = plt.subplots()
    ax = pltutil.plot_env(ax)
    fig.savefig('tmp/tmp1.png')
    plt.close(fig)


def test2():    # test data sampling function
    leader = Leader()
    leader.meta = Meta()
    ff = Follower(0)
    
    print(leader.meta.sample_tasks(4))
    
    # random sample
    data = leader.meta.sample_task_theta_random(0, [], 10)
    print(data)
    data1 = leader.meta.sample_task_theta_random(0, data, 5)
    print(data1)
    
    # trajectory sample
    x_traj, a_traj = np.random.rand(leader.dimT+1, leader.dimx), np.random.rand(leader.dimT, leader.dimua)
    data2 = leader.meta.sample_task_theta(0, x_traj, a_traj, data, 10)
    print(data2)

    # near obstacle sample
    data3 = leader.meta.sample_task_theta_obs(0, [], 10)
    print(data3)
    data4 = leader.meta.sample_task_theta_obs(0, data3, 5)
    print(data4)


def test3():    # plot colormap
    leader = Leader()
    leader.meta = Meta()
    theta = 0
    ff = Follower(theta)
    def fz(x, y):    # compute follower's cost function
        cost = 0
        p = np.array([x, y])
        cost += ff.coeff[0] * (p-ff.pd) @ (p-ff.pd)
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
        cost += J_obs*50
        return cost
    
    x, y = np.arange(0, 10.1, 0.1), np.arange(0, 10.1, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((X.shape[0]-1, Y.shape[0]-1))
    for i in range(X.shape[0]-1):
        for j in range(X.shape[1]-1):
            Z[i, j] = fz(X[i,j], Y[i,j])
    fig, ax = plt.subplots()
    plot = ax.pcolormesh(X, Y, Z, cmap='RdPu')
    ax.set_aspect(1)
    ax.set_title('color map for follower '+str(theta))
    fig.colorbar(plot)
    fig.savefig('tmp/tmp2.png')
    plt.close(fig)


def test4():    # test nn related
    model = torch.nn.Sequential(torch.nn.Linear(2,3), torch.nn.Softmax())
    for n, p in model.named_parameters():
        print(n)
        print(p)
        print(p.grad)
    
    X, Y = torch.rand(1,2), torch.rand(1,3)
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(model(X), Y)
    h = torch.autograd.functional.hessian(loss_fn, (model(X),Y))
    print(h)

    loss.backward()
    

def test5():    # test BRNet gradient and intermediate computation
    torch.set_default_dtype(torch.float64)
    leader = Leader()
    brnet = BRNet()
    x, a = 10*torch.rand(leader.dimx)-5, torch.rand(leader.dimua)
    #x, a = torch.tensor([1.,2,3,4,5]), torch.tensor([1.,1])
    print(brnet(x,a))
    #print(brnet(x[None,:], a[None,:]))
    #print(brnet.get_intermediate_output(x, a))
    print("")
    jac_x, jac_a = brnet.compute_input_jac(x,a)
    print('analytic: ', jac_x, jac_a)
    print("")
    
    #check gradient using forward prediction
    eps = 1.4901161193847656e-08
    jac_xx, jac_aa = torch.zeros_like(jac_x), torch.zeros_like(jac_a)
    for i in range(leader.dimx):
        tmp = torch.zeros(leader.dimx)
        tmp[i] = eps
        res = (brnet(x+tmp, a) - brnet(x, a) ) / eps
        jac_xx[:, i] = res.data
        #jac_xx[:, i] = res.numpy().astype(float)
    for i in range(leader.dimua):
        tmp = torch.zeros(leader.dimua)
        tmp[i] = eps
        res = (brnet(x, a+tmp) - brnet(x, a) ) / eps
        jac_aa[:, i] = res.data
        #jac_aa[:, i] = res.numpy().astype(float)
    print('forward: ', jac_xx, jac_aa)
    print("difference:")
    print(jac_xx-jac_x)
    print(jac_aa-jac_a)


def test6():    # leader's optimal control
    leader = Leader()
    leader.meta = Meta()
    x0 = np.array([1,8, 0,8, 0.78])
    theta = 0
    br_data = np.load('data/br_data/f' + str(theta) + '_rand.npy')
    br = BRNet()
    br.load_state_dict(torch.load('data/indiv_data/br'+str(theta)+'.pth'))

    x_init, ua_init = leader.init1(x0)
    x_init, ua_init = leader.init2(x0)
    leader.obj_oc(x_init, ua_init)
    grad_x, grad_a = leader.grad_obj_oc(x_init, ua_init)

    # forward derivative approximation test
    eps=1.4901161193847656e-08
    approx_x = np.zeros_like(grad_x)
    for t in range(leader.dimT):
        for i in range(leader.dimx):
            tmp = np.zeros(leader.dimx)
            tmp[i] = eps
            xtmp_traj = x_init.copy()
            xtmp_traj[t+1, :] = x_init[t+1,:] + tmp
            approx_x[t, i] = (leader.obj_oc(xtmp_traj, ua_init) - leader.obj_oc(x_init, ua_init)) / eps
    print(grad_x - approx_x)
    approx_a = np.zeros_like(grad_a)
    for t in range(leader.dimT):
        for i in range(leader.dimua):
            tmp = np.zeros(leader.dimua)
            tmp[i] = eps
            uatmp_traj = ua_init.copy()
            uatmp_traj[t,:] = ua_init[t,:] + tmp
            approx_a[t, i] = (leader.obj_oc(x_init, uatmp_traj) - leader.obj_oc(x_init, ua_init)) / eps
    print(grad_a - approx_a)
    
    x_traj, ua_traj = leader.compute_opt_traj(br.state_dict(), x_init, ua_init)
    print(leader.obj_oc(x_traj, ua_traj))
    print(leader.grad_obj_oc(x_traj, ua_traj))


def test7():    # parameter change
    br = BRNet()
    mom_dict = br.get_zero_grad_dict()
    
    def change_dict(mom_dict):
        for key in mom_dict.keys():
            mom_dict[key] = 0.9 * mom_dict[key] + 1
        return

    change_dict(mom_dict)
    print(mom_dict)


def test8():    # test follower's real br and learned br model
    pltutil = PlotUtils()
    leader = Leader()
    leader.meta = Meta()
    theta = 0
    ff = Follower(theta)
    br = BRNet()
    fname = 'data/indiv_data/br' + str(theta) + '.pth'
    br.load_state_dict(torch.load(fname))
    
    D_rand = np.load('data/br_data/f' + str(theta) + '_rand.npy')
    D_obs = np.load('data/br_data/f' + str(theta) + '_obs.npy')
    D = D_rand
    #D = leader.to_torch(D)

    for i in range(20):
        idx = np.random.choice(D.shape[0])
        data = D[idx, :]
        x, ua, ub = data[:leader.dimx], data[leader.dimx: leader.dimx+leader.dimua], data[leader.dimx+leader.dimua:]
        pa, pb, phib = x[0:2], x[2:4], x[-1]
        
        #pb = [1.96341623, 2.78558818]
        #phib = [1]
        #x = np.array(pb + pb + phib)    # get rid
        #ua = np.array([0.1, 0.1])
    
        print('ua: ', ua)
        print('data (vb, wb): ', ub)
        pltutil.plot_leader_follower_pos(x)

        ub_pred = br(leader.to_torch(x), leader.to_torch(ua))
        ub_real = ff.get_br(x, ua)
        print('real (vb, wb): ', ub_real)
        print('predict (vb, wb): ', ub_pred.detach().numpy())
        print("")
    
    #pltutil.plot_follower_pos(np.array(pb+phib))
    #print(br(x, ua))
    

def test9():    # test leader's mpc
    theta = 0
    # possible x0: 
    x0 = np.array([1,8, 0,8, 0.5])
    x0 = np.array([5,1, 6,0, 3.])
    x0 = np.array([0,4.5, 0,4, 0.])

    import main as main_script
    x,ua, ub = main_script.receding_horizon(theta, x0)
    np.save('data/rc_traj_'+str(theta)+'_x_type'+str(theta)+'_'+str(x0[0:2])+'.npy', x)
    np.save('data/rc_traj_'+str(theta)+'_ua_type'+str(theta)+'_'+str(x0[0:2])+'.npy', ua)
    np.save('data/rc_traj_'+str(theta)+'_ub_type'+str(theta)+'_'+str(x0[0:2])+'.npy', ub)


def test10():   # test training results
    theta = 1
    ep_train = np.load('data/indiv_data/ep_train'+str(theta)+'.npy')
    ep_test = np.load('data/indiv_data/ep_test'+str(theta)+'.npy')

    fig, ax = plt.subplots()
    ax.plot(ep_train.mean(axis=1), label='ave train')
    ax.plot(ep_test.mean(axis=1), label='ave test')
    ax.set_title('training loss for follower'+str(theta))
    ax.legend()
    fig.savefig('tmp/tmp5.png')
    plt.close(fig)

    meta_cost = np.load('data/meta_data/meta_loss.npy')[:17500]
    fig, ax = plt.subplots()
    ax.plot(meta_cost)
    ax.set_title('meta training')
    plt.savefig('tmp/tmp4.png')
    plt.close()
    

def test12():   # train new adapted model
    theta = 3
    leader = Leader()
    leader.meta = Meta()
    br = BRNet()
    br.load_state_dict(torch.load('data/meta_data/meta_br.pth'))
    br_data_rand = np.load('data/br_data/f' + str(theta) + '_rand.npy')
    br_data_obs = np.load('data/br_data/f' + str(theta) + '_obs.npy')

    D_theta = leader.meta.sample_task_theta(theta, br_data_rand, br_data_obs, N=2000)
    D_theta = br_data_obs[leader.rng.choice(br_data_obs.shape[0], 2000), :]
    #D_theta = np.load('data/gdtest'+str(theta)+'.npy')
    #D_theta = D_theta[:1000, :]
    x, ua, ub = D_theta[:, :leader.dimx], D_theta[:, leader.dimx: leader.dimx+leader.dimua], D_theta[:, leader.dimx+leader.dimua:]
    x, ua, ub = leader.to_torch(x), leader.to_torch(ua), leader.to_torch(ub)
    loss_fn = torch.nn.MSELoss(reduction='mean')
    lr = 1e-4
    mom = 0.5
    optimizer = torch.optim.SGD(br.parameters(), lr=lr, momentum=mom)
    epoch, ITER = 200, 50
    tmp = []
    for ep in range(epoch):
        idx = leader.rng.choice(D_theta.shape[0], 200)
        idx = torch.from_numpy(idx)
        xx, uua, uub = x[idx,:], ua[idx,:], ub[idx,:]
        for i in range(ITER):
            loss = loss_fn(br(xx,uua), uub) *(leader.dimx+leader.dimua)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tmp.append(loss.item())
    print(tmp)
    # save adapted response model
    fname = 'data/meta_data/adapt' + str(theta) + '.pth'
    torch.save(br.state_dict(), fname)
    return tmp


if __name__ == "__main__":
    #test1()     # test environment plot
    #test2()     # test sampling
    #test3()     # plot color map for follower's cost
    #test4()     # test nn related
    #test5()     # test BRnet gradient
    #test6()     # test leader's optimal control related
    #test7()     # test dict parameter change
    #test8()     # test follower's real br and learned br model
    #test9()     # test leader's mpc
    #test10()    # test training results
    test12()