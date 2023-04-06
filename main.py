"""
This is the main script for implementing sg meta-learing algorithms.
"""
import os
import numpy as np
import torch
from utils import Leader, Follower, Meta, BRNet, PlotUtils


def generate_br_data():
    """
    This function generates br data using randomly generated x_traj and ua_traj.
    The data is stored and used for data set in meta-learning.
    Each type of follower has a separate data set.
    """
    leader = Leader()
    leader.meta = Meta()
    if not os.path.exists('data/'):
        os.mkdir('data/')
        os.mkdir('data/br_data/')
    elif not os.path.exists('data/br_data/'):
        os.mkdir('data/br_data/')
    else:
        pass
    
    # print parameters used for generating the data
    from contextlib import redirect_stdout
    import param_settings
    fname = 'data/br_data/parameters.txt'
    with open(fname, 'w') as out:
        with redirect_stdout(out):
            print('Parameters that used to simulate the data set. \n')
            param_settings.print_key_parameters()
    
    # generate data
    for theta in range(leader.meta.total_type):
        print('Generating simulation data for follower {}...'.format(theta))
        fname = 'data/br_data/f'+str(theta)+'_rand.npy'
        if not os.path.exists(fname):
        #if True:
            data_theta = leader.meta.sample_task_theta_random(theta, [], N=15000)
            np.save(fname, data_theta)
        
        fname = 'data/br_data/f' + str(theta) + '_obs.npy'
        if not os.path.exists(fname):
        #if True:
            data_theta_obs = leader.meta.sample_task_theta_obs(theta, [], N=15000)
            np.save(fname, data_theta_obs)


def sg_meta():
    """
    This function implements the meta-learning algorithm to find the generalized br model.
    """
    leader = Leader()
    leader.meta = Meta()
    br_data_rand, br_data_obs = [], []
    for i in range(leader.meta.total_type):
        fname = 'data/br_data/f' + str(i) + '_rand.npy'
        br_data_rand.append( np.load(fname) )
        leader.rng.shuffle(br_data_rand[i])
        br_data_rand[i] = br_data_rand[i][:10000,:]
        
        fname = 'data/br_data/f' + str(i) + '_obs.npy'
        br_data_obs.append( np.load(fname) )
        leader.rng.shuffle(br_data_obs[i])
        br_data_obs[i] = br_data_obs[i][:5000,:]

    iter, ITER = 0, int(5e5)
    brnet = BRNet()
    br = brnet.state_dict()     # update state_dict in the training
    mom_dict = brnet.get_zero_grad_dict()   # record momentum in SGD (outer loop)
    tmp = []
    import time
    start_t = time.time()
    while iter < ITER:
        br_list = []
        D_train_list, D_test_list = [], []
        task_sample = leader.meta.sample_tasks(5)
        #task_sample = [0]   # for testing

        for i in range(len(task_sample)):
            theta = task_sample[i]
            data_theta_rand = br_data_rand[theta]
            data_theta_obs = br_data_obs[theta]
            
            # sample D_train for follower theta
            D_train = leader.meta.sample_task_theta(theta, data_theta_rand, data_theta_obs, N=100)
            #D_train = leader.meta.sample_task_theta_random(theta, data_theta_rand, N=100)
            D_train_list.append(D_train)
            
            # inner loop update
            br_theta = leader.meta.update_br_theta(br, D_train)
            br_list.append( br_theta )
            
            # sample D_test for follower theta
            D_test = leader.meta.sample_task_theta(theta, data_theta_rand, data_theta_obs, N=100)
            #D_test = leader.meta.sample_task_theta_random(theta, data_theta_rand, N=100)
            D_test_list.append(D_test)
        # outer loop learning, manually perform SGD
        br = leader.meta.update_br_meta(task_sample, br, br_list, D_test_list, mom_dict)
        iter += 1

        # soft warmup for choosing beta
        ttmp = 5e-5 + 0.5*(5e-4 - 5e-5) * (1 + np.cos(0.5*iter/100))
        #print(ttmp)
        leader.meta.beta = ttmp

        # compute meta loss and print things
        meta_loss = np.zeros(len(task_sample))
        for i in range(len(task_sample)):
            meta_loss[i] = leader.meta.compute_meta_cost_theta(br, D_test_list[i])
        tmp.append(meta_loss.mean())
        if iter % 1000 == 0:
            print('Meta iter {}/{}, meta loss: {:.7f}'.format(iter, ITER, meta_loss.mean()))

    end_t = time.time()
    print("Meta-training done. Elapsed time: {:.3f} min.".format((end_t-start_t)/60))
    np.save('data/meta_data/meta_loss.npy', np.array(tmp))

    """
    import matplotlib.pyplot as plt
    plt.plot(tmp)
    plt.savefig('tmp/tmp4.png')
    plt.close()
    """

    # save meta response model
    if not os.path.exists('data/'):
        os.mkdir('data/')
        os.mkdir('data/meta_data/')
    elif not os.path.exists('data/meta_data/'):
        os.mkdir('data/meta_data/')
    else:
        pass
    fname = 'data/meta_data/meta_br.pth'
    torch.save(br, fname)


def sg_meta_gpu():
    """
    This function implements the meta-learning algorithm to find the generalized br model.
    """
    def prepare_data(data):
        data_tensor = torch.from_numpy(data).double().to(device)
        #data_tensor = data_tensor.to(device)
        x, ua, ub = data_tensor[:, :leader.dimx], data_tensor[:, leader.dimx: leader.dimx+leader.dimua], data_tensor[:, leader.dimx+leader.dimua:]
        return x, ua, ub
    
    device = 'cuda:3'   # or other devices
    leader = Leader()
    leader.meta = Meta()
    br_data_rand, br_data_obs = [], []      # load offline data set
    for i in range(leader.meta.total_type):
        fname = 'data/br_data/f' + str(i) + '_rand.npy'
        br_data_rand.append( np.load(fname) )
        leader.rng.shuffle(br_data_rand[i])
        br_data_rand[i] = br_data_rand[i][:10000,:]
        
        fname = 'data/br_data/f' + str(i) + '_obs.npy'
        br_data_obs.append( np.load(fname) )
        leader.rng.shuffle(br_data_obs[i])
        br_data_obs[i] = br_data_obs[i][:5000,:]

    iter, ITER = 0, int(2)
    br = BRNet().to(device)
    loss_fn = torch.nn.MSELoss(reduction='mean')
    optimizer_meta = torch.optim.SGD(br.parameters(), lr=leader.meta.beta, momentum=leader.meta.mom)
    tmp = []
    import time
    start_t = time.time()
    while iter < ITER:
        br_list = []
        D_train_list, D_test_list = [], []
        task_sample = leader.meta.sample_tasks(5)
        #task_sample = [0]   # for testing

        for i in range(len(task_sample)):
            theta = task_sample[i]
            data_theta_rand = br_data_rand[theta]
            data_theta_obs = br_data_obs[theta]
            
            # sample D_train for follower theta
            D_train = leader.meta.sample_task_theta(theta, data_theta_rand, data_theta_obs, N=100)
            #D_train = leader.meta.sample_task_theta_random(theta, data_theta_rand, N=100)
            D_train_list.append(D_train)
            
            # inner loop update
            br_i = BRNet().to(device)
            br_i.load_state_dict(br.state_dict())
            x, ua, ub = prepare_data(D_train)
            loss_i = loss_fn(br_i(x,ua), ub) *(leader.dimx+leader.dimua)
            optimizer_i = torch.optim.SGD(br_i.parameters(), lr=leader.meta.alp, momentum=0)
            optimizer_i.zero_grad()
            loss_i.backward()
            optimizer_i.step()
            br_list.append( br_i.state_dict() )

            # sample D_test for follower theta
            D_test = leader.meta.sample_task_theta(theta, data_theta_rand, data_theta_obs, N=100)
            #D_test = leader.meta.sample_task_theta_random(theta, data_theta_rand, N=100)
            D_test_list.append(D_test)

        # outer loop learning, manually perform SGD
        for i in range(leader.meta.total_type):
            br.load_state_dict(br_list[i])
            x, ua, ub = prepare_data(D_test_list[i])
            loss = loss_fn(br(x,ua), ub) *(leader.dimx+leader.dimua)/len(task_sample)
            optimizer_meta.zero_grad()
            loss.backward()
            optimizer_meta.step()
        iter += 1

        # compute meta loss and print things
        meta_loss = np.zeros(len(task_sample))
        for i in range(len(task_sample)):
            meta_loss[i] = leader.meta.compute_meta_cost_theta(br.state_dict(), D_test_list[i])
        tmp.append(meta_loss.mean())
        if iter % 1000 == 0:
            print('Meta iter {}/{}, meta loss: {:.7f}'.format(iter, ITER, meta_loss.mean()))

    end_t = time.time()
    print("Meta-training done. Elapsed time: {:.3f} min.".format((end_t-start_t)/60))
    np.save('data/meta_data/meta_loss.npy', np.array(tmp))

    # import matplotlib.pyplot as plt
    # plt.plot(tmp)
    # plt.savefig('tmp/tmp4.png')
    # plt.close()

    # save meta response model
    if not os.path.exists('data/'):
        os.mkdir('data/')
        os.mkdir('data/meta_data/')
    elif not os.path.exists('data/meta_data/'):
        os.mkdir('data/meta_data/')
    else:
        pass
    fname = 'data/meta_data/meta_br.pth'
    torch.save(br, fname)


def sg_adapt(theta):
    """
    This function implements the adaptation algorithm to find the adapted br model for type theta follower.
    """
    leader = Leader()
    leader.meta = Meta()
    br = BRNet()
    br.load_state_dict(torch.load('data/meta_data/meta_br.pth'))
    br_data_rand = np.load('data/br_data/f' + str(theta) + '_rand.npy')
    br_data_obs = np.load('data/br_data/f' + str(theta) + '_obs.npy')

    D_theta = leader.meta.sample_task_theta(theta, br_data_rand, br_data_obs, N=2000)
    D_theta = np.load('data/gdtest'+str(theta)+'.npy')
    D_theta = D_theta[:1000, :]
    x, ua, ub = D_theta[:, :leader.dimx], D_theta[:, leader.dimx: leader.dimx+leader.dimua], D_theta[:, leader.dimx+leader.dimua:]
    x, ua, ub = leader.to_torch(x), leader.to_torch(ua), leader.to_torch(ub)
    loss_fn = torch.nn.MSELoss(reduction='mean')
    lr = 1e-4
    mom = 0.6
    optimizer = torch.optim.SGD(br.parameters(), lr=lr, momentum=mom)
    ITER = 50
    tmp = []
    for i in range(ITER):
        loss = loss_fn(br(x,ua), ub) *(leader.dimx+leader.dimua)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tmp.append(loss.item())
    print(tmp)
    # save adapted response model
    fname = 'data/meta_data/adapt' + str(theta) + '.pth'
    torch.save(br.state_dict(), fname)
    return tmp


def train_model_theta(theta, device=None):
    """
    This function trains a separate br model for type theta follower for comparison.
    Difference between sg_adapt: no initial br_meta.
    determine NN layers and performance.
    """
    #device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
    def generate_batch_data(data, batch_size):
        """
        This function generates batch data for training and testing.
        """
        batch_data = []
        ITER = data.shape[0] // batch_size
        x, ua, ub = data[:, :leader.dimx], data[:, leader.dimx: leader.dimx+leader.dimua], data[:, leader.dimx+leader.dimua:]
        x, ua, ub = leader.to_torch(x), leader.to_torch(ua), leader.to_torch(ub)
        x, ua, ub = x.to(device), ua.to(device), ub.to(device)
        for i in range(ITER):
            idx = i*batch_size
            batch_data.append([x[idx: idx+batch_size,:], ua[idx: idx+batch_size,:], ub[idx: idx+batch_size,:]])
        return batch_data

    import time
    start_t = time.time()
    print('Training individual brnet for follower {}...\n------------'.format(theta))
    leader = Leader()
    leader.meta = Meta()
    br = BRNet().to(device)
    br_data_rand = np.load('data/br_data/f' + str(theta) + '_rand.npy')
    leader.rng.shuffle(br_data_rand)
    br_data_obs = np.load('data/br_data/f' + str(theta) + '_obs.npy')
    leader.rng.shuffle(br_data_obs)

    # prepare training and testing data sets
    N_train = 15000
    N_test = 1000
    batch_size = 100
    D_train = leader.meta.sample_task_theta(theta, br_data_rand[:10000], br_data_obs[:10000], N=N_train)
    D_test = leader.meta.sample_task_theta(theta, br_data_rand[10000:], br_data_obs[10000:], N=N_test)
    
    lr = 2.5e-4
    mom = 0.8
    epoch = 40000
    loss_fn = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(br.parameters(), lr=lr, momentum=mom)
    #optimizer = torch.optim.Adam(br.parameters(), lr=0.0001)
    ep_train = []
    ep_test = []
    for t in range(epoch):
        print("Epoch: {}\n-------------".format(t+1))
        epoch_train_loss = []
        # shuffle training data in each epoch
        leader.rng.shuffle(D_train)
        batch_data = generate_batch_data(D_train, batch_size)
        for i in range(len(batch_data)):   # iterate over all training data organized by batch
            x, ua, ub = batch_data[i][0], batch_data[i][1], batch_data[i][2]
            loss = loss_fn(br(x,ua), ub) *(leader.dimx+leader.dimua)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss.append(loss.item())
            if (i+1) % 50 == 0:
                print('iter: {}/{}, loss: {:.7f}'.format(i+1, len(batch_data), loss.item()))

        # for testing
        epoch_test_loss = []
        leader.rng.shuffle(D_test)
        batch_test_data = generate_batch_data(D_test, batch_size)
        for i in range(len(batch_test_data)):
            x, ua, ub = batch_test_data[i][0], batch_test_data[i][1], batch_test_data[i][2]
            with torch.no_grad():
                loss = loss_fn(br(x,ua), ub) *(leader.dimx+leader.dimua)
                epoch_test_loss.append(loss.item())
        
        ep_train.append(epoch_train_loss)
        ep_test.append(epoch_test_loss)

    end_t = time.time()
    print("Elapsed time: {:.3f} min.".format((end_t-start_t)/60))
    
    """
    # plot epoch loss for visualization
    ep_train = np.array(ep_train)
    ep_test = np.array(ep_test)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(ep_train.mean(axis=1), label='ave train')
    ax.plot(ep_test.mean(axis=1), label='ave test')
    ax.set_title('training loss for follower'+str(theta))
    ax.legend()
    fig.savefig('tmp/tmp5.png')
    plt.close(fig)
    """

    # save individual response model
    if not os.path.exists('data/indiv_data/'):
        os.mkdir('data/indiv_data/')
    fname = 'data/indiv_data/br' + str(theta) + '.pth'
    torch.save(br.state_dict(), fname)
    np.save('data/indiv_data/ep_train'+str(theta)+'.npy', ep_train)
    np.save('data/indiv_data/ep_test'+str(theta)+'.npy', ep_test)


def receding_horizon(theta, x0=None):
    """
    This function implements the leader's receding horizon control to guide type theta follower.
    """
    fname = 'data/indiv_data/br' + str(theta) + '.pth'
    fname = 'data/meta_data/adapt' + str(theta) + '.pth'
    if not os.path.exists(fname):
        raise Exception("No BR model found for type {}. Run adaptation or other learning algorithms first.".format(theta))

    leader = Leader()
    follower = Follower(theta)
    br = BRNet()
    br.load_state_dict(torch.load(fname))
    #x0 = np.array([1,8, 0,8, 0.5])
    #x0 = np.array([5,1, 6,0, 3.])
    #x0 = np.array([0,4.5, 0,4, 0.])
    xd = np.array(leader.pd + leader.pd + [0])
    iter, ITER = 0, 150
    eps = 4e-1
    x_traj, ua_traj, ub_traj = [], [], []
    x_traj.append(x0)
    x_t = x0

    # get initial trajectory
    x_init, ua_init = leader.init2(x0)
    while iter < ITER:
        print('rc iter:', iter)
        x_opt, a_opt = leader.compute_opt_traj(br.state_dict(), x_init, ua_init)
        ua_t = a_opt[0, :]
        # get follower's real response
        ub_t = follower.get_br(x_t, ua_t)
        x_tp1 = follower.dynamics(x_t, ua_t, ub_t)

        x_traj.append(x_tp1)
        ua_traj.append(ua_t)
        ub_traj.append(ub_t)
        print(x_t, ua_t, ub_t)

        if np.linalg.norm(x_tp1[leader.dimxa: leader.dimx-1]-follower.pd) <= eps:
            break
        
        # reformulate initial conditions
        x_t = x_tp1
        #x_init[:-1] = x_init[1:]
        #x_init[-1] = xd
        #ua_init[:-1] = ua_init[1:]
        #ua_init[-1] = np.zeros(leader.dimua)
        x_init, ua_init = leader.init2(x_t)
        iter += 1
        pltutil = PlotUtils()
        pltutil.plot_traj(np.array(x_traj))
    x_traj, ua_traj, ub_traj = np.array(x_traj), np.array(ua_traj), np.array(ub_traj)
    return x_traj, ua_traj, ub_traj
        

def main():
    # examples to run different functions.
    generate_br_data()
    sg_meta()
    sg_adapt(0)
    x, ua, ub = receding_horizon(0)     # save if needed
    train_model_theta(0, 'cpu')


if __name__ == "__main__":
    main()
    #sg_meta_gpu()
    