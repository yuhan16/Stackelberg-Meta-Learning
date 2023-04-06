"""
use large NN trained with 5 types of data to do sg-adaptation directly.
"""
import numpy as np
from utils import Leader, Follower, Meta, BRNet
import torch


def large_nn_train():
    """
    Train a NN use all five types of data
    """
    import time, os

    def generate_batch_data(data, batch_size):
        """
        This function generates batch data for training and testing.
        """
        batch_data = []
        ITER = data.shape[0] // batch_size
        x, ua, ub = data[:, :leader.dimx], data[:, leader.dimx: leader.dimx+leader.dimua], data[:, leader.dimx+leader.dimua:]
        x, ua, ub = leader.to_torch(x), leader.to_torch(ua), leader.to_torch(ub)
        #x, ua, ub = x.to(device), ua.to(device), ub.to(device)
        for i in range(ITER):
            idx = i*batch_size
            batch_data.append([x[idx: idx+batch_size,:], ua[idx: idx+batch_size,:], ub[idx: idx+batch_size,:]])
        return batch_data

    device = 'cuda:3'
    leader = Leader()
    leader.meta = Meta()
    br = BRNet()#.to(device)
    br_data_rand = []
    br_data_obs = []
    for i in range(leader.meta.total_type):
        fname = 'data/br_data/f' + str(i) + '_rand.npy'
        br_data_rand.append( np.load(fname) )
        leader.rng.shuffle(br_data_rand[i])
        br_data_rand[i] = br_data_rand[i]
        
        fname = 'data/br_data/f' + str(i) + '_obs.npy'
        br_data_obs.append( np.load(fname) )
        leader.rng.shuffle(br_data_obs[i])
        br_data_obs[i] = br_data_obs[i]    
    
    # select data
    N_train = 15000
    N_test = 1000
    batch_size = 100
    D_train = []
    D_test = []
    for i in range(leader.meta.total_type):
        N_theta = N_train // leader.meta.total_type
        D_train.append( leader.meta.sample_task_theta(i, br_data_rand[i], br_data_obs[i], N=N_theta))
        leader.rng.shuffle(D_train[i])
    D_train = np.vstack(D_train)
    leader.rng.shuffle(D_train)
    
    # train neural network
    lr = 1e-4
    mom = 0.8
    epoch = 10000
    loss_fn = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(br.parameters(), lr=lr, momentum=mom)
    #optimizer = torch.optim.Adam(br.parameters(), lr=0.0001)
    ep_train = []
    ep_test = []
    start_t = time.time()
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
                #print('iter: {}/{}, loss: {:.7f}'.format(i+1, len(batch_data), loss.item()*(leader.dimx+leader.dimua)))
                #print('iter: {}/{}, loss: {:.7f}'.format(i+1, len(batch_data), loss.item()/batch_size))
        
        ep_train.append(epoch_train_loss)
    end_t = time.time()
    print("Elapsed time: {:.3f} min.".format((end_t-start_t)/60))
    # plot epoch loss for visualization
    ep_train = np.array(ep_train)
    #ep_test = np.array(ep_test)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(ep_train.mean(axis=1), label='ave train')
    #ax.plot(ep_test.mean(axis=1), label='ave test')
    ax.set_title('training loss for whole')
    ax.legend()
    #fig.savefig('tmp/tmp7.png')
    plt.close(fig)

    # save individual response model
    if not os.path.exists('data/whole_data/'):
        os.mkdir('data/whole_data/')
    fname = 'data/whole_data/br_whole.pth'
    #torch.save(br.state_dict(), fname)
    #np.save('data/whole_data/ep_train.npy', ep_train)
    #np.save('data/whole_data/ep_test'+str(theta)+'.npy', ep_test)


def adapt(theta):
    leader = Leader()
    leader.meta = Meta()
    br = BRNet()
    br.load_state_dict(torch.load('data/whole_data/br_whole.pth'))
    D_theta = np.load('data/gdtest'+str(theta)+'.npy')
    D_theta = D_theta[:1000, :]
    x, ua, ub = D_theta[:, :leader.dimx], D_theta[:, leader.dimx: leader.dimx+leader.dimua], D_theta[:, leader.dimx+leader.dimua:]
    x, ua, ub = leader.to_torch(x), leader.to_torch(ua), leader.to_torch(ub)
    #loss_fn = torch.nn.MSELoss()
    loss_fn = torch.nn.MSELoss(reduction='mean')
    lr = 1e-4
    mom = 0.6
    optimizer = torch.optim.SGD(br.parameters(), lr=lr, momentum=mom)
    ITER = 50
    tmp = []
    for i in range(ITER):
        loss = loss_fn(br(x,ua), ub) * (leader.dimx+leader.dimua)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tmp.append(loss.item())
    print(tmp)
    return tmp


if __name__ == '__main__':
    large_nn_train()
    #adapt(4)