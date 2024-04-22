import os
import time
import numpy as np
import torch
from sg_meta.agent import Leader
from sg_meta.model import BRNet
from sg_meta.meta import Meta
from sg_meta.utils import save_results
from sg_meta import train_data_dir


def ave_output_model():
    """
    Average the output space to obtain the meta model. Mix all types of data.
    """
    fname = os.path.join(train_data_dir, 'ave_output', 'ave_br.pth')
    if not os.path.exists(fname):
        # train model and specify learning parameters
        param = {
            'learning_rate': 2.5e-4,
            'momentum': 0.8,    # used for SGD training
            'n_epoch': 40000,
            'batch_size': 100,
            'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
            'n_train':  15000,  # number of training data
            'n_test': 1000,     # number of testing data
        }
        train_model_mix(param)
    
    return torch.load(fname, map_location='cpu')


def ave_output_adapt(theta, ave_dict):
    leader = Leader()
    leader.meta = Meta()
   
    # load data and model
    br_data_rand = np.load( os.path.join(train_data_dir, 'br_data', f'f{theta}_rand.npy') )
    br_data_obs = np.load( os.path.join(train_data_dir, 'br_data', f'f{theta}_obs.npy') )
    br = BRNet()
    br.load_state_dict(ave_dict)

    # use same adaptation parameter as meta learning
    N_data = leader.meta.n_data_samp_adapt
    ITER = leader.meta.n_adapt_iter
    lr = leader.meta.alp_adapt
    mom = leader.meta.mom_adapt

    D_theta = leader.meta.sample_task_theta(theta, br_data_rand, br_data_obs, N=N_data)
    x, ua, ub = D_theta[:, :leader.dimx], D_theta[:, leader.dimx: leader.dimx+leader.dimua], D_theta[:, leader.dimx+leader.dimua:]
    x, ua, ub = leader.to_torch(x), leader.to_torch(ua), leader.to_torch(ub)
    loss_fn = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(br.parameters(), lr=lr, momentum=mom)
    ave_adapt_loss = []

    # perform adaptation
    for i in range(ITER):
        loss = loss_fn(br(x,ua), ub) * (leader.dimx+leader.dimua)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ave_adapt_loss.append(loss.item())
    
    save_flag = True
    if save_flag:
        dname = os.path.join(train_data_dir, 'ave_output')
        res = {
            f'adapt{theta}_loss.npy': np.array(ave_adapt_loss),
            f'adapt{theta}.pth': br.state_dict(),
        }
        save_results(dname, res)


def train_model_mix(param):
    """
    Train a NN use mixed data of all five types of follower's data.
    param specifies: learning rate, momentum, epoch_size, batch_size, device, training data number, testing data number.
    """
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

    # parse parameters
    lr = param['learning_rate']
    mom = param['momentum']
    epoch = param['n_epoch']
    batch_size = param['batch_size']
    device = param['device']
    N_train = param['n_train']
    N_test = param['n_test']

    leader = Leader()
    leader.meta = Meta()
    br = BRNet().to(device)

    # get full data
    br_data_rand = []
    br_data_obs = []
    for i in range(leader.meta.total_type):
        fname = os.path.join(train_data_dir, 'br_data', f'f{i}_rand.npy')
        br_data_rand.append( np.load(fname) )
        fname = os.path.join(train_data_dir, 'br_data', f'f{i}_obs.npy')
        br_data_obs.append( np.load(fname) )

    # sample data from all types of followers    
    D_train = []
    D_test = []
    for i in range(leader.meta.total_type):
        N_theta = N_train // leader.meta.total_type
        D_train.append( leader.meta.sample_task_theta(i, br_data_rand[i], br_data_obs[i], N=N_theta) )
        
        N_theta = N_test // leader.meta.total_type
        D_test.append( leader.meta.sample_task_theta(i, br_data_rand[i], br_data_obs[i], N=N_theta) )
    D_train = np.vstack(D_train)
    D_test = np.vstack(D_test)
    
    loss_fn = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(br.parameters(), lr=lr, momentum=mom)
    #optimizer = torch.optim.Adam(br.parameters(), lr=0.0001)
    ep_train = []
    ep_test = []

    start_t = time.time()
    print('Training brnet with mixed data...\n------------')
    for t in range(epoch):
        print("Epoch: {}\n-------------".format(t+1))
        epoch_train_loss = []

        # shuffle training data in each epoch
        leader.rng.shuffle(D_train)
        batch_data = generate_batch_data(D_train, batch_size)
        for i in range(len(batch_data)):
            x, ua, ub = batch_data[i][0], batch_data[i][1], batch_data[i][2]
            loss = loss_fn(br(x,ua), ub) *(leader.dimx+leader.dimua)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss.append(loss.item())
            if (i+1) % 50 == 0:
                print('iter: {}/{}, loss: {:.7f}'.format(i+1, len(batch_data), loss.item()))
        ep_train.append(epoch_train_loss)
        
        # for testing
        epoch_test_loss = []
        leader.rng.shuffle(D_test)
        batch_test_data = generate_batch_data(D_test, batch_size)
        for i in range(len(batch_test_data)):
            x, ua, ub = batch_test_data[i][0], batch_test_data[i][1], batch_test_data[i][2]
            with torch.no_grad():
                loss = loss_fn(br(x,ua), ub) *(leader.dimx+leader.dimua)
                epoch_test_loss.append(loss.item())
        ep_test.append(epoch_test_loss)

    end_t = time.time()
    print(f'Elapsed time: {(end_t-start_t)/60:.3f} min.')

    # save individual response model
    save_flag = True
    if save_flag:
        dname = os.path.join(train_data_dir, 'ave_output')
        res = {
            'ave_br.pth': br.state_dict(),
            'ep_train_ave.npy': np.array(ep_train),
            'ep_test_ave.npy': np.array(ep_test),
        }
        save_results(dname, res)



if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    
    theta = 4   # [0,...,4]

    ave_dict = ave_output_model()
    ave_output_adapt(theta, ave_dict)
