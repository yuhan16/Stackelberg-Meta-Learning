import os
import time
import numpy as np
import torch
from sg_meta.agent import Leader
from sg_meta.model import BRNet
from sg_meta.meta import Meta
from sg_meta.utils import save_results
from sg_meta import train_data_dir


def ave_param_model():
    '''
    Average over parameter space to obtain the meta model.
    '''
    leader = Leader()
    leader.meta = Meta()
    
    br_dict_list = []
    for i in range(leader.meta.total_type):
        br_dict_list.append( get_indiv_br_model(i) )

    # get an empty br_dict list
    ave = {}
    for key in br_dict_list[0].keys():
        ave[key] = torch.zeros_like( br_dict_list[0][key] )
    
    # average the param dict, according to type pdf
    for key in ave.keys():
        for i in range(leader.meta.total_type):
            ave[key] += leader.meta.type_pdf[i] * br_dict_list[i][key]
    
    #br = BRNet()
    #br.load_state_dict(ave)
    #print(br)
    #print(br(torch.rand(5), torch.rand(2)))
    return ave  


def ave_param_adapt(theta, ave_dict):
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

    # do adaptation
    for i in range(ITER):
        loss = loss_fn(br(x,ua), ub) * (leader.dimx+leader.dimua)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ave_adapt_loss.append(loss.item())

    save_flag = True
    if save_flag:
        dname = os.path.join(train_data_dir, 'ave_param')
        res = {
            f'adapt{theta}_loss.npy': np.array(ave_adapt_loss),
            f'adapt{theta}.pth': br.state_dict(),
        }
        save_results(dname, res)
    

def get_indiv_br_model(theta):
    fname = os.path.join(train_data_dir, 'indiv_data', f'br{theta}.pth')
    if not os.path.exists(fname):
        # train individual model and specify learning parameters
        param = {
            'learning_rate': 2.5e-4,
            'momentum': 0.8,    # used for SGD training
            'n_epoch': 40000,
            'batch_size': 100,
            'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
            'n_train': 15000,  # number of training data
            'n_test': 1000,     # number of testing data
        }
        train_model_theta(theta, param)
    
    return torch.load(fname, map_location='cpu')      # return state_dict, load to cpu


def train_model_theta(theta, param):
    """
    This function trains an individual br model for different type theta of followers.
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
    
    # prepare training and testing data sets
    br_data_rand = np.load( os.path.join(train_data_dir, f'br_data/f{theta}_rand.npy') )
    br_data_obs = np.load( os.path.join(train_data_dir, f'br_data/f{theta}_obs.npy') )
    D_train = leader.meta.sample_task_theta(theta, br_data_rand, br_data_obs, N=N_train)
    D_test = leader.meta.sample_task_theta(theta, br_data_rand, br_data_obs, N=N_test)

    loss_fn = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(br.parameters(), lr=lr, momentum=mom)
    #optimizer = torch.optim.Adam(br.parameters(), lr=0.0001)
    ep_train = []
    ep_test = []

    start_t = time.time()
    print('Training individual brnet for follower {}...\n------------'.format(theta))
    for t in range(epoch):
        print("Epoch: {}\n-------------".format(t+1))
        epoch_train_loss = []
        
        leader.rng.shuffle(D_train)         # shuffle training data in each epoch
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

    save_flag = True
    if save_flag:
        dname = os.path.join(train_data_dir, 'indiv_data')
        res = {
            f'ep_train{theta}.npy': np.array(ep_train),
            f'ep_test{theta}.npy': np.array(ep_test),
            f'br{theta}.pth': br.state_dict(),
        }
        save_results(dname, res)



if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)

    theta = 4   # [0,...,4]

    ave_dict = ave_param_model()
    ave_param_adapt(theta, ave_dict)
