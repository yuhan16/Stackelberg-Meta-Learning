import os
import time
import numpy as np
import torch
from sg_meta.agent import Leader
from sg_meta.meta import Meta
from sg_meta.model import BRNet
from sg_meta.utils import save_results
from sg_meta import train_data_dir


def sg_meta():
    """
    This function implements the meta-learning algorithm to find the generalized br model.
    """
    torch.set_default_dtype(torch.float64)
    leader = Leader()
    leader.meta = Meta()
    br_data_rand, br_data_obs = [], []
    for i in range(leader.meta.total_type):
        fname = os.path.join(train_data_dir, 'br_data', f'f{i}_rand.npy')
        br_data_rand.append( np.load(fname) )
        br_data_rand[i] = br_data_rand[i][:10000,:]     # or use full data 
        
        fname = os.path.join(train_data_dir, 'br_data', f'f{i}_obs.npy')
        br_data_obs.append( np.load(fname) )
        br_data_obs[i] = br_data_obs[i][:5000,:]        # or use full data

    brnet = BRNet()
    br = brnet.state_dict()     # update state_dict in the training
    mom_dict = brnet.get_zero_grad_dict()   # record momentum in SGD (outer loop)
    meta_loss_list = []
    
    start_t = time.time()
    for iter in range(leader.meta.n_iter):
        br_list = []
        D_train_list, D_test_list = [], []
        task_sample = leader.meta.sample_tasks(leader.meta.total_type)
        #task_sample = [0]   # for testing

        for i in range(len(task_sample)):
            theta = task_sample[i]
            data_theta_rand = br_data_rand[theta]
            data_theta_obs = br_data_obs[theta]
            
            # sample D_train for follower theta
            D_train = leader.meta.sample_task_theta(theta, data_theta_rand, data_theta_obs, N=leader.meta.n_data_samp)
            D_train_list.append(D_train)
            
            # inner loop update
            br_theta = leader.meta.update_br_theta(br, D_train)
            br_list.append( br_theta )
            
            # sample D_test for follower theta
            D_test = leader.meta.sample_task_theta(theta, data_theta_rand, data_theta_obs, N=leader.meta.n_data_samp)
            D_test_list.append(D_test)
        
        # outer loop learning, manually perform SGD
        br = leader.meta.update_br_meta(task_sample, br, br_list, D_test_list, mom_dict)

        # or use soft warmup for choosing beta
        beta_tmp = 5e-5 + 0.5*(5e-4 - 5e-5) * (1 + np.cos(0.5*iter/100))
        leader.meta.beta = beta_tmp

        # compute meta loss and print things
        meta_loss = np.zeros(len(task_sample))
        for i in range(len(task_sample)):
            meta_loss[i] = leader.meta.compute_meta_cost_theta(br, D_test_list[i])
        meta_loss_list.append(meta_loss.mean())
        
        if iter % 1000 == 0:
            print(f'Meta iter {iter+1}/{leader.meta.n_iter}, meta loss: {meta_loss.mean():.5f}')

    end_t = time.time()
    print(f'Meta-training completed. Training time: {(end_t-start_t)/60:.3f} min.')
    
    # save results
    save_flag = True
    if save_flag:
        dname = os.path.join(train_data_dir, 'meta_data')
        res = {
            'meta_br.pth': br,
            'meta_loss.npy': np.array(meta_loss_list),
        }
        save_results(dname, res)


def sg_adapt(theta):
    """
    This function implements the adaptation algorithm to find the adapted br model for type theta follower.
    """
    torch.set_default_dtype(torch.float64)
    leader = Leader()
    leader.meta = Meta()
    
    # load data and meta model
    fname = os.path.join(train_data_dir, 'br_data', f'f{theta}_rand.npy')
    br_data_rand = np.load(fname)
    fname = os.path.join(train_data_dir, 'br_data', f'f{theta}_obs.npy')
    br_data_obs = np.load(fname)
    br = BRNet()
    fname = os.path.join(train_data_dir, 'meta_data', 'meta_br.pth')
    br.load_state_dict(torch.load(fname))

    D_theta = leader.meta.sample_task_theta(theta, br_data_rand, br_data_obs, N=leader.meta.n_data_samp_adapt)
    x, ua, ub = D_theta[:, :leader.dimx], D_theta[:, leader.dimx: leader.dimx+leader.dimua], D_theta[:, leader.dimx+leader.dimua:]
    x, ua, ub = leader.to_torch(x), leader.to_torch(ua), leader.to_torch(ub)

    loss_fn = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(br.parameters(), lr=leader.meta.alp_adapt, momentum=leader.meta.mom_adapt)
    adapt_loss = []
    
    # perform adaptation
    for i in range(leader.meta.n_adapt_iter):
        loss = loss_fn(br(x,ua), ub) *(leader.dimx+leader.dimua)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        adapt_loss.append(loss.item())
    
    # save results
    save_flag = True
    if save_flag:
        dname = os.path.join(train_data_dir, 'meta_data')
        res = {
            f'adapt{theta}_loss.npy': np.array(adapt_loss),
            f'adapt{theta}.pth': br.state_dict(),
        }
        save_results(dname, res)



if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)

    sg_meta()

    theta = 4   # [0, ..., 4]
    sg_adapt(theta)