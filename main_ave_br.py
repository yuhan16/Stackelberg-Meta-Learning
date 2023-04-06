"""
Average over parameter space and do sg-adaptation.
"""
import numpy as np
from utils import Leader, Follower, Meta, BRNet
import torch


def adapt(theta, br_dict):
    leader = Leader()
    leader.meta = Meta()
    br = BRNet()
    br.load_state_dict(br_dict)
    br_data_rand = np.load('data/br_data/f' + str(theta) + '_rand.npy')
    br_data_obs = np.load('data/br_data/f' + str(theta) + '_obs.npy')

    D_theta = leader.meta.sample_task_theta(theta, br_data_rand, br_data_obs, N=1000)
    D_theta = np.load('data/gdtest'+str(theta)+'.npy')
    D_theta = D_theta[:1000, :]
    x, ua, ub = D_theta[:, :leader.dimx], D_theta[:, leader.dimx: leader.dimx+leader.dimua], D_theta[:, leader.dimx+leader.dimua:]
    x, ua, ub = leader.to_torch(x), leader.to_torch(ua), leader.to_torch(ub)
    #loss_fn = torch.nn.MSELoss()
    loss_fn = torch.nn.MSELoss(reduction='mean')
    lr = 1e-4
    mom = 0.5
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
    

def ave_model():
    leader = Leader()
    leader.meta = Meta()
    br = BRNet()
    br_dict_list = []
    for i in range(leader.meta.total_type):
        fname = 'data/indiv_data/br' + str(i) + '.pth'
        br_dict_list.append( torch.load(fname) )

    # get an empty br list
    ave = {}
    for key in br_dict_list[0].keys():
        ave[key] = torch.zeros_like( br_dict_list[0][key].cpu() )
    
    # average the param dict, according to type pdf
    for key in ave.keys():
        for i in range(leader.meta.total_type):
            ave[key] += leader.meta.type_pdf[i] * br_dict_list[i][key].cpu()
    
    #print(tmp)
    br.load_state_dict(ave)
    print(br)
    print(br(torch.rand(5), torch.rand(2)))
    return ave


def main():
    theta = 4
    ave_dict = ave_model()
    return adapt(theta, ave_dict)


if __name__ == '__main__':
    import os
    if not os.path.exists('data/indiv_data'):
        os.mkdir('data/indiv_data')
    main()