import os
import json
import numpy as np
import torch
from .agent import Leader, Follower
from .model import BRNet
from . import config_data_dir


class Meta(Leader):
    """
    This function defines meta-learning related functions. Inherited from the Leader class
    """
    def __init__(self) -> None:
        super().__init__()
        fname = os.path.join(config_data_dir, 'parameters.json')
        param = json.load(open(fname))

        #self.seed = param['seed']
        #self.rng = np.random.default_rng(self.seed)     # or use None
        
        self.total_type = param['total_type']
        self.type_pdf = param['type_pdf']

        #self.dimx, self.dimxa, self.dimxb = param['dimx'], param['dimxa'], param['dimxb']
        #self.dimua, self.dimub = param['dimua'], param['dimub']
        #self.ws_len, self.obs, self.pd = param['ws_length'], param['obstacle_settings'], param['destination']

        # meta learning parameters
        self.alp = param['alp']         # inner loop learning rate
        self.mom = param['moment']      # momentum for sgd
        self.beta = param['beta']       # outer loop (meta) learning rate
        self.kappa = param['kappa']     # D_rand_num / D_obs_num ratio
        self.n_iter = param['n_iter']   # meta learning steps
        self.n_data_samp = param['n_data_samp']     # number of sampled data in each meta training

        # adaptation parameters
        self.alp_adapt = param['alp_adapt']         # GD step for adaptation
        self.mom_adapt = param['moment_adapt']      # momentum for adaptation
        self.n_adapt_iter = param['n_iter_adapt']   # adaptation steps
        self.n_data_samp_adapt = param['n_data_samp_adapt']     # number of sampled data for adaptation


    def sample_tasks(self, N):
        """
        This function samples N types of followers using the type distribution.
        """
        return self.rng.choice(self.total_type, N, p=self.type_pdf)
    

    def sample_task_theta(self, theta, data_rand, data_obs, N=10):
        """
        This function samples N br data points for type theta follower.
        br data point: [x, ua, br(x,ua)]
        D = [D_rand, D_obs], |D_rand| / |D_obs| = kappa.
        """
        n_obs = N // (self.kappa+1)
        n_rand = N - n_obs
        if n_rand > data_rand.shape[0] or n_obs > data_obs.shape[0]:
            raise Exception("Required data size larger than the existing data set.")
        
        D_r = self.sample_task_theta_random(theta, data_rand, N=n_rand)
        D_o = self.sample_task_theta_obs(theta, data_obs, N=n_obs)
        task_data = np.vstack( (D_r, D_o) )
        self.rng.shuffle(task_data)
        
        return task_data
        
    def sample_task_theta_random(self, theta, data=[], N=10):
        """
        This function samples br data from given data set or generates new br data using randomly generated x_traj and a_traj.
        """
        task_data = []
        if len(data) == 0:   # generate new data
            rng = np.random.default_rng(seed=None)  # self rng or new rng
            follower = Follower(theta)
            for i in range(N):
                # sample pa, pb in the square working space but not in side the obstacle
                # pb should be within certain distance of pa. large divication makes no sense.
                while True:
                    pa = rng.random(self.dimxa) * self.ws_len
                    if self.is_safe(pa):
                        break
                while True:
                    pb = rng.normal(pa, scale=1.5, size=(self.dimxb-1))     # pb is near pa, control scale
                    #pb = rng.random(self.dimxb-1) * self.ws_len
                    if follower.is_safe(pb):
                        break
                phib = (rng.random(1)-0.5) * np.pi      # phib in [-pi, pi]
                x = np.concatenate((pa, pb, phib))
               
                ua = rng.random(self.dimua) * 2 - 1     # sample ua, |ua| <= 1
                if np.linalg.norm(ua) > 1:
                    ua = ua / np.linalg.norm(ua)

                br = follower.get_br(x, ua)
                task_data.append( np.concatenate((x, ua, br)) )
            task_data = np.vstack(task_data)     # task_data[i, :] = [x, a, br]
        else:   # sample from data set
            if data.shape[0] < N:
                raise Exception("Sample size greater than the given data size. Change sample size N.")
            idx = self.rng.choice(data.shape[0], N, replace=False)
            task_data = data[idx, :]
        
        return task_data
    

    def sample_task_theta_obs(self, theta, data=[], N=10):
        """
        This function samples br data near the obstacle. Sample the region {p: rc <= |Lam (p-p_obs)| <= 2*rc}
        """
        task_data = []
        if len(data) == 0:  # generate new data
            rng = np.random.default_rng(seed=None)  # self rng or new rng
            follower = Follower(theta)
            for i in range(len(self.obs)):
                obs_i = self.obs[i]
                xc, yc, rc = obs_i[0], obs_i[1], obs_i[2]
                x_scale, y_scale = obs_i[4], obs_i[5]
                Lam = np.diag([1/x_scale, 1/y_scale])
                data_obs = []
                for j in range( int(np.ceil(N/len(self.obs))) ):
                    # sample pa, pb in |Lam (p-p_obs)| <= 2*rc
                    while True:     # find pa
                        pa = rng.random(self.dimxa) * (2*rc*np.array([x_scale, y_scale])) + np.array([xc, yc])
                        f1 = Lam @ (pa - np.array([xc, yc]))
                        if obs_i[3] == 1:
                            f2 = np.linalg.norm(f1, ord=1)
                        elif obs_i[3] == 2:
                            f2 = np.linalg.norm(f1, ord=2)
                        else:
                            f2 = np.linalg.norm(f1, ord=np.inf)
                        if self.is_safe(pa) and f2 <= 2*rc:
                            break
                    while True:     # find pb
                        pb = rng.normal(pa, scale=1, size=(self.dimxb-1))     # pb is near pa, control scale
                        f1 = Lam @ (pb - np.array([xc, yc]))
                        if obs_i[3] == 1:
                            f2 = np.linalg.norm(f1, ord=1)
                        elif obs_i[3] == 2:
                            f2 = np.linalg.norm(f1, ord=2)
                        else:
                            f2 = np.linalg.norm(f1, ord=np.inf)
                        if self.is_safe(pb) and f2 <= 2*rc:
                            break
                    phib = (rng.random(1)-0.5) * np.pi      # phib in [-pi, pi]
                    x = np.concatenate((pa, pb, phib))
                    ua = rng.random(self.dimua) * 2 - 1     # sample ua, |ua| <= 1
                    if np.linalg.norm(ua) > 1:
                        ua = ua / np.linalg.norm(ua)
                    br = follower.get_br(x, ua)
                    data_obs.append( np.concatenate((x, ua, br)) )
                task_data.append( np.vstack(data_obs) )
            task_data = np.vstack(task_data)
            task_data[rng.choice(task_data.shape[0], N, replace=False), :]    # task_data size may exceed N due to ceil()
        else: 
            if data.shape[0] < N:
                raise Exception("Sample size greater than the given data size. Change sample size N.")
            idx = self.rng.choice(data.shape[0], N, replace=False)
            task_data = data[idx, :]
            
        return task_data


    def sample_task_theta_traj(self, theta, x_traj, a_traj, data=[], N=10):
        """
        This function samples br data near the given x_traj and a_traj.
        """
        if len(data) == 0:
            raise Exception("Data set empty for sampling near the trajectory.")
        if data.shape[0] < N:
            raise Exception("Sample size greater than the given data size. Change sample size N.")
        
        task_data = []
        for t in range(self.dimT):
            x_t, a_t = x_traj[t], a_traj[t]

            # sample k nearest (measured by l2 norm) br data in the data set
            k = N // 2 #10
            tmp = data[:, :self.dimx+self.dimua] - np.kron( np.ones((data.shape[0],1)), np.concatenate((x_t,a_t)) )
            tmp = np.linalg.norm(tmp, axis=1)
            if tmp.shape[0] > k:
                idx = np.argsort(tmp)[: k]
                task_data.append( data[idx, :] )
            else:
                task_data.append(data)
        task_data = np.vstack(task_data)

        # sample N br data from task_data
        if task_data.shape[0] < N:
            raise Exception("Sampled data less than the requirement. Change k.")
        idx = self.rng.choice(task_data.shape[0], N, replace=False)
        task_data = data[idx, :]

        return task_data
    

    def update_br_theta(self, br_dict, data):
        """
        This function updates the inner meta-learning problem for type theta follower.
        Perform one-step GD using given data. The momentum in SGD is useless for one-step GD
        """
        x, ua, ub = data[:, :self.dimx], data[:, self.dimx: self.dimx+self.dimua], data[:, self.dimx+self.dimua:]
        x, ua, ub = self.to_torch(x), self.to_torch(ua), self.to_torch(ub)
        br = BRNet()
        br.load_state_dict(br_dict)

        loss_fn = torch.nn.MSELoss(reduction='mean')
        loss = loss_fn(br(x,ua), ub) *(self.dimx+self.dimua)
        optimizer = torch.optim.SGD(br.parameters(), lr=self.alp, momentum=self.mom)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return br.state_dict()
    

    def update_br_meta(self, task_sample, br_k_dict, br_in_list, data_list, mom_dict):
        """
        This function updates the outer meta-learning problem.
        br_in_list contains the intermediate br parameters for different theta (according to task_sample).
        """
        def compute_accumulated_grad_dict(grad_list):
            """
            This function computes the accumulated gradient for brnet given a gradient list.
            """
            br = BRNet()
            dp = br.get_zero_grad_dict()
            for i in range(len(grad_list)):
                for key in dp.keys():
                    dp[key] += grad_list[i][key]
            return dp
        
        grad_list = []
        loss_fn = torch.nn.MSELoss(reduction='mean')
        for i in range(len(task_sample)):
            theta = task_sample[i]
            data_theta = data_list[i]
            br_in_theta = br_in_list[i]

            # compute accumulated gradient
            x, ua, ub = data_theta[:, :self.dimx], data_theta[:, self.dimx: self.dimx+self.dimua], data_theta[:, self.dimx+self.dimua:]
            x, ua, ub = self.to_torch(x), self.to_torch(ua), self.to_torch(ub)
            br = BRNet()
            br.load_state_dict(br_in_theta)
            loss = loss_fn(br(x,ua), ub) *(self.dimx+self.dimua)
            loss.backward()
            grad_list.append( br.get_grad_dict() )
        
        # one-step gradient update with sgd
        dp = compute_accumulated_grad_dict(grad_list)
        br_kp1_dict = br_k_dict.copy()
        for key in br_kp1_dict.keys():
            mom_dict[key] = self.mom * mom_dict[key] + (dp[key]/len(task_sample))       # update momentum
            br_kp1_dict[key] = br_k_dict[key] - self.beta * mom_dict[key]

        return br_kp1_dict   # No need to return mom_dict because python changes dictionary directly.


    def compute_meta_cost_theta(self, br_dict, data):
        """
        This function computes the meta cost given data and trained brnet.
        """
        x, ua, ub = data[:, :self.dimx], data[:, self.dimx: self.dimx+self.dimua], data[:, self.dimx+self.dimua:]
        x, ua, ub = self.to_torch(x), self.to_torch(ua), self.to_torch(ub)
        br = BRNet()
        br.load_state_dict(br_dict)
        loss_fn = torch.nn.MSELoss(reduction='mean')
        loss = loss_fn(br(x,ua), ub) *(self.dimx+self.dimua)
        
        return loss.item()

