"""
Todo: 
1. finish generate_initial_guess() function.
2. store random data samples offline. No need to sample random data online during the training.
"""

import random
import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import minimize, NonlinearConstraint
import torch
import param

class Leader:
    """
    Leader class deals with leader's utils.
    - obj_oc, obj_br, grad_obj_oc, grad_obj_br computes cost and gradient separately, for test purpose.
    """
    def __init__(self) -> None:
        self.dist_leader = param.dist_leader
        self.dimx, self.dimxA, self.dimxB = param.dimx, param.dimxA, param.dimxB
        self.dima, self.dimb = param.dima, param.dimb
        self.dt, self.Tf, self.dimT = param.dt, param.Tf, int(param.Tf / param.dt)
    
    def read_task_info(self, theta):
        """
        This function initializes the object variable to task theta.
        """
        self.theta = theta
        obs, p0A, p0B, pf, self.alp, _ = param.get_param_from_type(theta)
        self.obs_num = len(obs)
        self.obs = np.array(obs)
        self.p0 = np.array(p0A)
        self.pf = np.array(pf)
        self.x0 = np.array(p0A + p0B)       # concatenate initial state
        self.xf = np.array(pf + pf)         # concatenate terminal state
        self.compute_oc_cost_parameter()    # compute OC parameters

    def compute_oc_cost_parameter(self):
        """
        This function precomputes parameters in the leader's OC problem to facilitate computation.
        J_oc = (x_t-xf).T q1 (x_t-xf) + (d@x_t).T q2' (d@x_t) + a_t.T r a_t + (x_T-xf).T qf1 (x_T-xf) + (d@x_T).T qf2' (d@x_T)
             = (X-Xf).T Q1 (X-Xf) + X.T Q2 X + A.T R A
        Define q2 = d.T q2' d and q2f = d.T q2f' d
        Parameters includes system matrices (B1,B2), cost matrices (Q1,Q2,R,Xf,c).
        """
        # dynamics matrix
        self.B1 = self.dt*np.vstack( (np.eye(self.dima), np.zeros((self.dimxB, self.dima))) )
        self.B2 = self.dt*np.vstack( (np.zeros((self.dimxA, self.dimb)), np.eye(self.dimb)) )
        
        # oc cost matrix
        q1 = np.array([self.alp[0], self.alp[0], self.alp[1], self.alp[1]]) * np.eye(self.dimx)
        qf1 = self.alp[4] * np.eye(4)
        d = np.hstack((np.eye(self.dimxA), -np.eye(self.dimxB)))
        q2 = d.T @ ( self.alp[2]*np.eye(self.dimxA) ) @ d   # dimxA = dimxB = dimx / 2
        qf2 = d.T @ ( self.alp[4]*np.eye(self.dimxA) ) @ d
        r = self.alp[3] * np.eye(self.dima)
        
        # construct big matrix for oc cost
        Q1 = np.zeros((self.dimT*self.dimx, self.dimT*self.dimx))
        Q1[: self.dimx*(self.dimT-1), : self.dimx*(self.dimT-1)] = np.kron(np.eye(self.dimT-1), q1)
        Q1[self.dimx*(self.dimT-1): , self.dimx*(self.dimT-1) :] = qf1
        
        Q2 = np.zeros((self.dimT*self.dimx, self.dimT*self.dimx))
        Q2[: self.dimx*(self.dimT-1), : self.dimx*(self.dimT-1)] = np.kron(np.eye(self.dimT-1), q2)
        Q2[self.dimx*(self.dimT-1): , self.dimx*(self.dimT-1) :] = qf2
        
        self.Q1 = Q1
        self.Q2 = Q2
        self.R = np.kron(np.eye(self.dimT), r)
        self.Xf = np.kron(np.ones(self.dimT), self.xf)
        self.c = (self.x0 - self.xf) @ q1 @ (self.x0 - self.xf) + self.x0 @ q2 @ self.x0
        
        self.q1, self.q2, self.qf1, self.qf2, self.r = q1, q2, qf1, qf2, r

    def solve_oc(self, brnet):
        T = 10
        x_traj, a_traj = np.random.rand(T, self.dimx), np.random.rand(T-1, self.dima)
        return x_traj, a_traj

    def solve_oc1(self, brnet):
        """
        This function solves the leader's OC problem given the brnet. 
        Return the optimal x_traj, a_traj.
        """
        x_init_list, a_init_list = self.generate_initial_guess(N=2)                     # generate multiple initial guesses to solve
        N_guess = len(x_init_list)
        x_traj_list, a_traj_list, cost_list = [], [], []
        for i in range(N_guess):
            #x_traj, a_traj = pmp_solver(self, brnet, x_init_list[i], a_init_list[i])    # use PMP to find optimal trajectory
            x_traj, a_traj = self.opt_solver(brnet, x_init_list[i], a_init_list[i])     # directly formulate problem and use external solver
            
            x_traj_list.append(x_traj)
            a_traj_list.append(a_traj)
            cost_list.append(self.obj_oc(x_traj, a_traj))
        
        # find the optimal solution
        idx = np.argmin(np.array(cost_list))
        x_traj_opt, a_traj_opt = x_traj_list[idx], a_traj_list[idx]
        # cost_opt = cost_list[idx]
        return x_traj_opt, a_traj_opt
    
    def generate_initial_guess(self, N=10):
        """
        This function generates N trajestories as initial guesses to solve OC. 
        """
        x_traj_list, a_traj_list = [], []
        for i in range(N):
            # some methods to generates the initial guesses.
            x_traj, a_traj = np.random.rand(self.dimT+1, self.dimx), np.random.rand(self.dimT, self.dima)
            
            x_traj[0, : self.dimx] = self.x0      # always set the first element as x0
            x_traj_list.append(x_traj)
            a_traj_list.append(a_traj)
        return x_traj_list, a_traj_list

    def opt_solver(self, brnet, x_traj, a_traj):
        """
        This function calls scipy.optimize to solve the leader's OC problem.
        decision variable X = [x_1, ..., x_T, a_0, ..., a_{T-1}]
        """
        nonlincon = []
        x0 = x_traj[0, :]

        # input constraints: 0 <= |a_t| <= 1, t = 0,..., T-1
        def con_a(X):
            f = np.zeros(self.dimT)
            for i in range(self.dimT):
                i_a = self.dimT * self.dimx + i * self.dima         # starting inded for a_t
                f[i] = X[i_a: i_a+self.dima] @ X[i_a: i_a+self.dima]
            return f
        def jac_con_a(X):
            jac = np.zeros((self.dimT, self.dimT*(self.dima+self.dimx)))
            for i in range(self.dimT):
                i_a = self.dimT * self.dimx + i * self.dima         # starting inded for a_t
                jac[i, i_a: i_a+self.dima] = 2 * X[i_a: i_a+self.dima]
            return csr_matrix(jac)
        nonlincon.append( NonlinearConstraint(con_a, np.zeros(self.dimT), np.ones(self.dimT), jac=jac_con_a) )
        """
        for i in range(self.dimT):
            idx = self.dimT * self.dimx + i * self.dima
            con_a = lambda X: X[idx: idx+self.dima] @ X[idx: idx+self.dima]
            def jac_con_a(X):
                jac = np.zeros((1, self.dimT*(self.dimx+self.dima)))
                jac[0, idx: idx+self.dima] = 2 * X[idx: idx+self.dima]
                return csr_matrix(jac)
            nonlincon.append( NonlinearConstraint(con_a, 0., 1., jac=jac_con_a) )
        """

        # safe distance constraints: |pB_t-obs_j| <= d_j (only for pB not pA)
        def con_d(X):
            f = np.zeros(self.dimT*self.obs_num)
            for i in range(self.dimT):
                i_x = i * self.dimx             # starting inded for x_t
                for j in range(self.obs_num):
                    f[i*self.obs_num+j] = np.linalg.norm( X[i_x+self.dimxA: i_x+self.dimx] - self.obs[j, 0:2] )
            return f
        def jac_con_d(X):
            jac = np.zeros((self.dimT*self.obs_num, self.dimT*(self.dimx+self.dima)))
            for i in range(self.dimT):
                i_x = i * self.dimx             # starting inded for x_t
                for j in range(self.obs_num):
                    jac[i*self.obs_num+j, i_x+self.dimxA: i_x+self.dimx] = \
                        (X[i_x+self.dimxA: i_x+self.dimx] - self.obs[j, 0:2]) / np.linalg.norm( X[i_x+self.dimxA: i_x+self.dimx] - self.obs[j, 0:2] )
            return csr_matrix(jac)
        lb_d = np.kron(np.ones(self.dimT), 5*self.obs[:, -1])
        nonlincon.append( NonlinearConstraint(con_d, lb_d, np.inf*np.ones(self.dimT*self.obs_num), jac=jac_con_d) )
        """
        for i in range(self.dimT):
            N_obs = self.obs_num
            for j in range(N_obs):
                idx = i * self.dimx
                con_d = lambda X: np.linalg.norm( (X[idx+self.dimxA, idx+self.dimx]-self.obs[j, 0:2]) )
                def jac_con_d(X):
                    jac = np.zeros((1, self.dimT*(self.dimx+self.dima)))
                    jac[0, idx+self.dimxA: idx+self.dimx] = 0.5 * np.ones(self.dimxB) / np.linalg.norm( (X[idx+self.dimxA, idx+self.dimx]-self.obs[j, 0:2]) )
                    return csr_matrix(jac)
                nonlincon.append( NonlinearConstraint(con_d, self.obs[j, 2], np.inf, jac=jac_con_d) )
        """

        # dynamics constraints: x_tp1 = x_t + B1 a_t + B2 br(x_t, a_t), t = 0, ..., T-1
        def con_x(X):
            f = np.zeros(self.dimT*self.dimx)
            for i in range(self.dimT):
                i_x = i * self.dimx                             # starting index for x_t
                i_a = self.dimT * self.dimx + i * self.dima     # starting index for a_t
                if i == 0:
                    f[i_x: i_x+self.dimx] = -X[i_x: i_x+self.dimx] + x0 + self.B1@ X[i_a: i_a+self.dima] \
                        + self.B2 @ brnet(torch.from_numpy(x0).float(), torch.from_numpy(X[i_a: i_a+self.dima]).float()).detach().numpy().astype(float)
                else:
                    f[i_x: i_x+self.dimx] = -X[i_x: i_x+self.dimx] + X[i_x-self.dimx: i_x] + self.B1 @ X[i_a: i_a+self.dima] \
                    + self.B2 @ brnet(torch.from_numpy(X[i_x-self.dimx: i_x]).float(), torch.from_numpy(X[i_a: i_a+self.dima]).float()).detach().numpy().astype(float)
            return f
        def jac_con_x(X):
            jac = np.zeros((self.dimT*self.dimx, self.dimT*(self.dimx+self.dima)))
            for i in range(self.dimT):
                i_x = i * self.dimx                             # starting index for x_t
                i_a = self.dimT * self.dimx + i * self.dima     # starting index for a_t
                if i == 0:
                    jac[0: self.dimx, 0: self.dimx] = -np.eye(self.dimx)                    # df_0 / dx_1
                    _, jac_a = brnet.compute_input_jac(torch.from_numpy(x0).float(), torch.from_numpy(X[i_a: i_a+self.dima]).float())
                    jac[0: self.dimx, i_a: i_a+self.dima] = self.B1 + self.B2 @ jac_a.numpy().astype(float)                 # df_0 / da_0
                else:
                    jac[i_x: i_x+self.dimx, i_x: i_x+self.dimx] = -np.eye(self.dimx)        # df_t / dx_tp1
                    jac_x, jac_a = brnet.compute_input_jac(torch.from_numpy(x0).float(), torch.from_numpy(X[i_a: i_a+self.dima]).float())
                    jac[i_x: i_x+self.dimx, i_x-self.dimx: i_x] = np.eye(self.dimx) + self.B2 @ jac_x.numpy().astype(float) # df_t / dx_t
                    jac[i_x: i_x+self.dimx, i_a: i_a+self.dima] = self.B1 + self.B2 @ jac_a.numpy().astype(float)           # df_t / da_t
            return csr_matrix(jac)
        nonlincon.append( NonlinearConstraint(con_x, np.zeros(self.dimT*self.dimx), np.ones(self.dimT*self.dimx), jac=jac_con_x) )
        """
        for i in range(self.dimT):
            i_x = i * self.dimx                             # starting index for x_t
            i_a = self.dimT * self.dimx + i * self.dima     # starting index for a_t
            if i == 0:
                con_x = lambda X: -X[i_x: i_x+self.dimx] + x0 + self.B1@ X[i_a: i_a+self.dima] \
                    + self.B2 @ brnet(torch.from_numpy(x0).float, torch.from_numpy(X[i_a: i_a+self.dima]).float()).detach().numpy().astype(float)
                def jac_con_x(X):
                    jac = np.zeros(self.dimx, self.dimT*(self.dimx+self.dima))
                    jac[:, 0: self.dimx] = -np.eye(self.dimx)                                       # df_0 / dx_1
                    _, jac_a = brnet.compute_input_jac(torch.from_numpy(x0).float, torch.from_numpy(X[i_a: i_a+self.dima]).float())
                    jac[:, i_a: i_a+self.dima] = self.B1 + self.B2 * jac_a.numpy().astype(float)    # df_0 / da_0
                    return csr_matrix(jac)
            else:
                con_x = lambda X: -X[i_x: i_x+self.dimx] + X[idx-self.dimx: idx] + self.B1 @ X[i_a: i_a+self.dima] \
                    + self.B2 @ brnet(torch.from_numpy(X[idx-self.dimx: idx]).float(), torch.from_numpy(X[i_a: i_a+self.dima]).float()).detach().numpy().astype(float)
                def jac_con_x(X):
                    jac = np.zeros(self.dimx, self.dimT*(self.dimx+self.dima))
                    jac[:, i_x: i_x+self.dimx] = -np.eye(self.dimx)                                         # df_t / dx_tp1
                    jac_x, jac_a = brnet.compute_input_jac(torch.from_numpy(x0).float, torch.from_numpy(X[i_a: i_a+self.dima]).float())
                    jac[:, i_x-self.dimx: i_x] = np.eye(self.dimx) + self.B2 * jac_x.numpy().astype(float)  # df_t / dx_t
                    jac[:, i_a: i_a+self.dima] = self.B1 + self.B2 * jac_a.numpy().astype(float)            # df_t / da_t
                    return csr_matrix(jac)
            nonlincon.append( NonlinearConstraint(con_x, 0, 0, jac=jac_con_x) )
        """

        # objective
        def J(X):
            """
            objective function, J_oc = (X-Xf) @ Q1 @ (X-Xf) + x @ Q2 @ x + a @ R @ a
            """
            idx = self.dimT * self.dimx         # X[:idx] = [x_1,..., x_T], X[idx:] = [a_0, ..., a_{T-1}]
            cost = (X[: idx] - self.Xf) @ self.Q1 @ (X[: idx] - self.Xf) + X[: idx] @ self.Q2 @ X[: idx] + X[idx: ] @ self.R @ X[idx: ] + self.c
            return cost
            
        def dJ(X):
            grad = np.zeros(self.dimT * (self.dimx+self.dima))
            idx = self.dimT * self.dimx         # X[:idx] = [x_1,..., x_T], X[idx:] = [a_0, ..., a_{T-1}]
            grad[: idx] = 2*self.Q1 @ (X[: idx] - self.Xf) + 2*self.Q2 @ X[: idx]
            grad[idx: ] = 2*self.R @ X[idx: ]
            return grad

        X0 = np.concatenate((x_traj[1:, :].flatten(), a_traj.flatten()))    # get rid of x_traj[0, :] = x0
        result = minimize(J, X0, jac=dJ, constraints=nonlincon)

        # convert results to trajectories
        idx = self.dimT*self.dimx
        x_opt, a_opt = result.x[: idx], result.x[idx: ]
        x_opt = np.concatenate((x0, x_opt))     # add x0 to trajectory
        print(result.success, result.message)
        return x_opt.reshape((self.dimT+1, self.dimx)), a_opt.reshape((self.dimT, self.dima))


    def obj_oc(self, x_traj, a_traj):
        """
        This function computes the obj for leader's optimal control cost.
        """
        x0 = x_traj[0, :]
        traj_len = a_traj.shape[0]
        idx = self.dimx * traj_len
        X = np.concatenate( (x_traj[1:, :].flatten(), a_traj.flatten()) )
        cost = (X[: idx] - self.Xf) @ self.Q1 @ (X[: idx] - self.Xf) + X[: idx] @ self.Q2 @ X[: idx] + X[idx: ] @ self.R @ X[idx: ] + self.c
        return cost

    def obj_br(self, brnet, task_data):
        """
        This function computes the obj for leader's BR net training cost.
        task_data is in the list form [..., [x_t,a_t,b_t], ...]
        """
        N = len(task_data)
        x_traj, a_traj, b_traj = self.list2traj(task_data)
        #x_a, b = torch.from_numpy( np.hstack((x_traj, a_traj)) ).float(), torch.from_numpy(b_traj).float()
        x, a, b = torch.from_numpy(x_traj).float(), torch.from_numpy(a_traj).float(), torch.from_numpy(b_traj).float()
        br_cost_fn = torch.nn.MSELoss(reduction='sum')        
        #cost = br_cost_fn(brnet(x_a), b).item() / N
        cost = br_cost_fn(brnet(x, a), b). item() / N
        return cost

    def grad_obj_oc(self, brnet, a_traj):
        """
        This function computes gradient of obj for leader's optimal control w.r.t. brnet parameter w.
        The gradient stores in brnet and can be retrived by the method get_grad().
        """
        traj_len = a_traj.shape[0]
        x0, xf = torch.from_numpy(self.x0).float(), torch.from_numpy(self.xf).float()
        a_traj = torch.from_numpy(a_traj).float()
        B1, B2, q1, qf1, q2, qf2, r = self.convert_parameter_to_torch()
        cost = 0

        for t in range(traj_len):
            a_t = a_traj[t, :]
            if t == 0:
                x_t = x0
            else:
                #xa_t = torch.cat((x_t,a_t))
                #x_t = x_t + B1 @ a_t + B2 @ brnet(xa_t)
                x_t = x_t + B1 @ a_t + B2 @ brnet(x_t, a_t)     # update dynamics
            cost += (x_t-xf) @ q1 @ (x_t-xf) + x_t @ q2 @ x_t + a_t @ r @ a_t    # stage cost
        cost += (x_t-xf) @ qf1 @ (x_t-xf) + x_t @ qf2 @ x_t     # terminal cost
        
        brnet.zero_grad()
        cost.backward()
        return brnet
    
    def grad_obj_br(self, brnet, task_data):
        """
        This function computes gradient of obj for leader's BR net training.
        task_data is in the list form [..., [x_t,a_t,b_t], ...]
        The gradient stores in brnet and can be retrived by the method get_grad().
        """
        N = len(task_data)
        x_traj, a_traj, b_traj = self.list2traj(task_data)
        #x_a, b = torch.from_numpy( np.hstack((x_traj, a_traj)) ).float(), torch.from_numpy(b_traj).float()
        x, a, b = torch.from_numpy(x_traj).float(), torch.from_numpy(a_traj).float(), torch.from_numpy(b_traj).float()
        br_cost_fn = torch.nn.MSELoss(reduction='sum')        
        #cost = br_cost_fn(brnet(x_a), b) / N
        cost = br_cost_fn(brnet(x, a), b) / N

        brnet.zero_grad()   # clear gradient before computing the gradient
        cost.backward()
        return brnet

    def grad_obj_L(self, brnet, task_data, a_traj):
        """
        This function computes gradient of obj function L.
        Returns a brnet with same parameter (W,b) but updated gradient (dW, db).
        The updated gradient can be retrived by the method get_grad() or get_grad_dict().
        """
        cost = 0

        # step 1: formulate JA
        traj_len = a_traj.shape[0]
        a_traj = torch.from_numpy(a_traj).float()
        x0, xf = torch.from_numpy(self.x0).float(), torch.from_numpy(self.xf).float()
        B1, B2, q1, qf1, q2, qf2, r = self.convert_parameter_to_torch()
        for t in range(traj_len):
            a_t = a_traj[t, :]
            if t == 0:
                x_t = x0
            else:
                x_t = x_t + B1 @ a_t + B2 @ brnet(x_t, a_t)     # update dynamics
            cost += (x_t-xf) @ q1 @ (x_t-xf) + x_t @ q2 @ x_t + a_t @ r @ a_t    # stage cost
        cost += (x_t-xf) @ qf1 @ (x_t-xf) + x_t @ qf2 @ x_t     # terminal cost

        # step 2: formulate QA
        N = len(task_data)
        x_traj, a_traj, b_traj = self.list2traj(task_data)
        #x_a, b = torch.from_numpy( np.hstack((x_traj, a_traj)) ).float(), torch.from_numpy(b_traj).float()
        x, a, b = torch.from_numpy(x_traj).float(), torch.from_numpy(a_traj).float(), torch.from_numpy(b_traj).float()
        br_cost_fn = torch.nn.MSELoss(reduction='sum')        
        #cost += br_cost_fn(brnet(x_a), b) / N
        cost += br_cost_fn(brnet(x, a), b) / N
        
        # step 3: let L = JA + QA and find gradient
        brnet.zero_grad()   # clear gradient before computing the gradient
        cost.backward()
        return brnet
        
    def convert_parameter_to_torch(self):
        """
        This function convert path planning parameters to torch tensors to facilitate torch computation.
        """
        #B1, B2, Q1, D, Qf = self.compute_parameter_numpy()
        B1, B2 = torch.from_numpy(self.B1).float(), torch.from_numpy(self.B2).float()
        q1, qf1 = torch.from_numpy(self.q1).float(), torch.from_numpy(self.qf1).float()
        q2, qf2 = torch.from_numpy(self.q2).float(), torch.from_numpy(self.qf2).float()
        r = torch.from_numpy(self.r).float()
        return B1, B2, q1, qf1, q2, qf2, r


    def list2traj(self, data):
        """
        This function separate the list to individual trajectories. It exlcudes x_T in x_traj
        """
        traj_len = len(data)
        x_traj = np.zeros((traj_len, 4))
        a_traj = np.zeros((traj_len, 2))
        b_traj = np.zeros((traj_len, 2))
        for i in range(traj_len):
            x_traj[i, :] = data[i][0]
            a_traj[i, :] = data[i][1]
            b_traj[i, :] = data[i][2]
        return x_traj, a_traj, b_traj


class Follower:
    """
    Follower class deals with follower's utils
    """
    def __init__(self, theta) -> None:
        self.theta = theta
        self.dt = param.dt
        obs, _, p0, pf, _, self.beta = param.get_param_from_type(theta)
        #self.obs, _, self.p0, self.pf, _, self.beta = param.get_param_from_type(theta)
        
        self.obs_num = len(obs)
        self.obs = np.array(obs)
        self.p0 = np.array(p0)
        self.pf = np.array(pf)

    def dynamics(self, x_t, a_t, b_t):
        """
        This function computes x_{t+1} = f(x_t, a_t, b_t).
        """
        x_tp1 = x_t + np.concatenate((a_t, b_t)) * self.dt
        return x_tp1
    
    def is_safe(self, p):
        """
        This function check whether the current pB is safe or not (within the safety region).
        """
        flag = True
        for i in range(self.obs_num):
            obs_i = self.obs[i, :]
            q_i, d_i = obs_i[0: 2], obs_i[-1]   # get obs position and safe distance
            dist = np.linalg.norm(p - q_i)
            if dist <= d_i:
                flag = False
                break
        return flag

    def obs_cost(self, p):
        """
        This function computes obstacle cost: \sum_i f(p, obs_i)
        """
        cost = 0
        for i in range(self.obs_num):
            obs_i = self.obs[i, :]
            q_i, d_i = obs_i[0: 2], obs_i[-1]   # get obs position and safe distance
            dist = np.linalg.norm(p - q_i)
            if dist < d_i:
                cost -= np.log(dist / d_i)
        return cost
    
    def grad_b_obs_cost(self, p):
        """
        This function computes gradient of obs cost w.r.t to b: grad_b \sum_i f(p, obs_i)
        """
        dcost = np.zeros(2)
        for i in range(self.obs_num):
            obs_i = self.obs[i, :]
            q_i, d_i = obs_i[0: 2], obs_i[-1]   # get obs position and safe distance
            dist = np.linalg.norm(p - q_i)
            if dist < d_i:
                dcost -= (d_i/dist) * ((p-q_i)/dist) @ (self.dt*np.eye(2))  # derivative related to dynamics
        return dcost
    
    def compute_obj(self, x, a, b):
        """
        This function comptues the follower's obj.
        """
        x_new = self.dynamics(x, a, b)
        pA_tp1 = x_new[0: 2]
        pB_tp1 = x_new[2: ]
        J = self.beta[0] * (pB_tp1 - self.pf) @ (pB_tp1 - self.pf) + self.beta[1] * (pA_tp1 - pB_tp1) @ (pA_tp1 - pB_tp1)
        + self.beta[2] * b @ b + self.beta[3] * self.obs_cost(pB_tp1)
        return J

    def get_br(self, x, a):
        """
        This function computes the follower's best response given x,a
        """
        def J(b, x,a):  # compute objective function, b is decision variable
            x_new = self.dynamics(x, a, b)
            pA_tp1, pB_tp1 = x_new[0: 2], x_new[2: ]
            J = self.beta[0] * (pB_tp1 - self.pf) @ (pB_tp1 - self.pf) + self.beta[1] * (pA_tp1 - pB_tp1) @ (pA_tp1 - pB_tp1)
            + self.beta[2] * b @ b + self.beta[3] * self.obs_cost(pB_tp1)
            return J
        
        def dJ(b, x,a): # compute grad of objective function, b is decision variable
            x_new = self.dynamics(x, a, b)
            pA_tp1, pB_tp1 = x_new[0: 2], x_new[2: ]
            dJ = 2*self.beta[0] * (pB_tp1 - self.pf) @ (self.dt*np.eye(2)) + 2*self.beta[1] * (pA_tp1 - pB_tp1) @ (-self.dt*np.eye(2))
            + 2*self.beta[2] * b + 2*self.beta[3] * self.grad_b_obs_cost(pB_tp1)
            return dJ

        nc1 = NonlinearConstraint((lambda b: b@b) , 0, 1, jac=(lambda b: 2*b.reshape(1,2)))     # 0 <= b@b <= 1
        b0 = np.random.rand(2)

        result = minimize(J, b0, args=(x,a), jac=dJ, constraints=[nc1])
        return result.x


class BRNet(torch.nn.Module):
    """
    BRNet class defines and trains a NN for BR model.
    """
    def __init__(self) -> None:
        super(BRNet, self).__init__()
        self.dimx = param.dimx
        self.dima = param.dima
        self.dimb = param.dimb
        self.device = 'cpu'

        self.linear1 = torch.nn.Linear(self.dimx+self.dima, 10)
        self.linear2 = torch.nn.Linear(10, 10)
        self.linear3 = torch.nn.Linear(10, self.dimb)
        self.activation = torch.nn.ReLU()

        self.linear1.weight.data.fill_(1)
        self.linear1.bias.data.fill_(1)
        self.linear2.weight.data.fill_(1)
        self.linear2.bias.data.fill_(1)
        self.linear3.weight.data.fill_(1)
        self.linear3.bias.data.fill_(1)
    """ 
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        return x
    """

    def forward(self, x, a):
        if x.ndim > 1:
            y = torch.cat((x, a), dim=1)
        else:
            y = torch.cat((x, a), dim=0)
        y = self.linear1(y)
        y = self.activation(y)
        y = self.linear2(y)
        y = self.activation(y)
        y = self.linear3(y)
        return y
    
    def compute_input_jac(self, x, a):
        """
        This function computes the jacobian of brnet w.r.t. input x and a.
        """
        # register hook for inner layer outpuut
        y = []  # y[i] is a 2d array
        def forward_hook(model, input, output):
            y.append( output.detach() )
        h1 = self.linear1.register_forward_hook(forward_hook)
        h2 = self.linear2.register_forward_hook(forward_hook)
        h3 = self.linear3.register_forward_hook(forward_hook)
        _ = self.forward(x, a)
        h1.remove()
        h2.remove()
        h3.remove()

        def d_activation(y):
            """
            This function computes derivative of activation functions. can be relu, tanh, sigmoid.
            Input is a 1d array, output is nxn matrix.
            """
            #df = torch.diag(1 - torch.tanh(y**2))  # for tanh(x)
            y[y < 0] = 0; df = torch.diag(y)        # for relu(x)
            return df
        p = self.get_data_dict()
        jac_x = p['linear3.weight'] @ d_activation(y[1]) @ p['linear2.weight'] @ d_activation(y[0]) @ p['linear1.weight'][:, : self.dimx]
        jac_a = p['linear3.weight'] @ d_activation(y[1]) @ p['linear2.weight'] @ d_activation(y[0]) @ p['linear1.weight'][:, self.dimx: ]
        return jac_x, jac_a
    
    def get_data(self):
        """
        This function gets the NN parameters as a dictionary. data format use torch
        """
        W = {}
        b = {}
        W["linear1"] = self.linear1.weight.data.detach().cpu()
        W["linear2"] = self.linear2.weight.data.detach().cpu()
        W["linear3"] = self.linear3.weight.data.detach().cpu()

        b["linear1"] = self.linear1.bias.data.detach().cpu()
        b["linear2"] = self.linear2.bias.data.detach().cpu()
        b["linear3"] = self.linear3.bias.data.detach().cpu()
        return W, b
    
    def get_grad(self):
        """
        This function gets the gradient of NN parameters as a dictionary. data format use torch
        """
        dW = {}
        db = {}
        dW["linear1"] = self.linear1.weight.grad.detach().cpu()
        dW["linear2"] = self.linear2.weight.grad.detach().cpu()
        dW["linear3"] = self.linear3.weight.grad.detach().cpu()

        db["linear1"] = self.linear1.bias.grad.detach().cpu()
        db["linear2"] = self.linear2.bias.grad.detach().cpu()
        db["linear3"] = self.linear3.bias.grad.detach().cpu()
        return dW, db

    def set_data(self, W, b):
        """
        This function sets the NN parameter to the given dictionary.
        """
        self.linear1.weight.data = W["linear1"]
        self.linear2.weight.data = W["linear2"]
        self.linear3.weight.data = W["linear3"]

        self.linear1.bias.data = b["linear1"]
        self.linear2.bias.data = b["linear2"]
        self.linear3.bias.data = b["linear3"]
    
    def get_data_dict(self):
        p_dict = {} 
        for n, p in self.named_parameters():    # order of iteration: (W1, b1), (W2, b2), ...
            p_dict[n] = p.detach().cpu()
        return p_dict
    
    def get_grad_dict(self):
        dp_dict = {}
        dp_dict["linear1.weight"] = self.linear1.weight.grad.detach().cpu()
        dp_dict["linear2.weight"] = self.linear2.weight.grad.detach().cpu()
        dp_dict["linear3.weight"] = self.linear3.weight.grad.detach().cpu()
        dp_dict["linear1.bias"] = self.linear1.bias.grad.detach().cpu()
        dp_dict["linear2.bias"] = self.linear2.bias.grad.detach().cpu()
        dp_dict["linear3.bias"] = self.linear3.bias.grad.detach().cpu()
        return dp_dict


class Meta:
    """
    Meta class implements meta learning algorithm.
    """
    def __init__(self):
        self.task_num = param.total_type
        self.task_pdf = param.type_pdf
        self.ws_len = param.ws_len
        self.kappa = param.kappa
        self.lr = param.lr
        self.lr_meta = param.lr_meta
        self.mu = param.mu
    
    def sample_tasks(self, K):
        """
        This function samples K tasks according to given task distribution
        """
        task = np.arange(self.task_num)
        task_sample = np.random.choice(task, K, p=self.task_pdf)
        return task_sample
    
    def sample_task_theta(self, follower, x_traj, a_traj, N=1000):
        """
        This function samples N points for individual tasks theta based on ration kappa.
        D = [D_traj, D_uniform], D_traj / D_uniform = kappa.
        """
        D_traj = self.sample_task_theta_traj(follower, x_traj, a_traj, N=1000)
        D_uniform = self.sample_task_theta_uniform(follower, N=1000)    # ??? or we can store random data offline and sample from them
        
        # use the ratio and shuffle the data
        n_uniform = N // (self.kappa+1)
        n_traj = N - n_uniform
        D_traj = random.choices(D_traj, k=n_traj)
        D_uniform = random.choices(D_uniform, k=n_uniform)
        task_data = D_traj + D_uniform
        random.shuffle(task_data)
        return task_data

    def sample_task_theta_uniform(self, follower, N=1000):
        """
        This function uniformly samples N points for the individual task theta.
        BR data points [x,a,br(x,a)] 
        """
        task_data = []
        for i in range(N):
            # sample pA, pB in the square working space
            pA = np.random.rand(2) * self.ws_len
            while True:
                pB = np.random.rand(2) * self.ws_len
                if follower.is_safe(pB):
                    break
            x = np.concatenate((pA,pB))
            # sample a
            a = np.random.rand(2)

            br = follower.get_br(x, a)
            task_data.append([x, a, br])
        return task_data
    
    def sample_task_theta_traj(self, follower, x_traj, a_traj, N=None):
        """
        This function samples br points for the individual task theta given x_traj and a_traj.
        BR data points is sampled at (x_t+noise, a+noise).
        parameter N determines the number of br samples.
        """
        mu, sigma = 0, 0.1
        traj_len = a_traj.shape[0]
        task_data = []

        if N is None:
            task_data_len = traj_len
            for i in range(task_data_len):
                x = x_traj[i, :] + np.random.normal(mu, sigma, 4)
                a = a_traj[i, :] + np.random.normal(mu, sigma, 2)
                br = follower.get_br(x, a)
                task_data.append([x, a, br])
        else:
            task_data_len = N
            for i in range(task_data_len):
                # randomly select one (x_t, a_t) from trajectory
                j = np.random.choice(traj_len)
                x = x_traj[j, :] + np.random.normal(mu, sigma, 4)
                a = a_traj[j, :] + np.random.normal(mu, sigma, 2)
                br = follower.get_br(x, a)
                task_data.append([x, a, br])
        return task_data

    def update_model(self, leader, brnet, task_sample, br_list, D2_list):
        """
        This function updates the entire brnet based on GD (with momentum).
        """
        brnet.zero_grad()   # clear gradient to compute accumulated gradient
        W, b = brnet.get_data()
        dW, db = brnet.get_grad()

        for theta in task_sample:
            leader.read_task_info(theta)
            _, a_traj = leader.solve_oc(br_list[theta])
            brnet1 = leader.grad_obj_L(br_list[theta], D2_list[theta], a_traj)
            
            # add accumulated gradient
            dW1, db1 = brnet1.get_grad()
            dW['linear1'] += dW1['linear1']
            dW['linear2'] += dW1['linear2']
            dW['linear3'] += dW1['linear3']
            db['linear1'] += db1['linear1']
            db['linear2'] += db1['linear2']
            db['linear3'] += db1['linear3']
        
        # perform one-step SGD with momentum
        W, b = self.one_step_sgd_momentum(W, b, dW, db, self.mu, self.lr_meta)
        brnet.set_data(W, b)
        return brnet

    def update_model_theta(self, leader, brnet, task_data, a_traj):
        """
        This function updates the brnet for task theta based on GD (with momentum).
        Return a brnet1 with updated parameter.
        """
        brnet1 = leader.grad_obj_L(brnet, task_data, a_traj)

        W, b = brnet.get_data() # in fact, no need for brnet to get W and b. same values in brnet1
        dW, db = brnet1.get_grad()
        W, b = self.one_step_sgd_momentum(W, b, dW, db, self.mu, self.lr)
        brnet1.set_data(W, b)

        """# new functions 
        # use (W,b) in brnet and (DW,db) in brnet1 to update (W,b) in brnet1
        dp = brnet1.get_grad_dict()
        with torch.no_grad():       # use no_grad to mannually update NN parameter
            for n, p in brnet1.named_parameters():
                p -= self.lr * (self.mu * p + dp[n])
        """

        return brnet1

    def one_step_sgd_momentum(self, W, b, dW, db, mu, lr):
        """
        This function performs one-step SGD with momentum given the parameter, learning rate, momentum.
        Code snippet for model update.
        """
        Wtmp, btmp = {}, {}
        if mu > 0:
            Wtmp['linear1'] = mu * W['linear1'] + dW['linear1']
            Wtmp['linear2'] = mu * W['linear2'] + dW['linear2']
            Wtmp['linear3'] = mu * W['linear3'] + dW['linear3']
            btmp['linear1'] = mu * b['linear1'] + db['linear1']
            btmp['linear2'] = mu * b['linear2'] + db['linear2']
            btmp['linear3'] = mu * b['linear3'] + db['linear3']
        else:
            Wtmp, btmp = dW, db
        
        #W = W - self.lr * Wtmp, b = b - self.lr * btmp
        W['linear1'] = W['linear1'] - lr * Wtmp['linear1']
        W['linear2'] = W['linear2'] - lr * Wtmp['linear2']
        W['linear3'] = W['linear3'] - lr * Wtmp['linear3']
        b['linear1'] = b['linear1'] - lr * btmp['linear1']
        b['linear2'] = b['linear2'] - lr * btmp['linear2']
        b['linear3'] = b['linear3'] - lr * btmp['linear3']
        return W, b
        