import os
import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
import torch
import matplotlib.pyplot as plt
import param


class Leader:
    """
    leader's objective, solve OC, generate initial guesses, etc.
    """
    def __init__(self) -> None:
        self.dimx, self.dimxA, self.dimxB = param.dimx, param.dimxA, param.dimxB
        self.dima, self.dimb = param.dima, param.dimb
        self.dt, self.Tf, self.dimT = param.dt, param.Tf, int(param.Tf / param.dt)
        self.mu, self.nu = param.mu, param.nu
        self.gam = param.gam
    
    def read_task_info(self, scn, theta):
        """
        This function initializes the object variable to scenario scn and task theta.
        """
        self.theta = theta
        self.scn = scn
        obs, x0A, x0B, pf, q, qf, _ = param.get_param(scn, theta)
        self.obs_num = len(obs)
        self.obs = np.array(obs)
        self.pf = np.array(pf)
        self.x0 = np.array(x0A + x0B)           # concatenate initial state
        self.xf = np.array(pf + pf + [0])       # concatenate terminal state
        self.compute_cost_parameter(q, qf)      # compute OC parameters
    
    def compute_cost_parameter(self, q, qf):
        """
        This function precomputes parameters in the leader's OC problem to facilitate computation.
        J_oc = (x_t-xf).T q1 (x_t-xf) + (d@x_t).T q2' (d@x_t) + a_t.T r a_t + (x_T-xf).T qf1 (x_T-xf) + (d@x_T).T qf2' (d@x_T)
             = (X-Xf).T Q1 (X-Xf) + X.T Q2 X + A.T R A
        Define q2 = d.T q2' d and qf2 = d.T qf2' d
        Parameters includes system matrices (B1,B2), cost matrices (Q1,Q2,R,Xf,c).
        """
        # dynamics matrix: x_tp1 = x_t + B1@a_t + B2@b_t + A(x_t)@b_t
        self.B1 = self.dt*np.vstack( (np.eye(self.dima), np.zeros((self.dimxB, self.dima))) )
        self.B2 = self.dt*np.vstack( (np.zeros((self.dimxA, self.dimb)), np.array([[0,0],[0,0],[0,1.]])) )
        
        # cost matrix
        self.q1, self.qf1 = np.diag(q[0]), np.diag(qf[0])
        d = np.hstack( (np.eye(2), -np.eye(2), np.zeros((2,1))) )   # extract pA - pB = d @ x
        self.q2, self.qf2 = d.T @ np.diag(q[1]) @ d, d.T @ np.diag(qf[1]) @ d
        self.q3 = q[2]  # no use
        self.r = np.diag(q[3])

        # construct big matrix for oc cost
        Q1 = np.zeros((self.dimT*self.dimx, self.dimT*self.dimx))
        Q1[: self.dimx*(self.dimT-1), : self.dimx*(self.dimT-1)] = np.kron(np.eye(self.dimT-1), self.q1)
        Q1[self.dimx*(self.dimT-1): , self.dimx*(self.dimT-1) :] = self.qf1
        
        Q2 = np.zeros((self.dimT*self.dimx, self.dimT*self.dimx))
        Q2[: self.dimx*(self.dimT-1), : self.dimx*(self.dimT-1)] = np.kron(np.eye(self.dimT-1), self.q2)
        Q2[self.dimx*(self.dimT-1): , self.dimx*(self.dimT-1) :] = self.qf2
        
        self.Q1 = Q1
        self.Q2 = Q2
        self.R = np.kron(np.eye(self.dimT), self.r)
        self.Xf = np.kron(np.ones(self.dimT), self.xf)
        self.c = (self.x0 - self.xf) @ self.q1 @ (self.x0 - self.xf) + self.x0 @ self.q2 @ self.x0

    def dynamics(self, x_t, a_t, b_t):
        """
        This function computes x_{t+1} = f(x_t, a_t, b_t).
        """
        xB_t = x_t[self.dimxA: ]
        dot_xB_t = np.array( [np.cos(xB_t[-1])*b_t[0], np.sin(xB_t[-1])*b_t[0], b_t[1]] ) 
        x_tp1 = x_t + np.concatenate((a_t, dot_xB_t)) * self.dt
        return x_tp1
    
    def generate_initial_guess(self, brnet):
        """
        This function generates random initial guesses to solve OC.
        """
        x_1, a_1 = np.random.rand(self.dimT+1, self.dimx), np.random.rand(self.dimT, self.dima)
        dx = (self.xf - self.x0) / self.dimT
        x_1[0, :] = self.x0      # always set the first element as x0
        Ax = lambda x: self.dt * np.array([[np.cos(x[-1]),0], [np.sin(x[-1]),0], [0,1]])
        
        for i in range(self.dimT):
            x_1[i+1, :] = x_1[i, :] + dx
            a_1[i, :] = (x_1[i+1, 0:self.dimxA] - x_1[i, 0:self.dimxA]) / self.dt   # a_t = (pA_tp1 - pA_t) / dt
            x_1[i+1, self.dimxA:] = x_1[i, self.dimxA:] + Ax(x_1[i,:]) @ self.get_b(brnet, x_1[i,:], a_1[i,:]) # find xB from xA and a
        
        # add randomness to the first trajectory to obtain new initial guesses
        x_traj, a_traj = x_1.copy(), a_1.copy()
        add_random = True
        if add_random is True:
            x_traj = x_1 + 0.2 * np.random.standard_normal(x_1.shape)
            a_traj = a_1 + 0.2 * np.random.standard_normal(a_1.shape)
        x_traj[0, :], x_traj[-1, :] = self.x0, self.xf      # always set the first as x0 and the last as xf
        
        return x_traj, a_traj 

    def project_trajectory(self, x_init, a_init):
        """
        This function 
        1. projects the outside trajectory to the square working space 
        2. projects path trajectory outside the obstacle
        3. projects input trajectory within feasible input region
        """
        ws = param.ws_len
        x_traj = x_init.copy()
        idx = np.argwhere(x_traj[:, 0] < 0) # for px^A
        idx = np.reshape(idx, idx.size)
        x_traj[idx, 0] = 0
        idx = np.argwhere(x_traj[:, 0] > ws)
        idx = np.reshape(idx, idx.size)
        x_traj[idx, 0] = ws

        idx = np.argwhere(x_traj[:, 1] < 0) # for py^A
        idx = np.reshape(idx, idx.size)
        x_traj[idx, 1] = 0
        idx = np.argwhere(x_traj[:, 1] > ws)
        idx = np.reshape(idx, idx.size)
        x_traj[idx, 1] = ws

        idx = np.argwhere(x_traj[:, 2] < 0); idx = idx.reshape(idx.size); x_traj[idx, 2] = 0    # for px^B
        idx = np.argwhere(x_traj[:, 2] > ws); idx = idx.reshape(idx.size); x_traj[idx, 2] = ws
        idx = np.argwhere(x_traj[:, 3] < 0); idx = idx.reshape(idx.size); x_traj[idx, 3] = 0    # for py^B
        idx = np.argwhere(x_traj[:, 3] > ws); idx = idx.reshape(idx.size); x_traj[idx, 3] = ws
            
        # project path trajectory outside the obstacle
        for t in range(self.dimT+1):
            pA_t = x_traj[t, :self.dimxA]
            pB_t = x_traj[t, self.dimxA:-1]
            for j in range(self.obs_num):
                q_j, d_j = self.obs[j, :2], self.obs[j, -1]
                eps = 1e-1      # for numerical reasons
                if np.linalg.norm(pA_t-q_j) <= d_j:
                    pA_new = (d_j+eps) * (pA_t-q_j) / np.linalg.norm(pA_t-q_j) + q_j
                    x_traj[t, :self.dimxA] = pA_new
                if np.linalg.norm(pB_t-q_j) <= d_j:
                    pB_new = (d_j+eps) * (pB_t-q_j) / np.linalg.norm(pB_t-q_j) + q_j
                    x_traj[t, self.dimxA:-1] = pB_new
    
        # project input trajectory to feasible input regions
        a_traj = a_init.copy()
        idx = np.argwhere(np.linalg.norm(a_traj, axis=1) > 1); idx = idx.reshape(idx.size)
        a_traj[idx, :] = ( a_traj[idx, :].T / np.linalg.norm(a_traj[idx, :], axis=1) ).T
        
        return x_traj, a_traj

    def solve_oc(self, brnet, x_init=None, a_init=None):
        """
        This function solves the leader's optimal control problem given a parameterized brnet.
        """
        if x_init is None or a_init is None:
            x_init, a_init = self.generate_initial_guess(brnet)
            x_init, a_init = self.project_trajectory(x_init, a_init)    # ensure feasibility of initial guess
        else:
            x_init, a_init = self.project_trajectory(x_init, a_init)    # ensure feasibility of initial guess

        # solve leader's OC problem with optimization and PMP
        x_opt, a_opt = self.opt_solver(brnet, x_init, a_init)          # directly formulate problem and use opt solver
        x_pmp, a_pmp = self.pmp_solver(brnet, x_opt, a_opt)             # use PMP to refine the trajectory
        #x_traj, a_traj = self.pmp_solver(brnet, x_init, a_init)        # directly use PMP to find the trajectory
        
        print("opt_traj = {}, pmp_traj = {}".format(self.obj_oc(x_opt, a_opt), self.obj_oc(x_pmp, a_pmp)))
        if self.obj_oc(x_opt, a_opt) >= self.obj_oc(x_pmp, a_pmp):
            x_traj, a_traj = x_pmp, a_pmp
        else:
            x_traj, a_traj = x_opt, a_opt

        #  determine whether to accept the current trajectory or not
        if self.obj_oc(x_traj, a_traj) >= self.obj_oc(x_init, a_init):
            x_traj, a_traj = x_init, a_init
            print("New OC trajectory worse than initial one. Use initial trajectory.")
        return x_traj, a_traj
    
    def opt_solver(self, brnet, x_traj, a_traj):
        """
        This function implements optimization approach to solve the leader's problem.
        The follower's dynamic constraints are penalized in the objective. Only leader's constraints preserve.
        decision var in optimization: X = [x_1, ..., x_T, a_0, ..., a_{T-1}]
        - x_traj = [x_0, ..., x_T]
        - a_traj = [a_0, ..., a_{T-1}]
        """
        constr = []
        x0 = x_traj[0, :]

        # input constraints: 0 <= |a_t| <= 1, t = 0,..., T-1
        def con_a(X):
            f = np.zeros(self.dimT)
            for i in range(self.dimT):
                i_a = self.dimT * self.dimx + i * self.dima         # starting inded for a_t
                a_t = X[i_a: i_a+self.dima]
                f[i] = a_t @ a_t
            return f
        def jac_con_a(X):
            jac = np.zeros((self.dimT, self.dimT*(self.dima+self.dimx)))
            for i in range(self.dimT):
                i_a = self.dimT * self.dimx + i * self.dima         # starting inded for a_t
                a_t = X[i_a: i_a+self.dima]
                jac[i, i_a: i_a+self.dima] = 2 * a_t
            return csr_matrix(jac)
        constr.append( NonlinearConstraint(con_a, np.zeros(self.dimT), np.ones(self.dimT), jac=jac_con_a) )

        # safe constraints for leader: |pA_t-obs_j| <= d_j (only for pA)
        def con_dA(X):
            f = np.zeros(self.dimT*self.obs_num)
            for i in range(self.dimT):
                i_x = i * self.dimx             # starting inded for x_t
                pA_t = X[i_x: i_x+self.dimxA]
                for j in range(self.obs_num):
                    f[i*self.obs_num+j] = np.linalg.norm( pA_t - self.obs[j, 0:2] )
            return f
        def jac_con_dA(X):
            jac = np.zeros((self.dimT*self.obs_num, self.dimT*(self.dimx+self.dima)))
            for i in range(self.dimT):
                i_x = i * self.dimx             # starting inded for x_t
                pA_t = X[i_x: i_x+self.dimxA]
                for j in range(self.obs_num):
                    jac[i*self.obs_num+j, i_x: i_x+self.dimxA] = \
                        (pA_t - self.obs[j, 0:2]) / np.linalg.norm( pA_t - self.obs[j, 0:2] )
            return csr_matrix(jac)

        # safe constraints for follower: |pB_t-obs_j| <= d_j (only for pB)
        def con_dB(X):
            f = np.zeros(self.dimT*self.obs_num)
            for i in range(self.dimT):
                i_x = i * self.dimx             # starting inded for x_t
                pB_t = X[i_x+self.dimxA: i_x+self.dimx-1]
                for j in range(self.obs_num):
                    f[i*self.obs_num+j] = np.linalg.norm( pB_t - self.obs[j, 0:2] )
            return f
        def jac_con_dB(X):
            jac = np.zeros((self.dimT*self.obs_num, self.dimT*(self.dimx+self.dima)))
            for i in range(self.dimT):
                i_x = i * self.dimx             # starting inded for x_t
                pB_t = X[i_x+self.dimxA: i_x+self.dimx-1]
                for j in range(self.obs_num):
                    jac[i*self.obs_num+j, i_x+self.dimxA: i_x+self.dimx-1] = \
                        (pB_t - self.obs[j, 0:2]) / np.linalg.norm( pB_t - self.obs[j, 0:2] )
            return csr_matrix(jac)
        if self.obs_num > 0:
            eps = 1e-1  # for numerical reasons
            lb_d = np.kron(np.ones(self.dimT), self.obs[:, -1]+eps)
            constr.append( NonlinearConstraint(con_dA, lb_d, np.inf*np.ones(self.dimT*self.obs_num), jac=jac_con_dA) )
            constr.append( NonlinearConstraint(con_dB, lb_d, np.inf*np.ones(self.dimT*self.obs_num), jac=jac_con_dB) )

        # dynamic constraints ONLY LEADER: xA_tp1 = xA_t + B1 @ a_t, linear constraints
        Aeq, beq = np.zeros((self.dimT*self.dimxA, self.dimT*(self.dimx+self.dima))), np.zeros(self.dimT*self.dimxA)
        BB1 = self.dt * np.eye(self.dima)
        for i in range(self.dimT):
            i_p = i * self.dimxA
            i_x = i * self.dimx
            i_a = i * self.dima + self.dimT*self.dimx
            if i == 0:
                beq[i_p: i_p+self.dimxA] = -x0[0:self.dimxA]
                Aeq[i_p: i_p+self.dimxA, i_x: i_x+self.dimxA] = -np.eye(self.dimxA)             # for x_1
                Aeq[i_p: i_p+self.dimxA, i_a: i_a+self.dima] = BB1      # for a_0
            else:
                Aeq[i_p: i_p+self.dimxA, i_x: i_x+self.dimxA] = -np.eye(self.dimxA)             # for x_tp1
                Aeq[i_p: i_p+self.dimxA, i_x-self.dimx: i_x-self.dimxB] = np.eye(self.dimxA)    # for x_t
                Aeq[i_p: i_p+self.dimxA, i_a: i_a+self.dima] = BB1      # for a_t
        lc1 = LinearConstraint(Aeq, beq, beq)
        constr.append( lc1 )
        
        # objective
        def J(X, mu):
            """
            objective function, J_oc = (X-Xf) @ Q1 @ (X-Xf) + x @ Q2 @ x + a @ R @ a + mu*|-xB_tp1 + xB_t + A(xB_t)br(x_t, a_t)|
            X[:idx] = [x_1,..., x_T], X[idx:] = [a_0, ..., a_{T-1}]
            """
            idx = self.dimT * self.dimx
            cost = (X[: idx] - self.Xf) @ self.Q1 @ (X[: idx] - self.Xf) + X[: idx] @ self.Q2 @ X[: idx] + X[idx: ] @ self.R @ X[idx: ] + self.c
            
            # soft dynamic constraint for xB
            #cost=0
            fB = lambda x: self.dt * np.array([[np.cos(x[-1]),0], [np.sin(x[-1]),0], [0,1]])
            D = np.hstack( (np.zeros((self.dimxB, self.dimxA)), np.eye(self.dimxB)) )             # D@x_t = xB_t
            for i in range(self.dimT):
                i_x = i * self.dimx
                i_a = self.dimT * self.dimx + i * self.dima
                x_tp1 = X[i_x: i_x+self.dimx]
                a_t = X[i_a: i_a+self.dima]
                if i == 0:
                    dif = -D@x_tp1 + D@x0 + fB(x0) @ self.get_b(brnet, x0, a_t)
                else:
                    x_t = X[i_x-self.dimx: i_x]
                    dif = -D@x_tp1 + D@x_t + fB(x_t) @ self.get_b(brnet, x_t, a_t)
                cost += mu * (dif @ dif)
            
            """
            # safety cost for xA
            #cost = 0
            for i in range(self.dimT):
                i_x = i * self.dimx             # starting inded for x_t
                xA_t = X[i_x: i_x+self.dimxA]
                eps = 1e-1                      # for numerical reasons
                for j in range(self.obs_num):
                    cost += (-nu)* np.log( np.linalg.norm(xA_t-self.obs[j, 0:2]) - self.obs[j,-1] - eps )

            # safety cost for xB
            #cost=0
            for i in range(self.dimT):
                i_x = i * self.dimx             # starting inded for x_t
                xB_t = X[i_x+self.dimxA: i_x+self.dimx-1]
                eps = 1e-1                      # for numerical reasons
                for j in range(self.obs_num):
                    cost += (-nu)* np.log( np.linalg.norm(xB_t-self.obs[j, 0:2]) - self.obs[j,-1] - eps )
            """

            return cost     
        def dJ(X, mu):
            grad = np.zeros(self.dimT * (self.dimx+self.dima))
            idx = self.dimT * self.dimx
            grad[: idx] = 2*self.Q1 @ (X[: idx] - self.Xf) + 2*self.Q2 @ X[: idx]
            grad[idx: ] = 2*self.R @ X[idx: ]
            
            # gradient of soft dynamic constraint for xB
            #grad = np.zeros(self.dimT * (self.dimx+self.dima))
            fB = lambda x: self.dt * np.array([[np.cos(x[-1]),0], [np.sin(x[-1]),0], [0,1]])
            D = np.hstack( (np.zeros((self.dimxB, self.dimxA)), np.eye(self.dimxB)) )             # D@x_t = xB_t
            for i in range(self.dimT):
                i_x = i * self.dimx
                i_a = self.dimT * self.dimx + i * self.dima
                x_tp1 = X[i_x: i_x+self.dimx]
                a_t = X[i_a: i_a+self.dima]
                if i == 0:
                    dif = -D@x_tp1 + D@x0 + fB(x0) @ self.get_b(brnet, x0, a_t)
                    jac_x, jac_a = self.get_b_jac(brnet, x0, a_t)        
                    grad[i_x: i_x+self.dimx] += 2*mu * dif @ (-D)                   # df_0/dx_1, no df_0/dx_0
                    grad[i_a: i_a+self.dima] += 2*mu * dif @ fB(x0) @ jac_a         # df_0/da_0
                else:
                    x_t = X[i_x-self.dimx: i_x]
                    dif = -D@x_tp1 + D@x_t + fB(x_t) @ self.get_b(brnet, x_t, a_t)
                    b_t = self.get_b(brnet, x_t, a_t)
                    jac_x, jac_a = self.get_b_jac(brnet, x_t, a_t)
                    grad[i_x-self.dimx: i_x] += 2*mu * dif @  ( D + fB(x_t) @ jac_x \
                        + self.dt*np.array([[0,0,0,0,-np.sin(x_t[-1])*b_t[0]], [0,0,0,0,np.cos(x_t[-1])*b_t[0]], self.dimx*[0]]) ) # df_t/dx_t
                    grad[i_x: i_x+self.dimx] += 2*mu * dif @ (-D)                   # df_t/dx_tp1
                    grad[i_a: i_a+self.dima] += 2*mu * dif @ fB(x_t) @ jac_a        # df_t/da_t
            
            """
            #gradient of soft safety constraint for xA
            grad = np.zeros(self.dimT * (self.dimx+self.dima))
            for i in range(self.dimT):
               i_x = i * self.dimx             # starting inded for x_t
               xA_t = X[i_x: i_x+self.dimxA]
               eps = 1e-1
               for j in range(self.obs_num):
                   grad[i_x: i_x+self.dimxA] += (-nu) / (np.linalg.norm(xA_t-self.obs[j, 0:2]) - self.obs[j,-1] - eps) \
                       * (xA_t - self.obs[j, 0:2]) / np.linalg.norm( xA_t - self.obs[j, 0:2] )

            #gradient of safety cost for xB
            grad = np.zeros(self.dimT * (self.dimx+self.dima))
            for i in range(self.dimT):
               i_x = i * self.dimx             # starting inded for x_t
               xB_t = X[i_x+self.dimxA: i_x+self.dimx-1]
               eps = 1e-1
               for j in range(self.obs_num):
                   grad[i_x+self.dimxA: i_x+self.dimx-1] += (-nu) / (np.linalg.norm(xB_t-self.obs[j, 0:2]) - self.obs[j,-1] - eps) \
                       * (xB_t - self.obs[j, 0:2]) / np.linalg.norm( xB_t - self.obs[j, 0:2] )
            """
            return grad

        mu = self.mu
        # nu = self.nu
        X0 = np.concatenate((x_traj[1:, :].flatten(), a_traj.flatten()))            # get rid of x_traj[0, :] = x0
        for i in range(1):  # iterate for different mu
            result = minimize(J, X0, jac=dJ, args=(mu), constraints=constr, options={'maxiter': 100, 'disp': False})
            #result = minimize(J, X0, args=(mu), constraints=nonlincon)
            print('mu={}: success:{}({}) -> {}, fun={}'.format(mu, result.success, result.status, result.message, result.fun))
            X0 = result.x
            mu = 10*mu
        #result = minimize(J, X0, args=(mu), constraints=nonlincon)
        #print('Status: {}, {}, fun = {}'.format(result.success, result.message, result.fun))
        #print(result.jac)
        #print(dJ(result.x, mu))
        #print(result.jac - dJ(result.x))

        # convert results to trajectories
        idx = self.dimT*self.dimx
        x_opt, a_opt = result.x[: idx], result.x[idx: ]
        x_opt = np.concatenate((x0, x_opt))     # add x0 to trajectory
        return x_opt.reshape((self.dimT+1, self.dimx)), a_opt.reshape((self.dimT, self.dima))

    def pmp_solver(self, brnet, x_init, a_init):
        """
        This function use GD to solve PMP and find a local optimal trajectory.
        The follower's dynamic constraints are involved.
        x_traj = [x_0, ..., x_T], a_traj = [a_0, ..., a_{T-1}], lam_traj = [lam_1, ..., lam_T]
        """
        # related inner function definitions
        def backward_pass_terminal(x_T):
            """
            Only process terminal cost: 
            (x_T-xd) @ qf1 @ (x_T-xd) + x_T @ qf2 @ x_T + nu*log(|pA_T-pj|-dj) + nu*log(|pB_T-pj|-dj). 
            """
            xd = np.concatenate( (self.pf, self.pf, np.zeros(1)) )
            pA_T, pB_T = x_T[: self.dimxA], x_T[self.dimxA: -1]
            lam_T = 2*self.qf1 @ (x_T-xd) + 2*self.qf2 @ x_T    # for q_theta(x_T)

            # for soft safety constraints
            eps = 1e-1
            for j in range(self.obs_num):
                q_j, d_j = self.obs[j, :2], self.obs[j, -1]     # get obs position and safe distance
                lam_T[: self.dimxA] += (-self.nu) / (np.linalg.norm(pA_T-q_j) - d_j - eps) * (pA_T - q_j) / np.linalg.norm(pA_T - q_j)
                lam_T[self.dimxA: -1] += (-self.nu) / (np.linalg.norm(pB_T-q_j) - d_j - eps) * (pB_T - q_j) / np.linalg.norm(pB_T - q_j)
            return lam_T

        def backward_pass(x_t, a_t, lam_tp1, brnet):
            """
            Running cost: (x_t-xd) @ q1 @ (x_t-xd) + x_t @ q2 @ x_t + nu*log(|pA_t-pj|-dj) + nu*log(|pB_t-pj|-dj).
            Dynamics: f(x_t, a_t) = x_t + B1@a_t + B2@b_t + A(x_t)b_t = x_t + B1@a_t + ff(x_t,b_t)
            """
            xd = np.concatenate( (self.pf, self.pf, np.zeros(1)) )
            pA_t, pB_t, phi_t = x_t[: self.dimxA], x_t[self.dimxA: -1], x_t[-1]
            b_t = self.get_b(brnet, x_t, a_t)
            jac_x, _ = self.get_b_jac(brnet, x_t, a_t)

            # running cost
            lam_t = 2*self.q1 @ (x_t-xd) + 2*self.q2 @ x_t

            # for soft safety constraints
            eps = 1e-1
            for j in range(self.obs_num):
                q_j, d_j = self.obs[j, :2], self.obs[j, -1]     # get obs position and safe distance
                lam_t[: self.dimxA] += (-self.nu) / (np.linalg.norm(pA_t-q_j) - d_j - eps) * (pA_t - q_j) / np.linalg.norm(pA_t - q_j)
                lam_t[self.dimxA: -1] += (-self.nu) / (np.linalg.norm(pB_t-q_j) - d_j - eps) * (pB_t - q_j) / np.linalg.norm(pB_t - q_j)
            
            # dynamics related, ff = [uA*dt, fB(x_t,a_t) ]. first term [xA_t, xB_t] is not included.
            dff_dx = np.zeros((self.dimx, self.dimx))
            dff_dx[2,:] = np.cos(phi_t) * jac_x[0, :] + b_t[0] * np.array([0,0,0,0,-np.sin(phi_t)])
            dff_dx[3,:] = np.sin(phi_t) * jac_x[0, :] + b_t[0] * np.array([0,0,0,0, np.cos(phi_t)])
            dff_dx[4,:] = jac_x[1, :]
            lam_t += lam_tp1 @ (np.eye(self.dimx) + self.dt * dff_dx)
            
            return lam_t

        def H_t_grad(x_t, a_t, lam_tp1):
            # evaluate the gradient of hamiltonian H_t
            phi_t = x_t[-1]
            _, jac_a = self.get_b_jac(brnet, x_t, a_t)
            Ax = self.dt * np.array([[0,0], [0,0], [np.cos(phi_t),0], [np.sin(phi_t),0], [0,1]])
            grad = 2*self.r @ a_t + lam_tp1 @ self.B1 + lam_tp1 @ Ax @ jac_a
            return grad

        def H_grad(x_traj, a_traj, lam_traj):
            # aggregate the gradient of hamiltonian H_0, H_1, ..., H_{T-1}
            traj_len = a_traj.shape[0]
            dH = np.zeros((traj_len, self.dima))    # axis=0 is control horizon
            for t in range(traj_len):
                dH[t, :] = H_t_grad(x_traj[t, :], a_traj[t, :], lam_traj[t, :])
            return dH
        
        traj_len = a_init.shape[0]
        x_traj, lam_traj = np.zeros((traj_len+1,self.dimx)), np.zeros((traj_len,self.dimx))
        a_traj = np.zeros((traj_len, self.dima))
        iter, ITER_MAX = 0, 200
        x0 = x_init[0,:]
        tol = 1e-1              # gradient norm tolarance
        step = 1e-3             # GD step size
        a_pre = a_init.copy()   # create reference. a_init changes with a_pre

        while True:
            # forward pass to update x_traj
            x_traj[0, :] = x0
            for t in range(traj_len):
                x_traj[t+1, :] = self.dynamics(x_traj[t,:], a_pre[t,:], self.get_b(brnet, x_traj[t,:], a_pre[t,:]))
            
            # backward pass to update lam_traj
            for t in reversed(range(traj_len)):
                if t == traj_len-1:
                    lam_traj[t, :] = backward_pass_terminal(x_traj[t+1, :])   # use x_T
                else:
                    lam_traj[t, :] = backward_pass(x_traj[t+1, :], a_pre[t+1, :], lam_traj[t+1, :], brnet)  # use x_t, a_t, p_tp1

            # compute gradient of Hamultonian
            dH = H_grad(x_traj, a_pre, lam_traj)

            # update a_traj with GD
            for t in range(traj_len):
                a_traj[t, :] =  a_pre[t, :] - step * dH[t, :]
            
            # stopping condition
            if iter > ITER_MAX or np.linalg.norm(dH, axis=1).max() < tol:
                break
            
            a_pre = a_traj.copy()
            iter += 1
            print('  pmp iter {}, leader cost: {}, |dH|: {}'.format(iter, self.obj_oc(x_traj, a_traj), np.linalg.norm(dH, axis=1).max()))
        return x_traj, a_traj


    def obj_oc(self, x_traj, a_traj):
        """
        This function computes the obj for leader's optimal control cost. 
        J_oc = (X-Xf) @ Q1 @ (X-Xf) + x @ Q2 @ x + a @ R @ a
        """
        x0 = x_traj[0, :]
        traj_len = a_traj.shape[0]
        idx = self.dimx * traj_len
        X = np.concatenate( (x_traj[1:, :].flatten(), a_traj.flatten()) )
        cost = (X[: idx] - self.Xf) @ self.Q1 @ (X[: idx] - self.Xf) + X[: idx] @ self.Q2 @ X[: idx] + X[idx: ] @ self.R @ X[idx: ] + self.c
        return cost

    def obj_br(self, brnet, task_data):
        """
        This function computes the obj for leader's BR data fitting cost.
        task_data is a 2d numpy array. task_data[i, :] = [x_t,a_t,b_t]
        """
        N = task_data.shape[0]
        x_traj, a_traj, b_traj = task_data[:, :self.dimx], task_data[:, self.dimx: self.dimx+self.dima], task_data[:, self.dimx+self.dima: ] 
        x, a, b = self.to_torch(x_traj), self.to_torch(a_traj), self.to_torch(b_traj)
        br_cost_fn = torch.nn.MSELoss(reduction='sum')        
        cost = br_cost_fn(brnet(x, a), b).item() / N
        return cost

    def obj_L(self, brnet, task_data, a_traj):
        """
        This function computes the meta function L.
        """
        x_traj, _ = self.get_conjectured_traj(brnet, a_traj)
        return self.obj_oc(x_traj, a_traj) + self.gam * self.obj_br(brnet, task_data)

    def grad_obj_oc(self, brnet, a_traj):
        """
        This function computes gradient of obj for leader's optimal control w.r.t. brnet parameter w.
        The gradient stores in brnet and can be retrived by the method get_grad().
        """
        traj_len = a_traj.shape[0]
        # convert parameters to torch format
        x0, xf, a_traj = self.to_torch(self.x0), self.to_torch(self.xf), self.to_torch(a_traj)
        B1 = self.to_torch(self.B1)
        q1, qf1 = self.to_torch(self.q1), self.to_torch(self.qf1)
        q2, qf2 = self.to_torch(self.q2), self.to_torch(self.qf2)
        r = self.to_torch(self.r)
        cost = 0

        for t in range(traj_len):
            a_t = a_traj[t, :]
            if t == 0:
                x_t = x0
            else:
                # update dynamics: x_tp1 = x_t + B1 a_t + fB(x) b(x_t,a_t)
                x_t = x_t + B1 @ a_t \
                    + self.dt * torch.tensor([[0.,0.], [0.,0.], [torch.cos(x_t[-1]),0], [torch.sin(x_t[-1]),0], [0,1]]) @ brnet(x_t, a_t)
            cost += (x_t-xf) @ q1 @ (x_t-xf) + x_t @ q2 @ x_t + a_t @ r @ a_t    # stage cost
        cost += (x_t-xf) @ qf1 @ (x_t-xf) + x_t @ qf2 @ x_t     # terminal cost
        
        brnet.zero_grad()
        cost.backward()
        return brnet
    
    def grad_obj_br(self, brnet, task_data):
        """
        This function computes gradient of obj for leader's BR data fitting cost w.r.t. brnet parameter w.
        task_data is a 2d numpy array. task_data[i, :] = [x_t,a_t,b_t]
        The gradient stores in brnet and can be retrived by the method get_grad().
        """
        N = task_data.shape[0]
        x_traj, a_traj, b_traj = task_data[:, :self.dimx], task_data[:, self.dimx: self.dimx+self.dima], task_data[:, self.dimx+self.dima: ] 
        x, a, b = self.to_torch(x_traj), self.to_torch(a_traj), self.to_torch(b_traj)
        br_cost_fn = torch.nn.MSELoss(reduction='sum')        
        cost = br_cost_fn(brnet(x, a), b) / N

        brnet.zero_grad()   # clear gradient before computing the gradient
        cost.backward()
        return brnet

    def grad_obj_L(self, brnet, task_data, a_traj):
        """
        This function computes gradient of meta function L.
        Returns a brnet with same parameter (W,b) but updated gradient (dW, db).
        The updated gradient in brnet can be retrived by the method get_grad() or get_grad_dict().
        """
        cost = 0
        brnet_new = BRNet(); 
        brnet_new.load_state_dict(brnet.state_dict())  # need deep copy or create a new brnet

        # step 1: formulate JA
        traj_len = a_traj.shape[0]
        x0, xf, a_traj = self.to_torch(self.x0), self.to_torch(self.xf), self.to_torch(a_traj)
        B1 = self.to_torch(self.B1)
        q1, qf1 = self.to_torch(self.q1), self.to_torch(self.qf1)
        q2, qf2 = self.to_torch(self.q2), self.to_torch(self.qf2)
        r = self.to_torch(self.r)

        for t in range(traj_len):
            a_t = a_traj[t, :]
            if t == 0:
                x_t = x0
            else:
                # update dynamics: x_tp1 = x_t + B1 a_t + A(x) b(x_t,a_t)
                x_t = x_t + B1 @ a_t \
                    + torch.tensor([[0.,0.], [0.,0.], [torch.cos(x_t[-1])*self.dt,0], [torch.sin(x_t[-1])*self.dt,0], [0,self.dt]]) @ brnet(x_t, a_t)
            cost += (x_t-xf) @ q1 @ (x_t-xf) + x_t @ q2 @ x_t + a_t @ r @ a_t    # stage cost
        cost += (x_t-xf) @ qf1 @ (x_t-xf) + x_t @ qf2 @ x_t     # terminal cost

        # step 2: formulate QA
        N = len(task_data)
        x_traj, a_traj, b_traj = task_data[:, :self.dimx], task_data[:, self.dimx: self.dimx+self.dima], task_data[:, self.dimx+self.dima: ] 
        x, a, b = self.to_torch(x_traj), self.to_torch(a_traj), self.to_torch(b_traj)
        br_cost_fn = torch.nn.MSELoss(reduction='sum')        
        
        # step 3: let L = JA + gam * QA and find gradient
        cost += self.gam * br_cost_fn(brnet_new(x, a), b) / N
        brnet_new.zero_grad()   # clear gradient before computing the gradient
        cost.backward()
        return brnet_new

    def get_b(self, brnet, x, a):
        """
        This function converts (x,a) to torch tensor and returns br(x,a) to numpy array. Make codes more concise.
        """
        return brnet(torch.from_numpy(x).float(), torch.from_numpy(a).float()).detach().numpy().astype(float)
    
    def get_b_jac(self, brnet, x, a):
        """
        This function converts (x,a) to torch tensor and returns the jacobian of br(x,a) w.r.t. (x,a) to numpy array. 
        Make codes more concise.
        """
        jac_x, jac_a = brnet.compute_input_jac(torch.from_numpy(x).float(), torch.from_numpy(a).float())
        return jac_x.numpy().astype(float), jac_a.numpy().astype(float)
    
    def to_torch(self, x):
        """
        This function converts numpy array to the torch version.
        """
        return torch.from_numpy(x).float()

    def get_conjectured_traj(self, brnet, a_traj):
        """
        This function generates the conjectured x_traj and b_traj based on a_traj and brnet.
        The trajectory is based on the leader's conjecture brnet, not the real interactive trajectory.
        """
        traj_len = a_traj.shape[0]
        x_traj, b_traj = np.zeros((traj_len+1, self.dimx)), np.zeros_like(a_traj)
        x_traj[0, :] = self.x0
        for i in range(traj_len):
            b_traj[i, :] = self.get_b(brnet, x_traj[i, :], a_traj[i, :])
            x_traj[i+1, :] = self.dynamics(x_traj[i, :], a_traj[i, :], b_traj[i, :])
        return x_traj, b_traj


class Follower:
    """
    Follower class deals with follower's utils
    """
    def __init__(self, scn, theta) -> None:
        self.theta = theta
        self.scn = scn
        self.dt = param.dt
        self.dimx, self.dimxA, self.dimxB = param.dimx, param.dimxA, param.dimxB
        self.dima, self.dimb = param.dima, param.dimb

        obs, _, x0B, pf, _, _, w = param.get_param(scn, theta)     # scenario parameter
        self.obs_num = len(obs)
        self.obs = np.array(obs)
        self.x0B = np.array(x0B)
        self.pf = np.array(pf)
        self.compute_cost_parameters(w)
        
    def compute_cost_parameters(self, w):
        """
        This function precomputes the follower's parameter to facilitate computation.
        J^B = (x_tp1-xd) @ Q1 @ (x_tp1-xd) + x_tp1 @ Q2 @ x_tp1 + Q3*angle_cost(x_tp1) + b_t @ Q4 b_t + h(|pB-obs_j|)
        """
        self.Q1 = np.diag(np.array(w[0]))
        d = np.hstack( (np.eye(2), -np.eye(2), np.zeros((2,1))) )   # extract pA - pB
        self.Q2 = d.T @ np.diag(np.array(w[1])) @ d
        self.Q3 = w[2]
        self.Q4 = np.diag(np.array(w[3]))

    def dynamics(self, x_t, a_t, b_t):
        """
        This function computes x_{t+1} = f(x_t, a_t, b_t).
        """
        xB_t = x_t[self.dimxA: ]
        dot_xB_t = np.array( [np.cos(xB_t[-1])*b_t[0], np.sin(xB_t[-1])*b_t[0], b_t[1]] ) 
        x_tp1 = x_t + np.concatenate((a_t, dot_xB_t)) * self.dt
        return x_tp1
    
    def is_safe(self, p):
        """
        This function check whether the current pB is safe or not (within the safety region).
        """
        flag = True
        for j in range(self.obs_num):
            q_j, d_j = self.obs[j, 0: 2], self.obs[j, -1]   # get obs position and safe distance
            dist = np.linalg.norm(p - q_j)
            if dist <= d_j:
                flag = False
                break
        return flag

    def obs_cost(self, x):
        """
        This function computes obstacle cost: \sum_j h(p, obs_j). h is a piece-wise function.
        Safety sensitivity is characterized by safety distance obs_i[-1].
        """
        p = x[self.dimxA: -1]   # equivalent to write Ax with A = [0_{2x2}, I_{2x2}, 0_{2x1}]
        cost = 0
        for j in range(self.obs_num):
            q_j, d_j = self.obs[j, 0: 2], self.obs[j, -1]   # get obs position and safe distance
            dist = np.linalg.norm(p - q_j)
            if dist <= d_j:
                cost -= 1e8     # if inside obstacle, large penalty
            elif dist < 1.5*d_j:
                cost -= np.log( (dist-d_j) / (0.5*d_j) )
            else:
                pass
        return cost
    
    def grad_b_obs_cost(self, x):
        """
        This function computes gradient of obs cost w.r.t to b: grad_b \sum_j h(p, obs_j). 
        h is a piece-wise function.
        """
        p = x[self.dimxA: -1]   # equivalent to write Ax with A = [0_{2x2}, I_{2x2}, 0_{2x1}]
        dcost = np.zeros(self.dimb)
        for j in range(self.obs_num):
            q_j, d_j = self.obs[j, 0: 2], self.obs[j, -1]   # get obs position and safe distance
            dist = np.linalg.norm(p - q_j)
            #if dist < d_j:
            #    jac_p_b = np.array([[np.cos(x[-1]), 0], [np.sin(x[-1]), 0]])    # derivative related to pB_tp1
            #    dcost -= (d_i/dist) * ((p-q_i)/dist) @ (self.dt * jac_p_b)      # derivative of -log(|pB-obj_j|/d_j) to uB
            if dist <= d_j:
                jac_p_b = np.array([[np.cos(x[-1]), 0], [np.sin(x[-1]), 0]])
                dcost -= 1e8 * ((p-q_j)/dist) @ (self.dt * jac_p_b)     # if inside obstacle, use very large gradient
            elif dist < 1.5*d_j:
                jac_p_b = np.array([[np.cos(x[-1]), 0], [np.sin(x[-1]), 0]])
                dcost -= (1/(dist-d_j)) * ((p-q_j)/dist) @ (self.dt * jac_p_b)
            else:
                pass
        return dcost
    
    def angle_cost(self, x):
        """
        This function penalizes the angle deviation of the robot, which makes robot take actions in any situation.
        cost = (pB - pA) @ [cos(phi), sin(phi)] / |pB-pA|
        """
        pA, pB, phi = x[: self.dimxA], x[self.dimxA: -1], x[-1]
        #cost = np.arccos( (pA-pB) @ np.array([np.cos(phi), np.sin(phi)]) / np.linalg.norm(pA-pB) ) ** 2
        cost = (pB-pA) @ np.array([np.cos(phi), np.sin(phi)]) / np.linalg.norm(pB-pA)
        return cost
    
    def grad_b_angle_cost(self, x, x_new):
        """
        This function computes the angle cost gradient w.r.t. uB = (vB,wB).
        """
        p_diff = x_new[self.dimxA: -1] - x_new[: self.dimxA]            # pB_tp1 - pA_tp1
        g = p_diff @ np.array([np.cos(x_new[-1]), np.sin(x_new[-1])])   # g = (pB_tp1 - pA_tp1) @ [cos(phi_tp1), sin(phi_tp1)]
        jac1 = np.zeros(self.dimb)  # dg/duB
        jac1[0] = self.dt*( np.cos(x[-1])*np.cos(x_new[-1]) + np.sin(x[-1])*np.sin(x_new[-1]) ) # dg/dv
        jac1[1] = p_diff @ ( self.dt*np.array([-np.sin(x_new[-1]), np.cos(x_new[-1])]) )        # dg/dw
        jac2 = g * p_diff @ ( self.dt*np.array([[np.cos(x[-1]), 0], [np.sin(x[-1]),0]]) )   
        jac = jac1 / np.linalg.norm(p_diff) - jac2 / np.linalg.norm(p_diff)**3
        return jac

    def compute_obj(self, x, a, b):
        """
        This function comptues the follower's obj. 
        J^B = (x_tp1-xd) @ Q1 @ (x_tp1-xd) + x_tp1 @ Q2 @ x_tp1 + Q3*angle(x_tp1) + b_t @ Q4 b_t + h(|pB-obs_j|).
        """
        x_new = self.dynamics(x, a, b)
        xd = np.concatenate( (self.pf, self.pf, np.zeros(1)) )
        J = (x_new-xd) @ self.Q1 @ (x_new-xd) + x_new @ self.Q2 @ x_new + self.Q3 * self.angle_cost(x_new) \
            + b @ self.Q4 @ b + self.obs_cost(x_new)
        return J

    def get_br(self, x, a):
        """
        This function computes the follower's best response given x,a. Note x_new is used in J^B not x.
        """
        xd = np.concatenate( (self.pf, self.pf, np.zeros(1)) )
        def J(b, x,a):  # compute objective function, b is decision variable
            x_new = self.dynamics(x, a, b)
            J = (x_new-xd) @ self.Q1 @ (x_new-xd) + x_new @ self.Q2 @ x_new + self.Q3 * self.angle_cost(x_new) \
                + b @ self.Q4 @ b + self.obs_cost(x_new)
            return J
        
        def dJ(b, x,a): # compute grad of objective function, b is decision variable
            x_new = self.dynamics(x, a, b)
            jac_x_b = np.zeros((self.dimx, self.dimb))
            jac_x_b[self.dimxA, 0] = np.cos(x[-1]) * self.dt     # dx/dvB at current time
            jac_x_b[self.dimxA+1, 0] = np.sin(x[-1]) * self.dt   # dx/dvB
            jac_x_b[self.dimxA+2, 1] = self.dt                   # dx/dwB
            dJ = 2*(self.Q1 @ (x_new-xd)) @ jac_x_b + 2*(self.Q2 @ x_new) @ jac_x_b + self.Q3 * self.grad_b_angle_cost(x, x_new) \
                + 2*self.Q4 @ b + self.grad_b_obs_cost(x_new)
            return dJ

        # constraints for vB and wB are separate, use linear constraints. 
        lc1 = LinearConstraint(np.eye(self.dimb), np.array([0., -1.]), np.array([1., 1.]))    # [0,-1] <= (vB,wB) <= [1,1]
        b0 = np.ones(self.dimb) * 0.1

        #result = minimize(J, b0, args=(x,a), constraints=[lc1], tol=1e-6, options={'disp':False})
        result = minimize(J, b0, args=(x,a), jac=dJ, constraints=[lc1], tol=1e-7, options={'disp':False})
        #print(result.jac)
        return result.x
        
    def get_interactive_traj(self, x0, a_traj):
        """
        This function generates the real interactive trajectory given x0 and leader's action trajection a_traj.
        The follower responses according to his real objective instead of brnet.
        """
        traj_len = a_traj.shape[0]
        x_traj, b_traj = np.zeros((traj_len+1, self.dimx)), np.zeros((traj_len, self.dimb))
        x_traj[0, :] = x0
        B1 = self.dt*np.vstack( (np.eye(self.dima), np.zeros((self.dimxB, self.dima))) )
        Ax = lambda x: self.dt*np.array([[0,0], [0,0], [np.cos(x[-1]),0], [np.sin(x[-1]),0], [0,1]])
        for i in range(traj_len):
            x_t, a_t = x_traj[i, :], a_traj[i, :]
            b_t = self.get_br(x_t, a_t)
            x_traj[i+1, :] = x_t + B1 @ a_t + Ax(x_t) @ b_t
            b_traj[i, :] = b_t
        return x_traj, b_traj

    def no_guidance(self, T_lim=60):
        """
        This function implements no guidance scenario for the follower. Jb = |xB-pd|_w1 + |uB|_W4 + h_theta(xB)
        """
        N_lim = int(T_lim / self.dt)
        x_traj, b_traj = np.zeros((N_lim+1, self.dimx)), np.zeros((N_lim, self.dimb))
        x_traj[0, :] = np.concatenate( (np.zeros(2), self.x0B) )
        xd = np.concatenate( (self.pf, self.pf, np.zeros(1)) )
        
        def J(b, x,a):  # compute objective function, b is decision variable
            x_new = self.dynamics(x, a, b)   # do not care about xA and uA
            J = (x_new-xd) @ self.Q1 @ (x_new-xd) + b @ self.Q4 @ b + self.obs_cost(x_new)
            return J
        def dJ(b, x,a): 
            x_new = self.dynamics(x, a, b)
            jac_x_b = np.zeros((self.dimx, self.dimb))
            jac_x_b[self.dimxA, 0] = np.cos(x[-1]) * self.dt     # dx/dvB at current time
            jac_x_b[self.dimxA+1, 0] = np.sin(x[-1]) * self.dt   # dx/dvB
            jac_x_b[self.dimxA+2, 1] = self.dt                   # dx/dwB
            dJ = 2*(self.Q1 @ (x_new-xd)) @ jac_x_b + 2*self.Q4 @ b + self.grad_b_obs_cost(x_new)
            return dJ
        
        # constraints for vB and wB are separate, use linear constraints. 
        lc1 = LinearConstraint(np.eye(self.dimb), np.array([0., -1.]), np.array([1., 1.]))    # [0,-1] <= (vB,wB) <= [1,1]
        b0 = np.ones(self.dimb) * 0.1

        for i in range(N_lim):
            x, a = x_traj[i, :], np.zeros(self.dima)
            result = minimize(J, b0, args=(x,a), jac=dJ, constraints=[lc1], tol=1e-7, options={'disp':False})
            #result = minimize(J, b0, args=(x,a), constraints=[lc1], tol=1e-7, options={'disp':False})
            #print(result.jac - dJ(result.x, x,a))
            b_traj[i, :] = result.x
            x_traj[i+1, :] = self.dynamics(x, a, result.x)
        return x_traj, b_traj




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
        self.W = torch.tensor([[1., 0], [0., 2]]).to(self.device)
        self.b = torch.tensor([0., -1.]).to(self.device)

        self.linear1 = torch.nn.Linear(self.dimx+self.dima, 20)
        self.linear2 = torch.nn.Linear(20, 20)
        self.linear3 = torch.nn.Linear(20, self.dimb)
        self.activation = torch.nn.ReLU()   # or tanh or sigmoid
        # for normalization
        #self.linear4 = torch.nn.Linear(self.dimb, self.dimb)    
        #self.normalize = torch.nn.Sigmoid()
        #self.linear4.requires_grad_(False)      # only use the last layer as a constant

        # random initialization 
        self.linear1.weight.data.normal_(mean=0., std=.1)
        self.linear1.bias.data.normal_(0., .1)
        self.linear2.weight.data.normal_(0, .1)
        self.linear2.bias.data.normal_(0, .1)
        self.linear3.weight.data.normal_(0, .1)
        self.linear3.bias.data.normal_(0, .1)
        
        # constant initialization for testing
        #self.linear1.weight.data.fill_(.01)     
        #self.linear1.bias.data.fill_(.01)
        #self.linear2.weight.data.fill_(.01)
        #self.linear2.bias.data.fill_(.01)
        #self.linear3.weight.data.fill_(.01)
        #self.linear3.bias.data.fill_(.01)
        
        #with torch.no_grad():
        #    self.linear4.weight.copy_(self.W)
        #    self.linear4.bias.copy_(self.b)
    
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
        #y = self.normalize(y); y = self.linear4(y)
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
        #h4 = self.linear4.register_forward_hook(forward_hook)
        _ = self.forward(x, a)
        h1.remove()
        h2.remove()
        h3.remove()
        #h4.remove()

        def d_activation(y):
            """
            This function computes derivative of activation functions. can be relu, tanh, sigmoid.
            Input is a 1d array, output is nxn matrix.
            """
            #df = torch.diag(1 - torch.tanh(y**2))  # for tanh(x)
            y[y < 0] = 0; df = torch.diag(y)        # for relu(x)
            return df
        def d_normalize(y):
            """
            This function computes the derivative of normalization functions. can be sigmoid, tanh.
            """
            df = torch.diag(y*(1-y))    # for sigmoid, need dot product
            #df = torch.diag(1-y**2)     # for tanh, need to change W4 and b4
            return df
        p = self.get_data_dict()
        #jac_x = p['linear4.weight'] @ d_normalize(y[2]) @ \
        jac_x=    p['linear3.weight'] @ d_activation(y[1]) @ p['linear2.weight'] @ d_activation(y[0]) @ p['linear1.weight'][:, : self.dimx]
        #jac_a = p['linear4.weight'] @ d_normalize(y[2]) @ \
        jac_a=    p['linear3.weight'] @ d_activation(y[1]) @ p['linear2.weight'] @ d_activation(y[0]) @ p['linear1.weight'][:, self.dimx: ]
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
        #dp_dict["linear4.weight"] = None #self.linear4.weight.grad.detach().cpu()
        dp_dict["linear1.bias"] = self.linear1.bias.grad.detach().cpu()
        dp_dict["linear2.bias"] = self.linear2.bias.grad.detach().cpu()
        dp_dict["linear3.bias"] = self.linear3.bias.grad.detach().cpu()
        #dp_dict["linear4.bias"] = None #self.linear4.bias.grad.detach().cpu()
        return dp_dict

    def get_zero_grad_dict(self):
        dp_dict = {} 
        for n, p in self.named_parameters():
            dp_dict[n] = torch.zeros_like(p)
        return dp_dict

    def get_intermediate_output(self, x, a):
        """
        This function gets the output of every Linear layer.
        """
        y = []
        def forward_hook(model, input, output):
            y.append( output.detach() )
        h1 = self.linear1.register_forward_hook(forward_hook)
        h2 = self.linear2.register_forward_hook(forward_hook)
        h3 = self.linear3.register_forward_hook(forward_hook)
        #h4 = self.linear4.register_forward_hook(forward_hook)
        _ = self.forward(x, a)
        h1.remove()
        h2.remove()
        h3.remove()
        #h4.remove()
        return y 
        

class Meta:
    """
    Meta class implements meta learning algorithms.
    """
    def __init__(self):
        self.task_num = param.total_type
        self.task_pdf = param.type_pdf
        self.ws_len = param.ws_len
        self.kappa = param.kappa
        self.lr = param.lr
        self.lr_meta = param.lr_meta
        self.momentum = param.momentum
    
    def sample_tasks(self, K):
        """
        This function samples K tasks according to given task distribution
        """
        task = np.arange(self.task_num)
        task_sample = np.random.choice(task, K, p=self.task_pdf)
        return task_sample
    
    def sample_task_theta(self, follower, x_traj, a_traj, N=100):
        """
        This function samples N points for individual tasks theta based on ration kappa.
        D = [D_traj, D_uniform], D_traj / D_uniform = kappa.
        """
        D_traj = self.sample_task_theta_traj(follower, x_traj, a_traj, N)
        fname = 'data/br_data/scenario' + str(follower.scn) + '/task'+str(follower.theta)+'_uniform.npy'
        if os.path.exists(fname):
            D_uniform = np.load(fname)          # store random data offline and sample from them
        else:
            D_uniform = self.sample_task_theta_uniform(follower, N)

        # use the ratio and shuffle the data
        n_uniform = N // (self.kappa+1)
        n_traj = N - n_uniform
        D_t = D_traj[np.random.choice(D_traj.shape[0], n_traj), :]
        D_u = D_uniform[np.random.choice(D_uniform.shape[0], n_uniform), :]
        task_data = np.vstack( (D_t, D_u) )
        np.random.shuffle(task_data)
        return task_data

    def sample_task_theta_uniform(self, follower, N=100):
        """
        This function uniformly samples N points for the individual task theta.
        BR data points [x,a,br(x,a)] 
        """
        task_data = []
        for i in range(N):
            # sample pA, pB in the square working space and wB in [-1,1]
            pA = np.random.rand(2) * self.ws_len
            while True:
                pB = np.random.rand(2) * self.ws_len
                wB = np.random.rand(1) * 2 - 1
                if follower.is_safe(pB):
                    break
            x = np.concatenate((pA,pB,wB))
            # sample a, |a| <= 1
            a = np.random.rand(2) * 2 - 1
            if np.linalg.norm(a) > 1:
                a = a / np.linalg.norm(a)

            br = follower.get_br(x, a)
            task_data.append(np.concatenate((x, a, br)))
        task_data = np.array(task_data)     # task_data[i, :] = [x, a, br]
        return task_data
    
    def sample_task_theta_traj(self, follower, x_traj, a_traj, N=None):
        """
        This function samples br points for the individual task theta given x_traj and a_traj.
        BR data points is sampled at (x_t+noise, a+noise).
        parameter N determines the number of br samples.
        """
        mu, sigma = 0, 0.5
        traj_len = a_traj.shape[0]
        dimx, dima = param.dimx, param.dima
        task_data = []

        if N is None:
            task_data_len = traj_len
            for i in range(task_data_len):
                x = x_traj[i, :] + 0.5 * np.random.normal(mu, sigma, dimx)
                a = a_traj[i, :] + 0.1 * np.random.normal(mu, sigma, dima)
                br = follower.get_br(x, a)
                task_data.append(np.concatenate((x, a, br)))
        else:
            task_data_len = N
            for i in range(task_data_len):
                # randomly select one (x_t, a_t) from trajectory
                j = np.random.choice(traj_len)
                x = x_traj[j, :] + 0.5 * np.random.normal(mu, sigma, dimx)
                a = a_traj[j, :] + 0.1 * np.random.normal(mu, sigma, dima)
                br = follower.get_br(x, a)
                task_data.append(np.concatenate((x, a, br)))
        task_data = np.array(task_data)
        return task_data

    def update_model(self, leader, brnet, task_sample, br_list, D2_list, a_traj_list):
        """
        This function updates the entire brnet based on GD (with momentum).
        """
        brnet.zero_grad()   # clear gradient to compute accumulated gradient
        dp = brnet.get_zero_grad_dict()

        for i in range(len(task_sample)):
            theta = task_sample[i]
            leader.read_task_info(leader.scn, theta)
            brnet_i = leader.grad_obj_L(br_list[i], D2_list[i], a_traj_list[i])
            
            # compute accumulated gradient
            dp_i = brnet_i.get_grad_dict()
            for n, p in brnet_i.named_parameters():
                if n == 'linear4.weight' or n == 'linear4.bias':    # linear4 is constant and has no grad
                    continue
                dp[n] += dp_i[n]
        # use accumulated gradient (DW,db) to update (W,b) in brnet
        ITER_MAX = 1
        with torch.no_grad():       # use no_grad to mannually update NN parameter
            for iter in range(ITER_MAX):
                for n, p in brnet.named_parameters():
                    if n == 'linear4.weight' or n == 'linear4.bias':    # linear4 is constant and has no grad
                        continue
                    p -= (self.lr_meta / len(task_sample)) * (self.momentum * p + dp[n])
        return brnet

    def update_model_theta(self, leader, brnet, task_data, a_traj):
        """
        This function updates the brnet for task theta based on GD (with momentum).
        Return a brnet1 with updated parameter.
        """
        brnet_mid = leader.grad_obj_L(brnet, task_data, a_traj)     # brnet_mid has same (W,b) as brnet but updated (DW,db) 
     
        # use updated (DW,db) to update (W,b) in brnet_mid
        dp = brnet_mid.get_grad_dict()
        ITER_MAX = 2
        with torch.no_grad():       # use no_grad to mannually update NN parameter
            for iter in range(ITER_MAX):
                for n, p in brnet_mid.named_parameters():
                    if n == 'linear4.weight' or n == 'linear4.bias':    # linear4 is constant and has no grad
                        continue
                    p -= self.lr * (self.momentum * p + dp[n])                # one step sgd with momentum
        return brnet_mid

    def train_brnet(self, brnet, task_data, N=100):
        """
        This function trains an initial brnet using N randomly sampled task data.
        task_data is a 2d numpy array. task_data[i, :] = [x_traj, a_traj, b_traj].
        """
        if N > len(task_data):
            raise Exception("Training data is less than the requested amount.")
        
        dimx, dima = param.dimx, param.dima  
        D = task_data[np.random.choice(task_data.shape[0], N), :]   # randomly pick N data points.
        x_traj, a_traj, b_traj = D[:, 0: dimx], D[:, dimx: dimx+dima], D[:, dimx+dima: ]
        x, a, b = torch.from_numpy(x_traj).float(), torch.from_numpy(a_traj).float(), torch.from_numpy(b_traj).float()
        
        loss_fn = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.SGD(brnet.parameters(), lr=0.001, momentum=self.momentum)
        epoch = 2
        batch_size = 10
        ITER_MAX = 10   # control iteration number. Can be overfitting
        for ep in range(epoch):
            idx = np.random.choice(np.arange(N), batch_size).tolist()
            x_train, a_train, b_train = x[idx, :], a[idx, :], b[idx, :]
            for t in range(ITER_MAX):
                y = brnet(x_train, a_train)
                cost = loss_fn(y, b_train)
                optimizer.zero_grad()
                cost.backward()
                optimizer.step()
                if t == ITER_MAX-1:
                    print('epoch {}, cost {}'.format(ep, cost.item()))
        return brnet



class Auxiliary():
    """
    This class defines auxiliary functions such as constraint checks and trajectory plotting.
    """
    def __init__(self) -> None:
        self.dimx, self.dimxA, self.dimxB = param.dimx, param.dimxA, param.dimxB
        self.dima, self.dimb = param.dima, param.dimb
        self.Tf, self.dt, self.dimT = param.Tf, param.dt, int(param.Tf / param.dt)
        self.B1 = self.dt*np.vstack( (np.eye(self.dima), np.zeros((self.dimxB, self.dima))) )
        
    def check_dynamics_diff(self, brnet, x_traj, a_traj):
        """
        This function checks if the given trajectory satisfy the dynamics constraint. 
        """
        traj_len = a_traj.shape[0]
        Ax = lambda x: self.dt * np.array([[0,0], [0,0], [np.cos(x[-1]),0], [np.sin(x[-1]),0], [0,1]])
        dif = np.zeros((traj_len, self.dimx))   # diff_t = f(x_t, a_t) - x_tp1
        for i in range(traj_len):
            dyn = x_traj[i, :] + self.B1 @ a_traj[i, :] + Ax(x_traj[i,:]) @ self.get_b(brnet, x_traj[i,:], a_traj[i,:])
            dif[i, :] = x_traj[i+1, :] - dyn
        return dif 

    def check_input_constraint(self, brnet, x_traj, a_traj):
        """
        This function checks if the optimized trajectory violates the control input constraints.
        """
        eps = 1e-4
        # for control input constraints
        tmp = a_traj[:, 0]**2 + a_traj[:, 1]**2
        idx = np.argwhere(tmp > 1+eps); idx = idx.reshape(idx.size)
        if idx.size == 0: 
            print('No input constraint violation.')
        else:
            print('Input constraint violation. idx: {}'.format(idx))
        
    def check_dynamics_constraint(self, brnet, x_traj, a_traj):
        """
        This function checks if the optimized trajectory violates the dynamics constraints.
        """
        eps = 1e-2
        # for dynamics constraints
        f = np.zeros(self.dimT*self.dimx)
        fA, fB = np.zeros(self.dimT), np.zeros(self.dimT)
        Ax = lambda x: self.dt * np.array([[0,0], [0,0], [np.cos(x[-1]),0], [np.sin(x[-1]),0], [0,1]])
        for i in range(self.dimT):
            i_x = i * self.dimx
            f[i_x: i_x+self.dimx] = -x_traj[i+1,:] + x_traj[i,:] + self.B1 @ a_traj[i, :] + Ax(x_traj[i,:]) @ self.get_b(brnet, x_traj[i,:], a_traj[i,:])
            fA[i] = np.linalg.norm(f[i_x: i_x+self.dimxA])
            fB[i] = np.linalg.norm(f[i_x+self.dimxA: i_x+self.dimx])
        idx = np.argwhere(np.abs(fA) > eps); idx = idx.reshape(idx.size)
        if idx.size == 0:
            print('No dynamic constraint violation for leader.')
        else:
            print('Dynamic constraint violation for leader. idx: {}'.format(idx))
        idx = np.argwhere(np.abs(fB) > eps); idx = idx.reshape(idx.size)
        if idx.size == 0:
            print('No dynamic constraint violation for follower.')
        else:
            print('Dynamic constraint violation for follower. idx: {}'.format(idx))
        
    def check_safety_constraint(self, leader, x_traj):
        f, g = [], []
        for t in range(self.dimT+1):
            pA, pB = x_traj[t, :self.dimxA], x_traj[t, self.dimxA: -1]
            for j in range(leader.obs_num):
                p_j, d_j = leader.obs[j, :2], leader.obs[j, -1]
                if np.linalg.norm(pA-p_j) < d_j:
                    f.append(t)
                if np.linalg.norm(pB-p_j) < d_j:
                    g.append(t)
        if len(f) == 0:
            print('No safety constraint violation for the leader.')
        else:
            print('Safety constraint violation for the leader:', f)
        if len(g) == 0:
            print('No safety constraint violation for the follower.')
        else:
            print('Safety constraint violation for the follower:', g)
    
    def get_b(self, brnet, x, a):
        return brnet(torch.from_numpy(x).float(), torch.from_numpy(a).float()).detach().numpy().astype(float)
    
    def plot_trajectory(self, x_traj, real_pB=None, sim_pB=None):
        """
        This function plots the leader and follower's path trajectory given x_traj. x_traj is numpy array.
        real_pB: interactive follower's trajectory
        sim_pB: leader's conjectured follower's trajectory 
        """
        fig = plt.figure()
        pA, pB = x_traj[:, 0: self.dimxA], x_traj[:, self.dimxA: -1]
        #traj_len = x_traj.shape[0]
        plt.plot(pA[:,0], pA[:,1], 'b^')
        plt.plot(pB[:,0], pB[:,1], 'gs')
        if real_pB is not None:
            plt.plot(real_pB[:,0], real_pB[:,1], 'ro')
        if sim_pB is not None:
            plt.plot(sim_pB[:,0], sim_pB[:,1], 'mx')
        plt.savefig('tmp_fig_test.png')
        plt.close(fig)
