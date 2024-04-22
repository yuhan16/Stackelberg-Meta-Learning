import os
import json
import numpy as np
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
import torch
from .model import BRNet
from . import config_data_dir


class Leader:
    """
    This class defines the leader's utility functions.
    """
    def __init__(self) -> None:
        fname = os.path.join(config_data_dir, 'parameters.json')
        param = json.load(open(fname))

        self.seed = param['seed']
        self.rng = np.random.default_rng(self.seed)

        self.ws_len, self.obs, self.pd = param['ws_length'], param['obstacle_settings'], param['destination']
        
        self.dimx, self.dimxa, self.dimxb = param['dimx'], param['dimxa'], param['dimxb']
        self.dimua, self.dimub = param['dimua'], param['dimub']
        self.dimT, self.dt, self.T = param['dimT'], param['dt'], param['T']     # dimT = T // dt

        self.q1 = np.array(param['leader_cost_matrix']['q1'], dtype=float)
        self.q2 = np.array(param['leader_cost_matrix']['q2'], dtype=float)
        self.q3 = np.array(param['leader_cost_matrix']['q3'], dtype=float)
        self.qf1 = np.array(param['leader_cost_matrix']['qf1'], dtype=float)
        self.qf2 = np.array(param['leader_cost_matrix']['qf2'], dtype=float)
        
        self.meta = None

    
    def is_safe(self, p):
        """
        This function checks the safety constraints. obstacle collision, outside working space.
        """
        for i in range(len(self.obs)):
            obs_i = self.obs[i]
            xc, yc, rc = obs_i[0], obs_i[1], obs_i[2]
            x_scale, y_scale = obs_i[4], obs_i[5]
            f1 = np.diag([1/x_scale, 1/y_scale]) @ (p-np.array([xc,yc])) 
            if obs_i[3] == 1:
                if np.linalg.norm(f1, ord=1) <= rc:
                    return False
            elif obs_i[3] == 2:
                if np.linalg.norm(f1, ord=2) <= rc:
                    return False
            else:
                if np.linalg.norm(f1, ord=np.inf) <= rc:
                    return False
            # add ws_len constraints
            if (p[0] < 0 or p[0] > self.ws_len) or (p[1] < 0 or p[1] > self.ws_len):
                return False
        return True
    

    def compute_opt_traj(self, br_dict, x_init, ua_init):
        """
        This function computes an optimal trajectory using the learned br model.
        br_dict: brnet state dict, not NN model object
        """    
        # solve leader's OC problem with optimization and PMP
        x_opt, ua_opt = self.oc_opt(br_dict, x_init, ua_init)           # directly formulate problem and use opt solver
        if self.obj_oc(x_opt, ua_opt) <= 1e5:
            x_pmp, ua_pmp = self.oc_pmp(br_dict, x_opt, ua_opt)         # use PMP to refine the trajectory
        else:
            x_pmp, ua_pmp = self.oc_pmp(br_dict, x_init, ua_init)       # directly use x_init to refine the trajectory

        print("opt_traj = {}, pmp_traj = {}".format(self.obj_oc(x_opt, ua_opt), self.obj_oc(x_pmp, ua_pmp)))
        
        # if self.obj_oc(x_opt, ua_opt) <= self.obj_oc(x_pmp, ua_pmp):
        #     x_traj, ua_traj = x_opt, ua_opt
        # else:
        #     x_traj, ua_traj = x_pmp, ua_pmp
        
        if self.obj_oc(x_pmp, ua_pmp) is np.nan or self.obj_oc(x_pmp, ua_pmp) > 1e5:
            if self.obj_oc(x_pmp, ua_pmp) < 1e5:
                x_traj, ua_traj = x_opt, ua_opt
            else:
                raise Exception("no feasible sol in receding horizon control.")
                x_traj, ua_traj = x_init, ua_init
        else:
            x_traj, ua_traj = x_pmp, ua_pmp
        return x_traj, ua_traj


    def oc_opt(self, br_dict, x_init, ua_init):
        """
        This function solves oc using an optimization solver.
        decision var in optimization: X = [x_1, ..., x_T, ua_0, ..., ua_{T-1}]
        - x_traj = [x_0, ..., x_T]
        - ua_traj = [ua_0, ..., ua_{T-1}]
        """
        constr = []
        x0 = x_init[0, :]
        br = BRNet()
        br.load_state_dict(br_dict)

        # input constraints: 0 <= |a_t|^2 <= 1, t = 0,..., T-1
        def con_a(X):
            f = np.zeros(self.dimT)
            for i in range(self.dimT):
                i_a = self.dimT * self.dimx + i * self.dimua         # starting inded for a_t
                a_t = X[i_a: i_a+self.dimua]
                f[i] = a_t @ a_t
            return f
        def jac_con_a(X):
            jac = np.zeros((self.dimT, self.dimT*(self.dimua+self.dimx)))
            for i in range(self.dimT):
                i_a = self.dimT * self.dimx + i * self.dimua         # starting inded for a_t
                a_t = X[i_a: i_a+self.dimua]
                jac[i, i_a: i_a+self.dimua] = 2 * a_t
            return jac
        constr.append( NonlinearConstraint(con_a, np.zeros(self.dimT), np.ones(self.dimT), jac=jac_con_a) )

        # safety cosntraints |pA_t-obs_j| >= d_j and |pB_t-obs_j| >= d_j
        def con_d(X):
            obs_num = len(self.obs)
            fa, fb = np.zeros(self.dimT*obs_num), np.zeros(self.dimT*obs_num)
            for i in range(self.dimT):
                i_x = i * self.dimx             # starting inded for x_t
                pa_t, pb_t = X[i_x: i_x+self.dimxa], X[i_x+self.dimxa: i_x+self.dimx-1]
                for j in range(obs_num):
                    obs_j = self.obs[j]
                    xc, yc, rc, = obs_j[0], obs_j[1], obs_j[2]
                    x_scale, y_scale = obs_j[4], obs_j[5]
                    if obs_j[3] == 1:
                        fa[i*obs_num+j] = np.linalg.norm( np.diag([1/x_scale, 1/y_scale]) @ (pa_t-np.array([xc,yc])), ord=1 )
                        fb[i*obs_num+j] = np.linalg.norm( np.diag([1/x_scale, 1/y_scale]) @ (pb_t-np.array([xc,yc])), ord=1 )
                    elif obs_j[3] == 2:
                        fa[i*obs_num+j] = np.linalg.norm( np.diag([1/x_scale, 1/y_scale]) @ (pa_t-np.array([xc,yc])), ord=2 )
                        fb[i*obs_num+j] = np.linalg.norm( np.diag([1/x_scale, 1/y_scale]) @ (pb_t-np.array([xc,yc])), ord=2 )
                    else:
                        fa[i*obs_num+j] = np.linalg.norm( np.diag([1/x_scale, 1/y_scale]) @ (pa_t-np.array([xc,yc])), ord=np.inf )
                        fb[i*obs_num+j] = np.linalg.norm( np.diag([1/x_scale, 1/y_scale]) @ (pb_t-np.array([xc,yc])), ord=np.inf )
            f = np.concatenate( (fa,fb) )
            f = fa  # only for pa
            return f
        def jac_con_d(X):
            obs_num = len(self.obs)
            jac_a = np.zeros((self.dimT*obs_num,self.dimT*(self.dimx+self.dimua)))
            jac_b = np.zeros_like(jac_a)
            for i in range(self.dimT):
                i_x = i * self.dimx             # starting inded for x_t
                pa_t, pb_t = X[i_x: i_x+self.dimxa], X[i_x+self.dimxa: i_x+self.dimx-1]
                for j in range(obs_num):
                    obs_j = self.obs[j]
                    xc, yc, rc, = obs_j[0], obs_j[1], obs_j[2]
                    x_scale, y_scale = obs_j[4], obs_j[5]
                    Lam = np.diag([1/x_scale, 1/y_scale])
                    f1 = Lam @ (pa_t-np.array([xc,yc]))
                    f2 = Lam @ (pb_t-np.array([xc,yc]))
                    if obs_j[3] == 1:
                        jac_a[i*obs_num+j, i_x: i_x+self.dimxa] = np.sign(f1) @ Lam
                        jac_b[i*obs_num+j, i_x+self.dimxa: i_x+self.dimx-1] = np.sign(f2) @ Lam
                    elif obs_j[3] == 2:
                        jac_a[i*obs_num+j, i_x: i_x+self.dimxa] = (f1/np.linalg.norm(f1,ord=2)) @ Lam
                        jac_b[i*obs_num+j, i_x+self.dimxa: i_x+self.dimx-1] = (f2/np.linalg.norm(f2,ord=2)) @ Lam
                    else:
                        tmp = np.zeros_like(f1)
                        idx = np.argmax(np.abs(f1))
                        tmp[idx] = np.sign(f1[idx])
                        jac_a[i*obs_num+j, i_x: i_x+self.dimxa] = tmp @ Lam
                        tmp = np.zeros_like(f2)
                        idx = np.argmax(np.abs(f2))
                        tmp[idx] = np.sign(f2[idx])
                        jac_b[i*obs_num+j, i_x+self.dimxa: i_x+self.dimx-1] = tmp @ Lam
            jac = np.vstack( (jac_a,jac_b) )
            jac = jac_a     # only for pa
            return jac
        eps = 5e-1      # for numerical reasons
        #lb_d = np.kron(np.ones(self.dimT*2), np.array(self.obs)[:, 2]+eps)
        #ub_d = np.inf*np.ones(self.dimT*2*len(self.obs))
        lb_d = np.kron(np.ones(self.dimT), np.array(self.obs)[:, 2]+eps)    # only for pa
        ub_d = np.inf*np.ones(self.dimT*len(self.obs))
        constr.append( NonlinearConstraint(con_d, lb_d, ub_d, jac=jac_con_d) )

        # dynamics constraints ONLY LEADER xa_tp1 = xa_t + B1 @ a_t, linear constraints
        Aeq, beq = np.zeros((self.dimT*self.dimxa, self.dimT*(self.dimx+self.dimua))), np.zeros(self.dimT*self.dimxa)
        BB1 = self.dt * np.eye(self.dimua)
        for t in range(self.dimT):
            i_p = t * self.dimxa
            i_x = t * self.dimx
            i_a = t * self.dimua + self.dimT*self.dimx
            if t == 0:
                beq[i_p: i_p+self.dimxa] = -x0[0:self.dimxa]
                Aeq[i_p: i_p+self.dimxa, i_x: i_x+self.dimxa] = -np.eye(self.dimxa)             # for x_1
                Aeq[i_p: i_p+self.dimxa, i_a: i_a+self.dimua] = BB1      # for a_0
            else:
                Aeq[i_p: i_p+self.dimxa, i_x: i_x+self.dimxa] = -np.eye(self.dimxa)             # for x_tp1
                Aeq[i_p: i_p+self.dimxa, i_x-self.dimx: i_x-self.dimxb] = np.eye(self.dimxa)    # for x_t
                Aeq[i_p: i_p+self.dimxa, i_a: i_a+self.dimua] = BB1      # for a_t
        lc1 = LinearConstraint(Aeq, beq, beq)
        constr.append( lc1 )

        # stay in the region, 0 <= pa, pb <= 10
        #D = np.hstack( (np.eye(self.dimx-1), np.zeros((self.dimx-1,1))) )     # D@x_t = [pa, pb]
        D = np.hstack( (np.eye(self.dimxa), np.zeros((self.dimxa, self.dimx-self.dimxa))) )    # D@x_t = [pa], only for pa
        DD = np.kron(np.eye(self.dimT), D)
        #AA = np.kron(np.eye(self.dimT), np.zeros((self.dimx-1,self.dimua)))
        AA = np.kron(np.eye(self.dimT), np.zeros((self.dimxa,self.dimua)))  # only for pa
        Ain = np.hstack( (DD, AA) )
        #b1, b2 = np.zeros(self.dimx-1), self.ws_len*np.ones(self.dimx-1)
        b1, b2 = np.zeros(self.dimxa), self.ws_len*np.ones(self.dimxa)      # only for pa
        bin_l = np.kron(np.ones(self.dimT), b1)
        bin_u = np.kron(np.ones(self.dimT), b2)
        lc2 = LinearConstraint(Ain, bin_l, bin_u)
        constr.append( lc2 )

        # objective
        def J(X, mu):
            """
            Objective function: J_oc = obj_oc(X) + mu*|-xB_tp1 + xB_t + dt*fb(xB_tp1) @ br(x_t, a_t)|
            Let dif = -xB_tp1 + xB_t + dt*fb(xB_tp1) @ br(x_t, a_t)
            X[:idx] = [x_1,..., x_T], X[idx:] = [a_0, ..., a_{T-1}]
            """
            idx = self.dimT * self.dimx
            x = X[:idx].reshape((self.dimT, self.dimx))
            x = np.vstack( (x0[None,:], x) )
            ua = X[idx:].reshape((self.dimT, self.dimua))
            cost = self.obj_oc(x, ua)
            
            # soft dynamic constraint for xB
            #cost=0     # testing 
            fb = lambda xnew: np.array([[np.cos(xnew[-1]),0], [np.sin(xnew[-1]),0], [0,1]])
            D = np.hstack( (np.zeros((self.dimxb, self.dimxa)), np.eye(self.dimxb)) )     # D@x_t = xB_t
            for t in range(self.dimT):
                i_x = t * self.dimx
                i_a = self.dimT * self.dimx + t * self.dimua
                x_tp1 = X[i_x: i_x+self.dimx]
                ua_t = X[i_a: i_a+self.dimua]
                if t == 0:
                    dif = -D@x_tp1 + D@x0 + self.dt*fb(x_tp1) @ self.get_br(br, x0, ua_t)
                else:
                    x_t = X[i_x-self.dimx: i_x]
                    dif = -D@x_tp1 + D@x_t + self.dt*fb(x_tp1) @ self.get_br(br, x_t, ua_t)
                cost += mu * (dif @ dif)    
            return cost
        def dJ(X, mu):
            """
            J_oc = obj_oc(X) + mu*|-xB_tp1 + xB_t + dt*fb(xB_tp1) @ br(x_t, a_t)|
            Let dif = -xB_tp1 + xB_t + dt*fb(xB_tp1) @ br(x_t, a_t), g = fb(x_tp1) @ br(x_t, a_t)
            - dg/dat = self.dt*fb(x_tp1) @ jac_a
            - dg/dxt = self.dt*fb(x_tp1) @ jac_x
            - dg/dxtp1 = [br1*[0,0,0,0,-sin(phi_tp1)], br1*[0,0,0,0,cos(phi_tp1)], [0,0,0,0,0]]
            """
            idx = self.dimT * self.dimx
            x = X[:idx].reshape((self.dimT, self.dimx))
            x = np.vstack( (x0[None,:], x) )
            ua = X[idx:].reshape((self.dimT, self.dimua))
            grad_x, grad_a = self.grad_obj_oc(x, ua)
            grad = np.concatenate( (grad_x.reshape(grad_x.size), grad_a.reshape(grad_a.size)) )
            
            # gradient of soft dynamic constraint for xB
            #grad = np.zeros(self.dimT * (self.dimx+self.dimua))    # testing
            fb = lambda xnew: np.array([[np.cos(xnew[-1]),0], [np.sin(xnew[-1]),0], [0,1]])
            D = np.hstack( (np.zeros((self.dimxb, self.dimxa)), np.eye(self.dimxb)) )     # D@x_t = xB_t
            for i in range(self.dimT):
                i_x = i * self.dimx
                i_a = self.dimT * self.dimx + i * self.dimua
                x_tp1 = X[i_x: i_x+self.dimx]
                a_t = X[i_a: i_a+self.dimua]
                if i == 0:
                    b_t = self.get_br(br, x0, a_t)
                    jac_x, jac_a = self.get_br_jac(br, x0, a_t) 
                    dif = -D@x_tp1 + D@x0 + self.dt*fb(x_tp1) @ b_t
                    grad[i_x: i_x+self.dimx] += 2*mu * dif @ (-D \
                        + self.dt*np.array([[0,0,0,0,-np.sin(x_tp1[-1])*b_t[0]], [0,0,0,0,np.cos(x_tp1[-1])*b_t[0]], self.dimx*[0]]))  # df_0/dx_1, no df_0/dx_0
                    grad[i_a: i_a+self.dimua] += 2*mu * dif @ (self.dt*fb(x_tp1) @ jac_a)   # df_0/da_0
                else:
                    x_t = X[i_x-self.dimx: i_x]
                    b_t = self.get_br(br, x_t, a_t)
                    jac_x, jac_a = self.get_br_jac(br, x_t, a_t)
                    dif = -D@x_tp1 + D@x_t + self.dt*fb(x_tp1) @ b_t
                    grad[i_x-self.dimx: i_x] += 2*mu * dif @ (D + self.dt*fb(x_tp1) @ jac_x)    # df_t/dx_t
                    grad[i_x: i_x+self.dimx] += 2*mu * dif @ (-D \
                        + self.dt*np.array([[0,0,0,0,-np.sin(x_tp1[-1])*b_t[0]], [0,0,0,0,np.cos(x_tp1[-1])*b_t[0]], self.dimx*[0]]))  # df_t/dx_tp1
                    grad[i_a: i_a+self.dimua] += 2*mu * dif @ (self.dt*fb(x_tp1) @ jac_a)   # df_t/da_t
            return grad

        # formulate initial conditions
        x0_traj = x_init[1:, :]    # delete x0 in x_traj
        X0 = np.concatenate( (x0_traj.reshape(x0_traj.size), ua_init.reshape(ua_init.size)) )
        mu = 50
        #result = minimize(J, X0, args=(mu), constraints=constr, options={'maxiter': 300, 'disp': True})
        result = minimize(J, X0, jac=dJ, args=(mu), constraints=constr, options={'maxiter': 300, 'disp': True})

        # convert results to trajectories
        idx = self.dimT*self.dimx
        x_opt, ua_opt = result.x[: idx], result.x[idx: ]
        x_opt = np.concatenate((x0, x_opt))     # add x0 to trajectory
        return x_opt.reshape((self.dimT+1, self.dimx)), ua_opt.reshape((self.dimT, self.dimua))


    def oc_pmp(self, br_dict, x_init, ua_init):
        """
        This function solves oc using pmp.
        Apply log barrier to safety constraints. Follower's dynamics is not in the objective function.
        - x_traj: [x_0, x_1, ..., x_T], dim=T+1
        - ua_traj: [ua_0, ua_1, ..., ua_{T-1}], dim=T
        - lam_traj: [lam_1, lam_2, ..., lam_T], dim=T
        Note: in opt, we use cos(phi_tp1) in dynamics because phi_tp1 is a decision variable.
              in pmp, we need cos(phi_tp1+w*dt) in dynamics because we need to do backward propagation.
        """
        br = BRNet()
        br.load_state_dict(br_dict)
        x0 = x_init[0, :]
        xd = np.array( self.pd + self.pd + [0] )
        mu = 1/2

        def backward_T(x_T):
            """
            Terminal cost: |x_T-xd|^2_qf1 + |pa-pb|^2_qf2 - mu*log(|pa_T-pj|-dj) - mu*log(|pb_T-pj|-dj). 
            """
            D = np.hstack( (np.eye(self.dimxa), -np.eye(self.dimxb-1), np.zeros((self.dimxa,1))) )      # D @ x = pa - pb
            pa_T, pb_T = x_T[: self.dimxa], x_T[self.dimxb: -1]
            lam_T = 2*self.qf1 @ (x_T-xd) + 2*(D.T @ self.qf2 @ D) @ x_T

            # for soft safety constraints
            eps = 5e-1
            dJa_cost, dJb_cost = np.zeros(self.dimx), np.zeros(self.dimx)
            dpaT_dx = np.hstack( (np.eye(self.dimxa), np.zeros((self.dimxa,self.dimxb))) )
            dpbT_dx = np.hstack( (np.zeros((self.dimxa,self.dimxa)), np.eye(self.dimxb-1), np.zeros((self.dimxa,1))) )
            for i in range(len(self.obs)):
                obs_i = self.obs[i]
                xc, yc, rc = obs_i[0], obs_i[1], obs_i[2]
                x_scale, y_scale = obs_i[4], obs_i[5]
                Lam = np.diag([1/x_scale, 1/y_scale])
                f1 = Lam @ (pa_T-np.array([xc,yc]))     # leader's constraint
                f2 = Lam @ (pb_T-np.array([xc,yc]))     # follower's constraint
                if obs_i[3] == 1:
                    dJa_cost -= mu/(np.linalg.norm(f1, ord=1)-rc-eps) * np.sign(f1) @ Lam @ dpaT_dx
                    dJb_cost -= mu/(np.linalg.norm(f2, ord=1)-rc-eps) * np.sign(f2) @ Lam @ dpbT_dx
                elif obs_i[3] == 2:
                    dJa_cost -= mu/(np.linalg.norm(f1, ord=1)-rc-eps) * (f1/np.linalg.norm(f1, ord=1)) @ Lam @ dpaT_dx
                    dJb_cost -= mu/(np.linalg.norm(f2, ord=1)-rc-eps) * (f2/np.linalg.norm(f2, ord=1)) @ Lam @ dpbT_dx
                else:
                    tmp = np.zeros_like(f1)
                    idx = np.argmax(np.abs(f1))
                    tmp[idx] = np.sign(f1[idx])
                    dJa_cost -= mu/(np.linalg.norm(f1, ord=1)-rc-eps) * tmp @ Lam @ dpaT_dx
                    
                    tmp = np.zeros_like(f2)
                    idx = np.argmax(np.abs(f2))
                    tmp[idx] = np.sign(f2[idx])
                    dJb_cost -= mu/(np.linalg.norm(f2, ord=1)-rc-eps) * tmp @ Lam @ dpbT_dx
            lam_T += dJa_cost + dJb_cost
            lam_T += dJa_cost   # only for pa
            return lam_T

        def backward(x_t, ua_t, lam_tp1):
            """
            Running cost: |x_t-xd|^2_q1 + |pa-pb|^2_q2 - mu*log(|pA_t-pj|-dj) - mu*log(|pB_t-pj|-dj).
            Dynamics: f(x_t, a_t) = x_t + B1@a_t + B2@b_t + fb(x_t)b_t = x_t + B1@a_t + ff(x_t,b_t)
            """
            D = np.hstack( (np.eye(self.dimxa), -np.eye(self.dimxb-1), np.zeros((self.dimxa,1))) )  # D @ x = pa - pb
            pa_t, pb_t, phi_t = x_t[: self.dimxa], x_t[self.dimxa: -1], x_t[-1]
            b_t = self.get_br(br, x_t, ua_t)
            jac_x, _ = self.get_br_jac(br, x_t, ua_t)

            # running cost
            lam_t = 2*self.q1 @ (x_t-xd) + 2*(D.T @ self.q2 @ D) @ x_t

            # for soft safety constraints
            eps = 5e-1
            dJa_cost, dJb_cost = np.zeros(self.dimx), np.zeros(self.dimx)
            dpat_dx = np.hstack( (np.eye(self.dimxa), np.zeros((self.dimxa,self.dimxb))) )
            dpbt_dx = np.hstack( (np.zeros((self.dimxa,self.dimxa)), np.eye(self.dimxb-1), np.zeros((self.dimxa,1))) )
            for i in range(len(self.obs)):
                obs_i = self.obs[i]
                xc, yc, rc = obs_i[0], obs_i[1], obs_i[2]
                x_scale, y_scale = obs_i[4], obs_i[5]
                Lam = np.diag([1/x_scale, 1/y_scale])
                f1 = Lam @ (pa_t-np.array([xc,yc]))     # leader's constraint
                f2 = Lam @ (pb_t-np.array([xc,yc]))     # follower's constraint
                if obs_i[3] == 1:
                    dJa_cost -= mu/(np.linalg.norm(f1, ord=1)-rc-eps) * np.sign(f1) @ Lam @ dpat_dx
                    dJb_cost -= mu/(np.linalg.norm(f2, ord=1)-rc-eps) * np.sign(f2) @ Lam @ dpbt_dx
                elif obs_i[3] == 2:
                    dJa_cost -= mu/(np.linalg.norm(f1, ord=1)-rc-eps) * (f1/np.linalg.norm(f1, ord=1)) @ Lam @ dpat_dx
                    dJb_cost -= mu/(np.linalg.norm(f2, ord=1)-rc-eps) * (f2/np.linalg.norm(f2, ord=1)) @ Lam @ dpbt_dx
                else:
                    tmp = np.zeros_like(f1)
                    idx = np.argmax(np.abs(f1))
                    tmp[idx] = np.sign(f1[idx])
                    dJa_cost -= mu/(np.linalg.norm(f1, ord=1)-rc-eps) * tmp @ Lam @ dpat_dx
                    
                    tmp = np.zeros_like(f2)
                    idx = np.argmax(np.abs(f2))
                    tmp[idx] = np.sign(f2[idx])
                    dJb_cost -= mu/(np.linalg.norm(f2, ord=1)-rc-eps) * tmp @ Lam @ dpbt_dx
            lam_t += dJa_cost + dJb_cost
            lam_t += dJa_cost   # only for pa
            
            # dynamics related, ff = dt*[ua_t, fb(x_t,ua_t) ], fb=[cos(phi+w*dt)*v, sin(phi+w*dt)*v, w], first term [xa_t, xb_t] not included.
            dff_dx = np.zeros((self.dimx, self.dimx))
            dff_dx[2,:] = np.cos(phi_t+self.dt*b_t[1]) * jac_x[0, :] + b_t[0] * np.array([0,0,0,0,-np.sin(phi_t+self.dt*b_t[1])])
            dff_dx[3,:] = np.sin(phi_t+self.dt*b_t[1]) * jac_x[0, :] + b_t[0] * np.array([0,0,0,0, np.cos(phi_t+self.dt*b_t[1])])
            dff_dx[4,:] = jac_x[1, :]
            lam_t += lam_tp1 @ (np.eye(self.dimx) + self.dt*dff_dx)
            return lam_t

        def forward(x_t, ua_t):     # forward dynamics
            x, ua, ub = x_t, ua_t, self.get_br(br, x_t, ua_t)
            fb = lambda x, ub: np.array([[np.cos(x[-1]+self.dt*ub[-1]),0], [np.sin(x[-1]+self.dt*ub[-1]),0], [0,1]])
            x_new = x + self.dt * np.concatenate( (ua, fb(x,ub) @ ub) )
            return x_new

        def H_t_grad(x_t, ua_t, lam_tp1):    # gradient of hamiltonian H_t
            phi_t = x_t[-1]
            b_t = self.get_br(br, x_t, ua_t)
            _, jac_a = self.get_br_jac(br, x_t, ua_t)

            # dynamics related, ff = dt*[uA, fb(x_t,a_t) ], fb=[cos(phi+w*dt)*v, sin(phi+w*dt)*v, w], first term [xa_t, xb_t] not included.
            dff_dua = np.zeros((self.dimx, self.dimua))
            dff_dua[:self.dimua, :self.dimua] = np.eye(self.dimua)
            dff_dua[2,:] = np.cos(phi_t+self.dt*b_t[1]) * jac_a[0, :] + b_t[0] * (-np.sin(phi_t+self.dt*b_t[1])) * self.dt*jac_a[1, :]
            dff_dua[3,:] = np.sin(phi_t+self.dt*b_t[1]) * jac_a[0, :] + b_t[0] * (+np.cos(phi_t+self.dt*b_t[1])) * self.dt*jac_a[1, :]
            dff_dua[4,:] = jac_a[1, :]
            grad = 2*self.q3 @ ua_t + lam_tp1 @ (self.dt*dff_dua)
            return grad

        x_traj, lam_traj = np.zeros((self.dimT+1,self.dimx)), np.zeros((self.dimT,self.dimx))
        ua_traj = np.zeros((self.dimT, self.dimua))
        dH = np.zeros((self.dimT, self.dimua))    # axis=0 is control horizon (dimT)
        x0 = x_init[0,:]
        ua_pre = ua_init.copy()   # use copy. otherwise ua_init changes with ua_pre
        ITER_MAX = 10000        # maximum iteration
        tol = 4e-5              # GD tolarance
        step = 6e-4             # GD step size
        cost_pre = np.inf
        for i in range(ITER_MAX):
            # forward pass to update x_traj
            x_traj[0, :] = x0
            for t in range(self.dimT):
                x_traj[t+1, :] = forward(x_traj[t,:], ua_pre[t,:])
            
            # backward pass to update lam_traj
            for t in reversed(range(self.dimT)):
                if t == self.dimT-1:
                    lam_traj[t, :] = backward_T(x_traj[t+1, :])   # use x_T
                else:
                    lam_traj[t, :] = backward(x_traj[t+1, :], ua_pre[t+1, :], lam_traj[t+1, :])  # use x_t, a_t, p_tp1

            # compute gradient of Hamultonian, axis=0 is control horizon (dimT)
            for t in range(self.dimT):
                dH[t, :] = H_t_grad(x_traj[t, :], ua_traj[t, :], lam_traj[t, :])

            # update a_traj with GD, |ua| <= 1
            for t in range(self.dimT):
                a = ua_pre[t, :] - step * dH[t, :]
                ua_traj[t, :] = a if a@a<=1 else a/np.linalg.norm(a)
                #ua_traj[t, :] =  ua_pre[t, :] - step * dH[t, :]
            
            # stopping condition
            ua_diff = np.linalg.norm(ua_traj-ua_pre, axis=1).max()
            if np.linalg.norm(ua_traj-ua_pre, axis=1).max() < tol:
                break
            #if np.linalg.norm(dH, axis=1).max() < tol:
            #    break
            if cost_pre <= self.obj_oc(x_traj, ua_traj):
                break
            else:
                cost_pre = self.obj_oc(x_traj, ua_traj)

            ua_pre = ua_traj.copy()
            if (i+1) % 100 == 0:
                print('-- pmp iter {}, leader cost: {:.5f}, |dua|: {:.3e}, |dH|: {:.5f}'.format(i+1, self.obj_oc(x_traj, ua_traj), ua_diff, np.linalg.norm(dH, axis=1).max()))
        return x_traj, ua_traj


    def obj_oc(self, x_traj, ua_traj):
        """
        This function computes the leader's control cost given trajectories.
        """
        cost = 0
        xd = np.array( self.pd + self.pd + [0] )
        for i in range(self.dimT):
            pa, pb = x_traj[i+1, :self.dimxa], x_traj[i+1, self.dimxa:-1]
            if i == self.dimT-1:
                cost += (x_traj[i+1, :] - xd) @ self.qf1 @ (x_traj[i+1, :] - xd)
                cost += (pa - pb) @ self.qf2 @ (pa - pb)
            else:
                cost += (x_traj[i+1, :] - xd) @ self.q1 @ (x_traj[i+1, :] - xd)
                cost += (pa - pb) @ self.q2 @ (pa - pb)
            cost += ua_traj[i, :] @ self.q3 @ ua_traj[i, :]
        return cost


    def grad_obj_oc(self, x_traj, ua_traj):
        """
        This function computesthe gradient of the leader's cost function.
        grad_x[i, :] is dJ_dxi, i = 1,..., T
        """
        grad_x, grad_a = np.zeros((self.dimT, self.dimx)), np.zeros((self.dimT, self.dimua))
        xd = np.array( self.pd + self.pd + [0] )
        D = np.hstack( (np.eye(self.dimxa), -np.eye(self.dimxb-1), np.zeros((self.dimxa,1))) )    # D @ x = pa - pb
        for i in range(self.dimT):
            if i == self.dimT-1:
                grad_x[i, :] += 2* self.qf1 @ (x_traj[i+1, :] - xd)
                grad_x[i, :] += 2* (D.T @ self.qf2 @ D) @ x_traj[i+1, :]
            else:
                grad_x[i, :] += 2* self.q1 @ (x_traj[i+1, :] - xd)
                grad_x[i, :] += 2* (D.T @ self.q2 @ D) @ x_traj[i+1, :]
            grad_a[i, :] = 2 * self.q3 @ ua_traj[i, :]
        # or return other forms
        return grad_x, grad_a
    

    def init1(self, x0=None): 
        """
        initialize x_traj and ua_traj by finding projection. compute a_traj. phi is gradient
        """
        def project_x(p):
            """
            This function projects p that is inside the obstacle to the outside of the obstacle (use 1.2*rc).
            Do projection as if xc=yc=0, then do translation (xc,rc) to get real position.
            """
            p_new = p.copy()
            for i in range(len(self.obs)):
                obs_i = self.obs[i]
                xc, yc, rc, = obs_i[0], obs_i[1], obs_i[2]
                x_scale, y_scale = obs_i[4], obs_i[5]
                f1 = np.diag([1/x_scale, 1/y_scale]) @ (p-np.array([xc,yc])) 
                z = p - np.array([xc,yc])       # translated coordinates
                if obs_i[3] == 1:    # new point in |Lam @ z_new|_1 = 1.3*rc
                    if np.linalg.norm(f1, ord=1) <= rc:
                        tx = x_scale * (1.3*rc - z[1]/y_scale)
                        ty = y_scale * (1.3*rc - z[0]/x_scale)
                        if tx < ty:
                            z_new = np.array([z[0]+np.sign(z[0])*tx, z[1]]) # project x coordinate
                        else:
                            z_new = np.array([z[0], z[1]+np.sign(z[1])*ty]) # project y coordinate
                        p_new = z_new + np.array([xc,yc])
                        break
                elif obs_i[3] == 2: # new point in |Lam @ z_new|_2 = 1.3*rc
                    if np.linalg.norm(f1, ord=2) <= rc:
                        Lam = np.diag([1/x_scale, 1/y_scale])
                        k = 1.3*rc / np.linalg.norm(Lam@z)
                        z1, z2 = k*z, -k*z
                        if np.linalg.norm(z1-z) < np.linalg.norm(z2-z):
                            p_new = z1 + np.array([xc, yc])
                        else:
                            p_new = z2 + np.array([xc, yc])
                        break
                else:
                    if np.linalg.norm(f1, ord=np.inf) <= rc:
                        #p_new = p
                        if p[1] >= yc: 
                            p_new[1] = yc + y_scale * 1.3*rc
                        else:
                            p_new[1] = yc - y_scale * 1.3*rc
                        break
            return p_new
        
        if x0 is None:
            x0 = np.random.rand(self.dimx)
        # find leader's traj
        xa, ua_traj = np.zeros((self.dimT+1, self.dimxa)), np.zeros((self.dimT, self.dimua))
        xa[0, :] = x0[:self.dimxa]
        xb = np.zeros((self.dimT+1, self.dimxb))
        xb[0,:] = x0[self.dimxa:]
        dxa = (self.pd - xa[0,:]) / self.dimT       # leader's incremental distance
        dxb = (self.pd - xb[0,:-1]) / self.dimT     # follower's incremental distance
        # linear allocation
        for t in range(self.dimT):
            # find xa and ua_traj
            xa[t+1,:] = xa[t,:] + dxa
            ua_traj[t,:] = dxa / np.linalg.norm(dxa)    # dt does not matter
            # find xb, including phib
            xb[t+1,:-1] = xb[t,:-1] + dxb
            xb[t+1,-1] = np.arctan(dxb[1]/dxb[0])       # dt does not matter
        # project unsafe points
        for t in range(self.dimT):
            pa = xa[t+1,:]     # initial condition is safe
            if not self.is_safe(pa):
                xa[t+1,:] = project_x(pa)
                tmp = xa[t+1,:] - xa[t,:]
                ua_traj[t,:] = tmp / np.linalg.norm(tmp)
            pb = xb[t+1,:-1]
            if not self.is_safe(pb):
                xb[t+1,:-1] = project_x(pb)
                tmp = xb[t+1,:-1] - xb[t,:-1]
                xb[t+1,-1] = np.arctan(tmp[1]/tmp[0])
                tmp = xb[t+2,:-1] - xb[t+1,:-1]
                xb[t+2,-1] = np.arctan(tmp[1]/tmp[0])
        x_traj = np.hstack((xa, xb))    # dim=0 is time horizon
        return x_traj, ua_traj
    

    def init2(self, x0=None):
        """
        initialize x_traj and ua_traj by solving an simplified oc. compute a_traj. phi is gradient.
        Treat leader/follower as a single integrator. Obj is |p-pd|+|u|. Position constraints.
        X = [x1, x2, ..., u1, u2, ...]
        """
        if x0 is None:
            x0 = np.random.rand(self.dimx)
        
        constr = []
        def J(X):
            q, r = np.eye(self.dimxa), self.q3
            Q, R = np.kron(np.eye(self.dimT), q), np.kron(np.eye(self.dimT), r)
            H = np.block([[Q, np.zeros((self.dimT*self.dimxa,self.dimT*self.dimua))], \
                        [np.zeros((self.dimT*self.dimua,self.dimT*self.dimxa)), R]])
            PD = np.kron(np.ones(self.dimT), self.pd)
            f = -2 * np.concatenate( (PD@Q, np.zeros(self.dimT*self.dimua)) )
            cost = X @ H @ X + f @ X + PD @ Q @ PD
            return cost
        def con_a(X):
            f = np.zeros(self.dimT)
            for i in range(self.dimT):
                i_a = self.dimT * self.dimxa + i * self.dimua         # starting inded for a_t, note x is xa.
                a_t = X[i_a: i_a+self.dimua]
                f[i] = a_t @ a_t
            return f
        constr.append( NonlinearConstraint(con_a, np.zeros(self.dimT), 2*np.ones(self.dimT)) )
        def con_obs(X):
            obs_num = len(self.obs)
            f = np.zeros(self.dimT*obs_num)
            for t in range(self.dimT):
                i_x = t * self.dimxa        # starting inded for x_t, x is xa
                pa_t = X[i_x: i_x+self.dimxa]
                for i in range(obs_num):
                    obs_i = self.obs[i]
                    xc, yc, rc, = obs_i[0], obs_i[1], obs_i[2]
                    x_scale, y_scale = obs_i[4], obs_i[5]
                    if obs_i[3] == 1:
                        f[t*obs_num+i] = np.linalg.norm( np.diag([1/x_scale, 1/y_scale]) @ (pa_t-np.array([xc,yc])), ord=1 )
                    elif obs_i[3] == 2:
                        f[t*obs_num+i] = np.linalg.norm( np.diag([1/x_scale, 1/y_scale]) @ (pa_t-np.array([xc,yc])), ord=2 )
                    else:
                        f[t*obs_num+i] = np.linalg.norm( np.diag([1/x_scale, 1/y_scale]) @ (pa_t-np.array([xc,yc])), ord=np.inf )
            return f
        def con_obs_jac(X):
            obs_num = len(self.obs)
            jac = np.zeros((self.dimT*obs_num, self.dimT*(self.dimxa+self.dimua)))
            for t in range(self.dimT):
                i_x = t * self.dimxa        # starting inded for x_t, x is xa
                pa_t = X[i_x: i_x+self.dimxa]
                for i in range(obs_num):
                    obs_i = self.obs[i]
                    xc, yc, rc, = obs_i[0], obs_i[1], obs_i[2]
                    x_scale, y_scale = obs_i[4], obs_i[5]
                    Lam = np.diag([1/x_scale, 1/y_scale])
                    f1 = np.diag([1/x_scale, 1/y_scale]) @ (pa_t-np.array([xc,yc]))
                    if obs_i[3] == 1:
                        jac[t*obs_num+i, i_x: i_x+self.dimxa] = np.sign(f1) @ Lam
                    elif obs_i[3] == 2:
                        jac[t*obs_num+i, i_x: i_x+self.dimxa] = f1 / np.linalg.norm(f1) @ Lam
                    else:
                        tmp = np.zeros_like(f1)
                        idx = np.argmax(np.abs(f1))
                        tmp[idx] = np.sign(f1[idx])
                        jac[t*obs_num+i, i_x: i_x+self.dimxa] = tmp @ Lam
            return jac
        lb_obs = 1.1*np.kron(np.ones(self.dimT), np.array(self.obs)[:,2])
        ub_obs = np.inf*np.ones(self.dimT*len(self.obs))
        constr.append( NonlinearConstraint(con_obs, lb_obs, ub_obs) )

        # single integrator dynamics
        a1, a2 = np.eye(self.dimxa), np.eye(self.dimua)
        A = np.kron( np.diag(np.ones(self.dimT)), -a1) + np.kron(np.diag(np.ones(self.dimT-1),k=-1), a1)
        B = np.kron( np.diag(np.ones(self.dimT)), a2*self.dt)
        Aeq = np.hstack((A,B))
        Beq = np.concatenate( (-x0[:self.dimxa], np.zeros(self.dimxa*(self.dimT-1))) )
        constr.append( LinearConstraint(Aeq, Beq, Beq) )

        # bounds
        Ain = np.hstack( (np.kron(np.eye(self.dimT), a1), np.kron(np.eye(self.dimT), np.zeros_like(a2))) )
        bin_lb = np.kron(np.ones(self.dimT), np.zeros(self.dimxa))
        bin_ub = np.kron(np.ones(self.dimT), np.ones(self.dimua)*self.ws_len)
        #constr.append( LinearConstraint(Ain, bin_lb, bin_ub) )

        # find leader's trajectory
        x0_traj, ua0_traj = self.init1(x0)
        xa0_traj = x0_traj[1:, :self.dimxa]
        X0 = np.concatenate( (xa0_traj.reshape(xa0_traj.size), ua0_traj.reshape(ua0_traj.size)) )
        result = minimize(J, X0, constraints=constr, options={'maxiter': 400, 'disp': False})
        xx = result.x
        xa_traj = xx[:self.dimT*self.dimxa].reshape((self.dimT, self.dimxa))
        xa_traj = np.vstack( (x0[None,:self.dimxa], xa_traj) )
        ua_traj = xx[self.dimT*self.dimxa:].reshape((self.dimT, self.dimua))

        # perform same thing to find the follower's trajectory, xb_traj is enough
        constr = []
        constr.append( NonlinearConstraint(con_a, np.zeros(self.dimT), np.ones(self.dimT)) )
        constr.append( NonlinearConstraint(con_obs, lb_obs, ub_obs) )
        Beq = np.concatenate( (-x0[self.dimxa:-1], np.zeros(self.dimxa*(self.dimT-1))) )
        constr.append( LinearConstraint(Aeq, Beq, Beq) )
        #constr.append( LinearConstraint(Ain, bin_lb, bin_ub) )
        xb0_traj = x0_traj[1:, self.dimxa:-1]
        X0 = np.concatenate( (xb0_traj.reshape(xb0_traj.size), ua0_traj.reshape(ua0_traj.size)) )
        result = minimize(J, X0, constraints=constr, options={'maxiter': 400, 'disp': False})
        xx = result.x
        xb_traj = xx[:self.dimT*self.dimxa].reshape((self.dimT, self.dimxa))
        xb_traj = np.vstack( (x0[None,self.dimxa:-1], xb_traj) )
        phi_traj = np.zeros((self.dimT+1, 1))
        phi_traj[0] = x0_traj[0, -1]
        for i in range(self.dimT):
            tmp = xb_traj[i+1,:] - xb_traj[i,:]
            phi_traj[i+1,0] = np.arctan(tmp[1]/tmp[0])
        x_traj = np.hstack( (xa_traj, np.hstack( (xb_traj,phi_traj))) )
        return x_traj, ua_traj


    def get_br(self, br, x, a):
        """
        This function converts (x,a) to torch tensor and returns br(x,a) to numpy array. Make codes more concise.
        """
        return br(self.to_torch(x), self.to_torch(a)).detach().numpy().astype(float)


    def get_br_jac(self, br, x, a):
        """
        This function converts (x,a) to torch tensor and returns the jacobian of br(x,a) w.r.t. (x,a) to numpy array. 
        Make codes more concise.
        """
        jac_x, jac_a = br.compute_input_jac(self.to_torch(x), self.to_torch(a))
        return jac_x.numpy().astype(float), jac_a.numpy().astype(float)


    def to_torch(self, x):
        """
        This function convert numpy data array x to the torch tensor.
        """
        return torch.from_numpy(x).double()
        #return torch.from_numpy(x).float()



class Follower:
    """
    This class defines the follower's utility functions.
    """
    def __init__(self, theta) -> None:
        fname = os.path.join(config_data_dir, 'parameters.json')
        param = json.load(open(fname))
        
        self.ws_len, self.obs, self.pd = param['ws_length'], param['obstacle_settings'], param['destination']
        
        self.dimx, self.dimxa, self.dimxb = param['dimx'], param['dimxa'], param['dimxb']
        self.dimua, self.dimub = param['dimua'], param['dimub']
        self.dt = param['dt']
        self.coeff = param['follower_cost_coefficient'][f'type{theta}']
    

    def is_safe(self, p):
        """
        This function checks the safety constraints. obstacle collision, outside working space.
        """
        for i in range(len(self.obs)):
            obs_i = self.obs[i]
            xc, yc, rc = obs_i[0], obs_i[1], obs_i[2]
            x_scale, y_scale = obs_i[4], obs_i[5]
            f1 = np.diag([1/x_scale, 1/y_scale]) @ (p-np.array([xc,yc])) 
            if obs_i[3] == 1:
                if np.linalg.norm(f1, ord=1) <= rc:
                    return False
            elif obs_i[3] == 2:
                if np.linalg.norm(f1, ord=2) <= rc:
                    return False
            else:
                if np.linalg.norm(f1, ord=np.inf) <= rc:
                    return False
            # maybe add ws_len constraints
            if (p[0] < 0 or p[0] > self.ws_len) or (p[1] < 0 or p[1] > self.ws_len):
                return False
        return True


    def dynamics(self, x, ua, ub):
        """
        This function defines the system dynamics. The follower first make turns then move forward.
        """
        fb = lambda x, ub: np.array([[np.cos(x[-1]+self.dt*ub[-1]),0], [np.sin(x[-1]+self.dt*ub[-1]),0], [0,1]])
        x_new = x + self.dt * np.concatenate( (ua, fb(x,ub) @ ub) )
        return x_new
    

    def compute_J(self, x, ua, ub):
        """
        This function computes follower's cost. 
        Four basis functions with corresponding parameters.
        """
        x_new = self.dynamics(x, ua, ub)
        pa_new, pb_new = x_new[:self.dimxa], x_new[self.dimxa:-1]
        mu = 10     # barrier parameter

        J = 0
        J += self.coeff[0] * (pb_new-self.pd) @ (pb_new-self.pd)    # destination cost
        J += self.coeff[1] * (pa_new-pb_new) @ (pa_new-pb_new)      # guidance cost
        J += self.coeff[2] * ub @ ub    # control cost
        J_obs = 0   # obstacle cost
        for i in range(len(self.obs)):
            obs_i = self.obs[i]
            xc, yc, rc = obs_i[0], obs_i[1], obs_i[2]
            x_scale, y_scale = obs_i[4], obs_i[5]
            f1 = np.diag([1/x_scale, 1/y_scale]) @ (pb_new-np.array([xc,yc]))
            if obs_i[3] == 1:
                f2 = np.linalg.norm(f1, ord=1)
            elif obs_i[3] == 2:
                f2 = np.linalg.norm(f1, ord=2)
            else:
                f2 = np.linalg.norm(f1, ord=np.inf)
            # check safety margin
            if self.coeff[3]*(f2-rc) <= 0:
                tmp = 1
                print("obs {} penerates log barrier, pb={}, pb_new={}.".format(i,x[2:4], pb_new))
                return np.inf
            if self.coeff[3]*(f2-rc) <= 1:
                J_obs -= np.log(self.coeff[3]*(f2-rc))
        J += J_obs * mu
        return J
    

    def compute_dJ(self, x, ua, ub):
        """
        This function computes the jacobian of the follower's cost w.r.t. ub.
        - f = |y(x)|_1, df/dx = sign(y) @ dy/dx
        - f = |y(x)|_2, df/dx = y/|y|_2 @ dy/dx
        - f = |y(x)|_inf, df/dx = [0,.., y_i*,..,0] @ dy/dx, i* = argmax|y_i|
        """
        x_new = self.dynamics(x, ua, ub)
        pa_new, pb_new, phib_new = x_new[:self.dimxa], x_new[self.dimxa:-1], x_new[-1]
        mu = 10     # barrier parameter
        dpbnew_dub = np.array([[np.cos(phib_new), -np.sin(phib_new)*self.dt*ub[0]], [np.sin(phib_new), np.cos(phib_new)*self.dt*ub[0]]]) * self.dt
        
        dJ = np.zeros(self.dimub)
        dJ += 2 * self.coeff[0] * (pb_new-self.pd) @ dpbnew_dub    # jac of destination cost
        dJ += 2 * self.coeff[1] * (pa_new-pb_new) @ (-dpbnew_dub)  # jac of guidance cost
        dJ += 2 * self.coeff[2] * ub
        dJ_obs = np.zeros(self.dimub)   # jac of obstacle cost
        for i in range(len(self.obs)):
            obs_i = self.obs[i]
            xc, yc, rc = obs_i[0], obs_i[1], obs_i[2]
            x_scale, y_scale = obs_i[4], obs_i[5]
            Lam = np.diag([1/x_scale, 1/y_scale])
            f1 = Lam @ (pb_new-np.array([xc,yc]))
            if obs_i[3] == 1:
                f2 = np.linalg.norm(f1, ord=1)
                if self.coeff[3]*(f2-rc) <= 1:
                    dJ_obs -= 1/(f2-rc) * np.sign(f1) @ Lam @ dpbnew_dub    # for l1 norm
            elif obs_i[3] == 2:
                f2 = np.linalg.norm(f1, ord=2)
                if self.coeff[3]*(f2-rc) <= 1:
                    dJ_obs -= 1/(f2-rc) * (f1/f2) @ Lam @ dpbnew_dub        # for l2 norm
            else:
                f2 = np.linalg.norm(f1, ord=np.inf)
                if self.coeff[3]*(f2-rc) <= 1:
                    tmp = np.zeros_like(f1)
                    idx = np.argmax(np.abs(f1))
                    tmp[idx] = np.sign(f1[idx])
                    dJ_obs -= 1/(f2-rc) * tmp @ Lam @ dpbnew_dub            # for linf norm
        dJ += dJ_obs * mu
        """
        # check gradient using forward prediction
        eps=1.4901161193847656e-08
        eps1, eps2 = np.array([eps, 0]), np.array([0, eps])
        tmp = [(self.compute_J(x,ua,ub+eps1)-self.compute_J(x,ua,ub)) / eps, (self.compute_J(x,ua,ub+eps2)-self.compute_J(x,ua,ub)) / eps]
        print(dJ, tmp)
        print(dJ - tmp)
        """
        return dJ
    

    def get_br(self, x, ua):
        """
        This function computes the best response given the current state x and the leader's control ua.
        """
        def J(ub, x,ua):    # follower's objective function
            return self.compute_J(x, ua, ub)

        def dJ(ub, x,ua):   # gradient 
            return self.compute_dJ(x, ua, ub)
        
        lc1 = LinearConstraint(np.eye(self.dimub), np.array([-0.,-1.]), np.array([1.,1.]))   # [-1,-1] <= (vB,wB) <= [1,1]

        # run multiple optimization to get a better local optimal solution
        ITER = 10
        J_list, ub_list = np.zeros(ITER), np.zeros((ITER, self.dimub))
        for i in range(ITER):
            # always use a safe initial condition
            while True:
                ub_init = np.random.rand(self.dimub)
                #ub_init[1] = ub_init[1] * 2 - 1
                ub_init = ub_init * 2 - 1
                x_pred = self.dynamics(x, ua, ub_init)
                if self.is_safe(x_pred[2:4]):
                    break
            result = minimize(J, ub_init, args=(x,ua), jac=dJ, constraints=[lc1], tol=1e-7, options={'disp':False})
            J_list[i], ub_list[i, :] = result.fun, result.x
        ub_opt = ub_list[np.argmin(J_list), :]
        """
        # check gradient using forward prediction
        from scipy.optimize import check_grad
        tmp = np.zeros(10)
        for i in range(tmp.shape[0]):
            tmp[i] = check_grad(J, dJ, np.random.rand(self.dimub), x,ua)
        print(tmp)
        """
        return ub_opt
        