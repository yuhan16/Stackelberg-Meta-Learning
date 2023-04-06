"""
This script implements different functions in the sg meta-learning algorithm.
"""
import torch
import numpy as np
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import param_settings as param


class Leader:
    """
    This class defines the leader's utility functions.
    """
    def __init__(self) -> None:
        self.rng = param.rng
        self.dimx, self.dimxa, self.dimxb = param.dimx, param.dimxa, param.dimxb
        self.dimua, self.dimub = param.dimua, param.dimub
        self.dimT, self.dt, self.T = param.dimT, param.dt, param.T
        self.ws_len, self.obs, self.pd = param.ws_len, param.obs, param.pd
        self.q1, self.q2, self.q3 = param.q1, param.q2, param.q3
        self.qf1, self.qf2 = param.qf1, param.qf2
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
            # maybe add ws_len constraints
            if (p[0] < 0 or p[0] > self.ws_len) or (p[1] < 0 or p[1] > self.ws_len):
                return False
        return True
    
    def compute_opt_traj(self, br_dict, x_init=None, ua_init=None):
        """
        This function computes an optimal trajectory using the learned br model.
        br_dict: brnet state dict, not NN model object
        """    
        if x_init is None:
            x0 = self.rng.random(self.dimx)
            # or use other random initial conditions
            x0_list = np.array(param.x0_list)
            x0 = x0_list[self.rng.choice(x0_list.shape)]
            x_init, ua_init = self.init2(x0)

        #x_init, ua_init = self.init2(x0)
        # solve leader's OC problem with optimization and PMP
        x_opt, ua_opt = self.oc_opt(br_dict, x_init, ua_init)        # directly formulate problem and use opt solver
        if self.obj_oc(x_opt, ua_opt) <= 1e5:
            x_pmp, ua_pmp = self.oc_pmp(br_dict, x_opt, ua_opt)          # use PMP to refine the trajectory
        else:
            x_pmp, ua_pmp = self.oc_pmp(br_dict, x_init, ua_init)         # directly use x_init to refine the trajectory

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
            #return csr_matrix(jac)
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
            #return csr_matrix(jac)
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
        initialize x_traj by finding projection. compute a_traj. phi is gradient
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
        initialize x_traj by solving an simplified oc. compute a_traj. phi is gradient.
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
        constr.append(  NonlinearConstraint(con_a, np.zeros(self.dimT), 2*np.ones(self.dimT)) )
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
        self.dimx, self.dimxa, self.dimxb = param.dimx, param.dimxa, param.dimxb
        self.dimua, self.dimub = param.dimua, param.dimub
        self.dt = param.dt
        self.ws_len, self.obs, self.pd = param.ws_len, param.obs, param.pd
        self.coeff = param.get_follower_param(theta)
    
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
        mu = 10
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
        J += J_obs * mu     # increase cost
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
        mu = 10
        dpbnew_dub = np.array([[np.cos(phib_new), -np.sin(phib_new)*self.dt*ub[0]], [np.sin(phib_new), np.cos(phib_new)*self.dt*ub[0]]]) * self.dt
        #dd = np.array([[np.cos(phib_new), -np.sin(phib_new)], [np.sin(phib_new), np.cos(phib_new)]]) @ np.diag([self.dt, (self.dt**2)*ub[0]])
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
        dJ += dJ_obs * mu   # increase cost
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
        

class Meta(Leader):
    """
    This function defines meta-learning related functions. Inherited from the Leader class
    """
    def __init__(self) -> None:
        super().__init__()
        self.total_type = param.total_type
        self.type_pdf = param.type_pdf
        self.alp, self.beta, self.mom = param.alp, param.beta, param.mom
        self.kappa = param.kappa

    def sample_tasks(self, N):
        """
        This function samples N types of followers using the type distribution.
        """
        return np.random.choice(self.total_type, N, p=self.type_pdf)
    
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
        if len(data) == 0:   # data set empty, generate new data
            rng = np.random.default_rng(seed=None)  # or use leader's global rng
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
        if len(data) == 0:  # data set empty, generate new data
            rng = np.random.default_rng(seed=None)  # or use leader's global rng
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
            """
            k = N // 2  # sample k nearest br data for each obstacle
            for i in range(len(self.obs)):
                obs_i = self.obs[i]
                xc, yc, rc = obs_i[0], obs_i[1], obs_i[2]
                tmpa = data[:, :self.dimxa] - np.kron(np.ones((data.shape[0],1)), np.array([xc, yc]))
                tmpb = data[:, self.dimxa: self.dimx-1] - np.kron(np.ones((data.shape[0],1)), np.array([xc, yc]))
                if obs_i[3] == 1:
                    tmpa = np.linalg.norm(tmpa, ord=1, axis=1)
                    tmpb = np.linalg.norm(tmpb, ord=1, axis=1)
                elif obs_i[3] == 2:
                    tmpa = np.linalg.norm(tmpa, ord=2, axis=1)
                    tmpb = np.linalg.norm(tmpb, ord=2, axis=1)
                else:
                    tmpa = np.linalg.norm(tmpa, ord=np.inf, axis=1)
                    tmpb = np.linalg.norm(tmpb, ord=np.inf, axis=1)
                if tmpa.shape[0] > k:
                    idx = np.argsort(tmpa+tmpb)[: k]
                    task_data.append( data[idx, :] )
                else:
                    task_data.append(data)
            task_data = np.vstack(task_data)
        # sample N br data from task_data
        if task_data.shape[0] < N:
            raise Exception("Sampled data less than the requirement. Change k.")
            """
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
        optimizer = torch.optim.SGD(br.parameters(), lr=self.alp, momentum=0*self.mom)
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


class BRNet(torch.nn.Module):   # 2 hidden layers
    """
    BRNet class defines and trains a NN for BR model.
    """
    def __init__(self) -> None:
        torch.set_default_dtype(torch.float64)
        super(BRNet, self).__init__()
        self.dimx = param.dimx
        self.dimua = param.dimua
        self.dimub = param.dimub
        #self.device = 'cpu'
        
        self.linear1 = torch.nn.Linear(self.dimx+self.dimua, 25)
        self.linear2 = torch.nn.Linear(25, 25)
        self.linear3 = torch.nn.Linear(25, self.dimub)
        self.activation = torch.nn.ReLU()   # or tanh or sigmoid
        
        # random initialization 
        self.linear1.weight.data.normal_(mean=0., std=.1)
        self.linear1.bias.data.normal_(0., .1)
        self.linear2.weight.data.normal_(0, .1)
        self.linear2.bias.data.normal_(0, .1)
        self.linear3.weight.data.normal_(0, .1)
        self.linear3.bias.data.normal_(0, .1)
        
        # constant initialization for testing
        # self.linear1.weight.data.fill_(.1)     
        # self.linear1.bias.data.fill_(.1)
        # self.linear2.weight.data.fill_(.1)
        # self.linear2.bias.data.fill_(.1)
        # self.linear3.weight.data.fill_(.1)
        # self.linear3.bias.data.fill_(.1)
        
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
        _ = self.forward(x, a)
        h1.remove()
        h2.remove()
        h3.remove()
        
        def d_activation(y):
            """
            This function computes derivative of activation functions. can be relu, tanh, sigmoid.
            Input is a 1d array, output is nxn matrix.
            """
            #df = torch.diag(1 - torch.tanh(y)**2)  # for tanh(x)
            df = torch.diag(1. * (y > 0))           # for relu(x)
            return df
        def d_normalize(y):
            """
            This function computes the derivative of normalization functions. can be sigmoid, tanh.
            """
            df = torch.diag(y*(1-y))    # for sigmoid, need dot product
            #df = torch.diag(1-y**2)     # for tanh
            return df
        p = self.state_dict()
        jac_x = p['linear3.weight'] @ d_activation(y[1]) @ p['linear2.weight'] @ d_activation(y[0]) @ p['linear1.weight'][:, : self.dimx]
        jac_a = p['linear3.weight'] @ d_activation(y[1]) @ p['linear2.weight'] @ d_activation(y[0]) @ p['linear1.weight'][:, self.dimx: ]
        return jac_x, jac_a
    
    def get_grad_dict(self):
        dp_dict = {}
        for n, p in self.named_parameters():
            dp_dict[n] = p.grad.detach().cpu()
        return dp_dict

    def get_zero_grad_dict(self):
        dp_dict = {} 
        for n, p in self.named_parameters():
            dp_dict[n] = torch.zeros_like(p.data)
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
        _ = self.forward(x, a)
        h1.remove()
        h2.remove()
        h3.remove()
        return y 
    

class BRNet1(torch.nn.Module):  # 1 hidden layer
    """
    BRNet class defines and trains a NN for BR model.
    """
    def __init__(self) -> None:
        torch.set_default_dtype(torch.float64)
        super(BRNet1, self).__init__()
        self.dimx = param.dimx
        self.dimua = param.dimua
        self.dimub = param.dimub
        #self.device = 'cpu'
        
        self.linear1 = torch.nn.Linear(self.dimx+self.dimua, 15)
        self.linear2 = torch.nn.Linear(15, self.dimub)
        self.activation = torch.nn.ReLU()   # or tanh or sigmoid
        
        # random initialization 
        self.linear1.weight.data.normal_(mean=0., std=.1)
        self.linear1.bias.data.normal_(0., .1)
        self.linear2.weight.data.normal_(0, .1)
        self.linear2.bias.data.normal_(0, .1)
        
        # constant initialization for testing
        # self.linear1.weight.data.fill_(.1)     
        # self.linear1.bias.data.fill_(.1)
        # self.linear2.weight.data.fill_(.1)
        # self.linear2.bias.data.fill_(.1)
        
    def forward(self, x, a):
        if x.ndim > 1:
            y = torch.cat((x, a), dim=1)
        else:
            y = torch.cat((x, a), dim=0)
        y = self.linear1(y)
        y = self.activation(y)
        y = self.linear2(y)
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
        _ = self.forward(x, a)
        h1.remove()
        
        def d_activation(y):
            """
            This function computes derivative of activation functions. can be relu, tanh, sigmoid.
            Input is a 1d array, output is n x n matrix.
            """
            #df = torch.diag(1 - torch.tanh(y)**2)  # for tanh(x)
            df = torch.diag(1. * (y > 0))           # for relu(x)
            return df
        p = self.state_dict()
        jac_x = p['linear2.weight'] @ d_activation(y[0]) @ p['linear1.weight'][:, : self.dimx]
        jac_a = p['linear2.weight'] @ d_activation(y[0]) @ p['linear1.weight'][:, self.dimx: ]
        return jac_x, jac_a
    
    def get_grad_dict(self):
        dp_dict = {}
        for n, p in self.named_parameters():
            dp_dict[n] = p.grad.detach().cpu()
        return dp_dict

    def get_zero_grad_dict(self):
        dp_dict = {} 
        for n, p in self.named_parameters():
            dp_dict[n] = torch.zeros_like(p.data)
        return dp_dict

    def get_intermediate_output(self, x, a):
        """
        This function gets the output of every Linear layer.
        """
        y = []
        def forward_hook(model, input, output):
            y.append( output.detach() )
        h1 = self.linear1.register_forward_hook(forward_hook)
        _ = self.forward(x, a)
        h1.remove()
        return y 


class PlotUtils:
    """
    This class defines plotting related functions.
    """
    def __init__(self) -> None:
        self.ws_len = param.ws_len
        self.obs = param.obs
        self.pd = param.pd
    
    def plot_env1(self):
        """
        This function plots the simulation environments.
        """
        fig, ax = plt.subplots()
        #fig.set_figwidth(4.8)
        ax = self.plot_obs(ax)
        print(fig.get_figwidth, fig.get_figheight)
        
        im = plt.imread('data/dest-logo.png')
        ax.imshow(im, extent=(8.5,9.5,8.5,9.5), zorder=-1)
        
        ax.set_xlim(0, 10.1)
        ax.set_ylim(0, 10.1)
        ax.set_aspect(1)
        ax.xaxis.set_tick_params(labelsize='large')
        ax.yaxis.set_tick_params(labelsize='large')
        # return ax
        
        fig.savefig('tmp/tmp1.png', dpi=100)
        plt.close(fig)

    def plot_env(self, ax):
        """
        This function plots the simulation environments.
        """
        ax = self.plot_obs(ax)
        im = plt.imread('data/dest-logo.png')
        ax.imshow(im, extent=(8.4,9.6,8.4,9.6), zorder=-1)
        
        ax.set_xlim(0, 10.1)
        ax.set_ylim(0, 10.1)
        ax.set_aspect(1)
        ax.xaxis.set_tick_params(labelsize='large')
        ax.yaxis.set_tick_params(labelsize='large')
        return ax
    
    def plot_obs(self, ax):
        """
        This function plots the obstacles.
        """
        def plot_l1_norm(ax, obs):      # plot l1 norm ball, a diamond
            xc, yc, rc = obs[0], obs[1], obs[2]
            x_scale, y_scale = obs[4], obs[5]
            # compute half width and height, construct vertex array
            width, height = rc*x_scale, rc*y_scale
            xy = [[xc, yc+height], [xc-width, yc], [xc, yc-height], [xc+width, yc]]
            p = mpatches.Polygon(np.array(xy), closed=True, ec='k', fc='0.9', fill=True, ls='--', lw=2)
            ax.add_patch(p)
            return ax
        
        def plot_l2_norm(ax, obs):      # plot l2 norm ball, an ellipse
            xc, yc, rc = obs[0], obs[1], obs[2]
            x_scale, y_scale = obs[4], obs[5]
            # compute width and height
            width, height = 2*rc*x_scale, 2*rc*y_scale
            p = mpatches.Ellipse((xc, yc), width, height, ec='k', fc='0.9', fill=True, ls='--', lw=2)
            ax.add_patch(p)
            return ax
        
        def plot_linf_norm(ax, obs):    # plot linf norm ball, a rectangle
            xc, yc, rc = obs[0], obs[1], obs[2]
            x_scale, y_scale = obs[4], obs[5]
            # compute x0, y0, width, and height
            width, height = 2*rc*x_scale, 2*rc*y_scale
            x0, y0 = xc-width/2, yc-height/2
            p = mpatches.Rectangle((x0,y0), width, height, ec='k', fc='0.9', fill=True, ls='--', lw=2)
            ax.add_patch(p)
            return ax            

        obs_num = len(self.obs)
        for i in range(obs_num):
            obs_i = self.obs[i]
            if obs_i[3] == 1:
                ax = plot_l1_norm(ax, obs_i)
            elif obs_i[3] == 2:
                ax = plot_l2_norm(ax, obs_i)
            else:
                ax = plot_linf_norm(ax, obs_i)
        return ax
    
    def plot_traj(self, x_traj):
        """
        This function plots the trajectory.
        """
        fig, ax = plt.subplots()
        ax = self.plot_env(ax)
        if x_traj.shape[1] <= 2:
            pa = x_traj[:, 0:2]
            ax.plot(pa[:,0], pa[:,1], 'o')
        else:
            pa = x_traj[:, 0:2]
            pb = x_traj[:, 2:4]
            ax.plot(pa[:,0], pa[:,1], 'o')
            ax.plot(pb[:,0], pb[:,1], 's')
        fig.savefig('tmp/tmp3.png', dpi=100)
        plt.close(fig)
    
    def plot_follower_pos(self, xb):
        """
        This function plots the follower's state, including the position and angle.
        """
        pb, phib = xb[:-1], xb[-1] 
        dx, dy = 0.5*np.cos(phib), 0.5*np.sin(phib)
        
        fig, ax = plt.subplots()
        ax = self.plot_env(ax)
        ax.plot(pb[0], pb[1], 'ro')
        ax.arrow(pb[0],pb[1], dx,dy, color='r', head_length=0.2, head_width=0.2)
        fig.savefig('tmp/tmp6.png')
        plt.close(fig)

    def plot_leader_follower_pos(self, x):
        """
        This function plots the leader and follower's state.
        """
        pa, pb, phib = x[0:2], x[2:4], x[-1]
        dx, dy = 0.5*np.cos(phib), 0.5*np.sin(phib)
        fig, ax = plt.subplots()
        ax = self.plot_env(ax)
        ax.plot(pa[0], pa[1], 'b^')
        ax.plot(pb[0], pb[1], 'ro')
        ax.arrow(pb[0],pb[1], dx,dy, color='r', head_length=0.2, head_width=0.2)
        fig.savefig('tmp/tmp6.png')
        plt.close(fig)