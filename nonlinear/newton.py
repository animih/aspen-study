import numpy as np
from scipy.sparse.linalg import spsolve
from scipy import sparse
from time import time

# class of newton-method solver
class newton():

    log_init = False

    def __init__(self, crit_abs=1e-4, crit_rel = 0, kmax = 25):
        self.kmax = kmax # maximum number of iterations
        self.crit_rel = crit_rel # relative error |R(u*)| < crit_rel * |R(u0)|
        self.crit_abs = crit_abs # absolute error |R(u*)| < crit_abs

    # initializer for timelog
    def init_log(self):
        self.log_init = True
        self.k = 0 # number of iterations
        self.res = 0 # overall resiudal build time
        self.jac = 0 # overall jacobian build time
        self.lin = 0 # overall linear system solve time

    # provide solver with function
    # val(X) and jac(X) method should be supported
    def init_func(self, f):
        self.f = f

    # solve nonlinear system
    # given start iteration (X_cur)
    # 'aspen' flag is needed only when Newton is used in subdomains
    # prev_res provides each subdomain with start resiudal
    def solve(self, X_cur, aspen = False, prev_res=None):

        converged = False

        if aspen:
            res_flag = True
            inds = self.f.inds
        else:
            res_flag = False

        X = np.copy(X_cur)

        for k in range(self.kmax):

            if not(res_flag):
                # residual
                t_res = - time()
                R = self.f.val(X)
                t_res += time()
                if(self.log_init):
                    self.res += t_res

            else:
                R = prev_res
                res_flag = False

            # convergence
            #delta = np.linalg.norm(R)
            delta = np.max(np.abs(R))
            #print(delta)
            # jacobian
            t_jac = - time()
            J = self.f.jac(X)
            t_jac += time()
            if k == 0:
                R0 = delta

            # in future it would be better to rewrite
            # the jacobian to be build in sparse format already
            # for now it is manually converted after (lazy realization)
            # therefore this operation is off timelog
            J = sparse.csr_matrix(J)

            is_conv_abs = delta <= self.crit_abs
            is_conv_rel = delta <= self.crit_rel*R0
            converged = is_conv_abs or is_conv_rel

            if converged:
                break

            if(self.log_init):
                self.jac += t_jac

            # lin solve
            t_lin = - time()
            if aspen:
                X[inds] += spsolve(J, -R).reshape(-1, 1) #np.linalg.solve(J, -R)
            else:
                X += spsolve(J, -R).reshape(-1, 1)#np.linalg.solve(J, -R)
            t_lin += time()
            
            if(self.log_init):
                self.lin += t_lin

        if(self.log_init):
            self.k += k

        if aspen:
            return X[inds], converged

        return X, converged