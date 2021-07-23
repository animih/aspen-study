import newton
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy import sparse
from time import time

import matplotlib.pyplot as plt


# ASPEN solver
# for any nonlinear system
class aspen():
    log_init = False

    def __init__(self, Nd, domains, crit_abs=1e-4, crit_loc = None, crit_rel = 0, max_gb = 15, max_lc = 25):

        self.Nd = Nd
        self.partion = np.copy(domains)
        self.crit_rel = 0
        self.crit_abs = crit_abs
        self.max_gb = max_gb
        self.max_lc = max_lc

        if crit_loc == None:
            crit_loc = crit_abs

        self.newton_solvers = []
        for i in range(Nd):
            self.newton_solvers.append(newton.newton(crit_loc, crit_rel, max_lc))

    def init_log(self):
        self.log_init = True
        self.gb_iters = 0
        self.lc_iters = np.zeros(self.Nd, dtype='int')
        self.gb_res = 0
        self.gb_jac = 0
        self.gb_lin = 0
        self.lc_res = np.zeros(self.Nd)
        self.lc_jac = np.zeros(self.Nd)
        self.lc_lin = np.zeros(self.Nd)

    class local_func():
        def __init__(self, f, inds):
            self.f = f
            self.inds = inds
        def val(self, X):
            return self.f.val(X, self.inds)
        def jac(self, X):
            return self.f.jac(X, self.inds)

    def init_func(self, f):
        self.f = f
        self.X_l = np.zeros((f.N, 1))

        for i in range(self.Nd):
            self.newton_solvers[i].init_func(self.local_func(self.f, self.partion[i]))

    def set_partion(self, partion):
        self.partion = partion

        for i in range(self.Nd):
            self.newton_solvers[i].init_func(self.local_func(self.f, partion[i]))

    def solve(self, X_cur):

        X = np.copy(X_cur)

        converged = False
        for j in range(self.max_gb):
            # residual
            t_res_gb = - time()
            R = self.f.val(X)
            t_res_gb += time()
            if(self.log_init):
                self.gb_res += t_res_gb
            # convergence
            #delta = np.linalg.norm(R)
            delta = np.max(np.abs(R))

            if j == 0:
                R0 = delta
            is_conv_abs = delta <= self.crit_abs
            is_conv_rel = delta <= self.crit_rel*R0
            converged = is_conv_abs or is_conv_rel
            if converged:
                break

            for i in range(self.Nd):
                domain = self.partion[i]

                if(self.log_init):
                    self.newton_solvers[i].init_log()

                self.X_l[domain], mes = self.newton_solvers[i].solve(X, aspen = True, prev_res = R[domain])

                if not(mes):
                    return X, mes

                if(self.log_init):
                    self.lc_iters[i] += self.newton_solvers[i].k
                    self.lc_res[i] += self.newton_solvers[i].res
                    self.lc_jac[i] += self.newton_solvers[i].jac
                    self.lc_lin[i] += self.newton_solvers[i].lin

            # jacobian
            t_jac_gb = - time()
            J, D = self.f.jac_gb(X, self.X_l, self.partion)
            t_jac_gb += time()

            if(self.log_init):
                self.gb_jac += t_jac_gb

            F = X - self.X_l

            J = sparse.csr_matrix(J)
            t_lin_gb = - time()
            X += spsolve(-J, D@F).reshape(-1, 1)
            t_lin_gb += time()

            if(self.log_init):
                self.gb_lin += t_lin_gb

        if(self.log_init) :
            self.gb_iters = j

        return X, converged
