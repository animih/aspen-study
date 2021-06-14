import newton
import numpy as np
from time import time

import matplotlib.pyplot as plt

class aspen():
    log_init = False

    def __init__(self, Nd, domain_borders, crit_abs=1e-4, newton_crit_rel = 0, max_gb = 10, max_lc = 25):

        self.Nd = Nd
        self.partion = np.copy(domain_borders)
        self.crit_rel = 0
        self.crit_abs = crit_abs
        self.max_gb = max_gb
        self.max_lc = max_lc
        
        self.newton = newton.newton(crit_abs, newton_crit_rel, max_lc)

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
        def __init__(self, f, Nx):
            self.f = f
            self.start = 0
            self.end = Nx
        def set_domain(self, start, end):
            self.start = start
            self.end = end
        def val(self, X):
            return self.f.val(X, self.start, self.end)
        def jac(self, X):
            return self.f.jac(X, self.start, self.end)

    # should be called before 'solve' method
    def init_func(self, f):
        self.f = f
        self.f_l = self.local_func(f, f.N)
        self.X_l = np.zeros((f.N, 1))

        #size = np.max(self.partion[1:]-self.partion[:-1])
        self.buf = np.zeros((f.N, 1))

        self.newton.init_func(self.f_l)

    def solve(self, X):

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

                if(self.log_init):
                    self.newton.init_log()
                start = self.partion[i]
                end = self.partion[i+1]
                self.f_l.set_domain(start, end)

                self.buf[:end-start] = X[start:end]
                self.X_l[start:end], mes = self.newton.solve(X, aspen = True)

                X[start:end] = self.buf[:end-start]

                if not(mes):
                    return X, mes

                if(self.log_init):
                    self.lc_iters[i] += self.newton.k
                    self.lc_res[i] += self.newton.res
                    self.lc_jac[i] += self.newton.jac
                    self.lc_lin[i] += self.newton.lin

            # jacobian
            t_jac_gb = - time()
            J, D = self.f.jac_gb(X, self.partion)
            t_jac_gb += time()

            if(self.log_init):
                self.gb_jac += t_jac_gb


            F = X - self.X_l

            t_lin_gb = - time()
            X += np.linalg.solve(-J, D@F)
            t_lin_gb += time()

            if(self.log_init):
                self.gb_lin += t_lin_gb

        if(self.log_init) :
            self.gb_iters = j

        return X, converged
