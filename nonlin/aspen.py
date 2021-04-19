import newton
import numpy as np
from time import time

class aspen():
    # domain borders should be defined before ASPEN

    log_init = False

    def __init__(self, Nd, domain_borders, crit_rel, crit_abs = 0, max_gb = 3, max_lc = 12):

        self.Nd = Nd
        self.partion = domain_borders
        self.crit_rel = crit_rel
        self.crit_abs = crit_abs
        self.max_gb = max_gb
        self.max_lc = max_lc
        self.newton = newton.newton(crit_rel, max_lc, crit_abs)

    def init_log(self):
        self.log_init = True
        self.gb_iters = 0
        self.lc_iters = np.zeros(self.Nd)
        self.gb_res = 0
        self.gb_jac = 0
        self.gb_lin = 0
        self.lc_res = np.zeros(self.Nd)
        self.lc_jac = np.zeros(self.Nd)
        self.lc_lin = np.zeros(self.Nd)

    class local_func():
        def __init__(self, f, X_prev, start, end):
            self.X_prev = X_prev
            self.f = f
            self.start = start
            self.end = end
        def val(self, X):
            return self.f.val_loc(X, self.X_prev, self.start, self.end)
        def jac(self, X):
            return self.f.jac_loc(X, self.X_prev, self.start, self.end)

    def solve(self, f, X0):

        converged = False
        X = np.copy(X0)

        domain_borders = self.partion
        f_l = self.local_func(f, X, 0, 0)

        for j in range(self.max_gb):
            # residual
            t_res_gb = - time()
            R = f.val(X)
            t_res_gb += time()

            if(self.log_init):
                self.gb_res += t_res_gb

            # convergence
            delta = np.linalg.norm(R)
            if j == 0:
                R0 = delta

            is_conv_abs = delta <= self.crit_abs
            is_conv_rel = delta <= self.crit_rel*R0
            converged = is_conv_abs or is_conv_rel

            if converged:
                break

            X_l = np.copy(X)

            f_l.X_prev = X

            for i in range(self.Nd):

                if(self.log_init):
                    self.newton.init_log()

                start = domain_borders[i]
                end = domain_borders[i+1]
                N = end-start

                f_l.start = start
                f_l.end = end

                X_l[start:end], mes = self.newton.solve(f_l, X[start:end])

                if not(mes):
                    return X, mes

                if(self.log_init):
                    self.lc_iters[i] += self.newton.k
                    self.lc_res[i] += self.newton.res
                    self.lc_jac[i] += self.newton.jac
                    self.lc_lin[i] += self.newton.lin

            # jacobian
            t_jac_gb = - time()
            J, D = f.jac_gb(X_l, X, domain_borders)
            t_jac_gb += time()

            if(self.log_init):
                self.gb_jac += t_jac_gb

            F = X - X_l

            # lin solve
            t_lin_gb = - time()
            X += np.linalg.solve(-J, D@F)
            t_lin_gb += time()

            if(self.log_init):
                self.gb_lin += t_lin_gb

        if(self.log_init) :
            self.gb_iters = j

        return X, converged
