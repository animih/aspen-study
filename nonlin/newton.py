import numpy as np
from time import time

class newton():

    log_init = False

    def __init__(self, crit_abs=1e-4, crit_rel = 0, kmax = 25):
        self.kmax = kmax
        self.crit_rel = crit_rel
        self.crit_abs = crit_abs

    def init_log(self):
        self.log_init = True
        self.k = 0
        self.res = 0
        self.jac = 0
        self.lin = 0

    def init_func(self, f):
        self.f = f

    def solve(self, X, aspen = False):

        converged = False

        if aspen:
            bg = self.f.start
            en = self.f.end

        for k in range(self.kmax):
            # residual
            t_res = - time()
            R = self.f.val(X)
            t_res += time()

            if(self.log_init):
                self.res += t_res

            # convergence
            delta = np.linalg.norm(R)
            #print(delta)
            if k == 0:
                R0 = delta

            is_conv_abs = delta <= self.crit_abs
            is_conv_rel = delta <= self.crit_rel*R0
            converged = is_conv_abs or is_conv_rel

            if converged:
                break

            # jacobian
            t_jac = - time()
            J = self.f.jac(X)
            t_jac += time()

            if(self.log_init):
                self.jac += t_jac

            # lin solve
            t_lin = - time()
            if aspen:
                X[bg:en] += np.linalg.solve(J, -R)
            else:
                X += np.linalg.solve(J, -R)
            t_lin += time()
            
            if(self.log_init):
                self.lin += t_lin

        if(self.log_init):
            self.k += k

        if aspen:
            return X[bg:en], converged

        return X, converged