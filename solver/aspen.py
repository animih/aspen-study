import numpy as np
from solver import solver
import matplotlib.pyplot as plt
import time

class aspen_solver(solver):
    def __init__(self, param, diffusion, partion_model):
        super(aspen_solver, self).__init__(param, diffusion)
        self.partion = partion_model
        self.timelog = self.TimeLog(Nd = self.param.Nd, Nt=self.param.Nt)
    class Parameters():
        def __init__(self, param):
            self.Nx = param['Nx']
            self.Nt = param['Nt']
            self.Nd = param['Nd']# number of domains
    class TimeLog():
        def __init__(self, Nd = 0, Nt=0):
            self.domain_iters = np.zeros((Nt+1, Nd))
            self.aspen_iters = np.zeros(Nt+1)

            self.gb_resbld = 0

            self.lc_resbld = np.zeros(Nd)
            self.lc_jacbld = np.zeros(Nd)
            self.lc_linsol = np.zeros(Nd)

            self.gb_jacbld = 0
            self.gb_linsol = 0

    def solve(self, debug = False):
        return_code = 0
        return_message = 'OK'

        # time settings
        t = 0.0
        dt = 1./self.param.Nt
        dt_min = dt
        dt_max = dt

        # initial condition
        X = np.concatenate((self.x0, self.x0), axis=1)

        # local stage solution is stored here
        X_local = np.zeros(self.param.Nx)

        # tempery storgae for domain
        X_local_tmp = np.zeros(self.param.Nx)

        #solution process itself
        R0 = 1
        nstep = 0
        while t < 1.0:
            dt = min(dt, 1-t)
            X[:, 1] = X[:, 0]

            self.t[nstep] = t
            self.X[:, nstep] = X[:, 0]

            # max number of aspen iterations
            max_iter = 25
            crit_rel = 1e-1
            crit_abs = 1e-3

            # partion strategy
            domain_borders = self.partion(self.param.Nx, self.param.Nd)

            # aspen iterations
            for j in range(max_iter):
                # Residual
                t_res = -time.time()
                Rt, W = self.buildResidual(X, dt)
                t_res += time.time()
                self.timelog.gb_resbld += t_res
                Rx = W[1:self.param.Nx+1] - W[0:self.param.Nx]
                Rs = self.q
                R = Rt + Rx + Rs

                # convergence
                delta = np.linalg.norm(R)
                if j == 0:
                    R0 = delta

                is_conv_abs = (delta * dt) <= crit_abs
                is_conv_rel = delta <= (crit_rel*R0)
                is_converged = is_conv_abs or is_conv_rel

                if debug:
                    print('R_norm =', delta)

                if is_converged:
                    self.timelog.aspen_iters[nstep] = j
                    break
                elif j == max_iter:
                    self.timelog.aspen_iters[nstep] = j
                    return_message = 'Aspen not converged'
                    break

                # Newton iteration parameters (local)
                kmax_lc = 18
                crit_rel_lc = 1e-1
                crit_abs_lc = 1e-3

                # local stage solution is stored here
                X_local = np.copy(X[:, 1])

                # tempery storgae for domain
                X_local_tmp = np.copy(X[:, 1])

                # for every domain do (local stage) :
                for i in range(self.param.Nd):
                    # boundaries of the partions
                    start = domain_borders[i]
                    end = domain_borders[i+1]
                    N = end-start

                    for k in range(kmax_lc):
                        # Domain Residual
                        t_res = -time.time()
                        Rt, W = self.buildDomainResidual(X[:, 0], X_local_tmp, dt, start, end)
                        t_res += time.time()
                        self.timelog.lc_resbld[i] += t_res
                        Rx = W[1:N+1] - W[0:N]
                        Rs = self.q[start:end]
                        R = Rt + Rx + Rs

                        # convergence
                        delta = np.linalg.norm(R)
                        if k == 0:
                            R0 = delta

                        is_conv_abs = (delta * dt) <= crit_abs_lc
                        is_conv_rel = delta <= (crit_rel_lc*R0)
                        is_converged = is_conv_abs or is_conv_rel

                        # if local solution is found
                        if is_converged:
                            # save to local and roll back
                            X_local[start:end] = X_local_tmp[start:end]
                            X_local_tmp[start:end] = X[start:end, 1]
                            # save to log
                            self.timelog.domain_iters[nstep, i] = k
                            if(debug):
                                out = 'iter: {}, domain: {}, k = {}'.format(j, i, k)
                                print(out)
                            break
                        elif (k + 1 == kmax_lc):
                            # save to local and roll back
                            X_local[start:end] = X_local_tmp[start:end]
                            X_local_tmp[start:end] = X[start:end, 1]
                            # save to log
                            self.timelog.domain_iters[nstep, i] = k
                            return_message = 'Local not converged'
                            break

                        # Jacobian
                        t_jac = -time.time()
                        Jt, Jf = self.buildDomainFlow(X[:, 0], X_local_tmp, dt, start, end)
                        t_jac += time.time()
                        self.timelog.lc_jacbld[i] += t_jac
                        Jx = self.buildJacobian(Jf,  N)
                        J = Jt + Jx 

                        # linear system solve
                        t_lin = -time.time()
                        dX = np.linalg.solve(J,   -R)
                        t_lin += time.time()
                        self.timelog.lc_linsol[i] += t_lin

                        is_weak = np.sum(np.isnan(dX)) + np.sum(np.isinf(dX))
                        if is_weak > 0:
                            dX[:] = 0

                        # update
                        X_local_tmp[start:end] += np.reshape(dX,   N)        
                # global stage

                t_jac = -time.time()
                Jx, Dx = self.buildGlobalJacobian(X[:, 1], X_local, domain_borders)
                t_jac += time.time()
                self.timelog.gb_jacbld += t_jac
                Jt = np.eye(self.param.Nx)/dt
                J, D = Jx+Jt, Dx+Jt

                F =  X[:, 1] - X_local
                t_lin = -time.time()
                dx = np.linalg.solve(-J, D@F)
                t_lin += time.time()
                self.timelog.gb_linsol += t_lin
                # update
                X[:, 1] += dx

            X[:, 0] = X[:, 1]
            t += dt
            nstep += 1
        return self.X, return_code, return_message

    # from start to end, non-including last
    def buildDomainFlow(self, X_prev, X_local, dt, start, end):
        N = end-start
        Jt = np.eye(N)/dt
        Jf = np.zeros((N+1, 2))
        for i in range(N+1):
            Jf[i, :] = self.calcFluxJac(X_local, start+i)

        return Jt, Jf

    # from start to end, non-including last
    def buildDomainResidual(self, X_prev, X_local, dt, start, end):
        Rt = (X_local[start:end] - X_prev[start:end]) / dt
        N = end-start
        Rt = np.reshape(Rt,   (N,   1))
        W = np.zeros((N + 1, 1))
        for i in range(N+1):
            W[i] = self.calcFluxRes(X_local, start+i)
        return Rt, W

    def buildGlobalJacobian(self, X, X_local, domain_borders):
        Jx, Dx = np.zeros((self.param.Nx, self.param.Nx)), np.zeros((self.param.Nx, self.param.Nx))

        k = 0
        left_bd = domain_borders[k]
        right_bd = domain_borders[k+1]

        Jf_new = np.zeros((self.param.Nx + 1, 2))
        for i in range(self.param.Nx+1):
            Jf_new[i, :] = self.calcFluxJac(X_local, i)

        for i in range(self.param.Nx):
            # change of domain
            if i == right_bd:
                k += 1
                left_bd = domain_borders[k]
                right_bd = domain_borders[k+1]

            if i != 0:
                if i != left_bd:
                    J1 = Jf_new[i, 0]
                    Jx[i, i-1] = -J1
                    Dx[i, i-1] = Jx[i, i-1]
                else:
                    J1 = self.calcFluxJac(X, i)[0, 0]
                    Jx[i, i-1] = -J1

            J1 = Jf_new[i, 1]
            J2 = Jf_new[i + 1, 0]
            Jx[i, i] = J2 - J1
            Dx[i, i] = Jx[i, i]

            if i + 1 != self.param.Nx:
                if i+1 != right_bd:
                    J2 = Jf_new[i+1, 1]
                    Jx[i, i+1] = J2
                    Dx[i, i+1] = Jx[i, i+1]
                else:
                    J2 = self.calcFluxJac(X, i+1)[0, 1]
                    Jx[i, i+1] = J2

        return Jx, Dx

    def calcFluxJac(self, X, i):
        val = np.zeros((1, 2))
        if self.bc.first:
            delta = np.zeros((2, 1))
            if i == 0:
                delta[0] = self.bc.val[1]
                delta[1] = X[i]
            elif i == self.param.Nx:
                delta[0] = X[i-1]
                delta[1] = self.bc.val[1]
            else:
                delta[0] = X[i-1]
                delta[1] = X[i]
            imax = 0
            if delta[1] >= delta[0]:
                imax = 1
            x = delta[imax]
            mult = self.D.model(x)
            for j in range(2):
                x_dx = 0
                if imax == j:
                    x_dx = mult[1]

                d_dx = -1
                if j == 1:
                    d_dx = 1

                if i == 0:
                    if imax == 0:
                        x_dx = 0
                    if j == 0:
                        d_dx = 0
                elif i == self.param.Nx:
                    if imax == 1:
                        x_dx = 0
                    if j == 1:
                        d_dx = 0

                val[0,j] = -self.D.val[i] * self.param.Nx**2*(x_dx *   (delta[1]-delta[0])+mult[0]*d_dx)
        else:
            pass
        return val
    def calcFluxRes(self, X, i):
        val = 0
        if self.bc.first:
            delta = np.zeros((2, 1))
            if i == 0:
                delta[0] = X[i]
                delta[1] = self.bc.val[0]
            elif i == self.param.Nx:
                delta[0] = self.bc.val[1]
                delta[1] = X[i - 1]
            else:
                delta = np.array([X[i], X[i - 1]])
            x = max(delta[0], delta[1])
            mult = self.D.model(x)
            val = - self.D.val[i]*self.param.Nx**2 * mult[0] * (delta[0] - delta[1])
        else:
            pass

        return val
    def buildResidual(self, X, dt):
        Rt = (X[:, 1] - X[:, 0]) / dt
        Rt = np.reshape(Rt,   (self.param.Nx,   1))
        mask = range(self.param.Nx+1)
        W = np.zeros((self.param.Nx + 1, 1))
        for i in mask:
            W[i] = self.calcFluxRes(X[:, 1], i)
        return Rt, W
    def buildJacobian(self, Jf, Nx):
        Jx = np.zeros((Nx, Nx))
        for i in range(Nx):
            if i != 0:
                J1 = Jf[i, 0]
                Jx[i, i-1] = -J1
            J1 = Jf[i, 1]
            J2 = Jf[i + 1, 0]
            Jx[i, i] = J2 - J1
            if i + 1 != Nx:
                J2 = Jf[i+1, 1]
                Jx[i, i+1] = J2
        return Jx