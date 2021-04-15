import numpy as np
from time import time

class solver():
    class Parameters():
        def __init__(self, param):
            self.Nx = param['Nx']
            self.Nt = param['Nt']
    class Boundary():
        def __init__(self, left, right, kind=1):
            self.first = (kind == 1)
            self.val = np.array((left, right))
    def __init__(self, param, diffusion):
        self.param = self.Parameters(param)
        self.X = np.zeros((self.param.Nx, self.param.Nt+1))
        self.t = np.arange(self.param.Nt+1)/self.param.Nt # fixed
        self.D = diffusion
        self.bc = self.Boundary(0, 0)
        self.x0 = np.zeros((self.param.Nx, 1))
        self.q = np.zeros((self.param.Nx, 1))
    def setBoundary(self, left, right, kind = 1):
        self.bc = self.Boundary(left, right, kind)
    def setInitial(self, Amp=0, Period=0, Scale=1):
        self.x0 = np.zeros((self.param.Nx, 1))
        for i in range(self.param.Nx):
            self.x0[i] = Amp * np.sin(Period * 2*np.pi * (i+1) / self.param.Nx) + Scale
        return self.x0
    def setSources(self, list_pos, list_val):
        self.q = np.zeros((self.param.Nx, 1))
        for Pos, Val in zip(list_pos, list_val):
            index = round(self.param.Nx*Pos)
            self.q[index-1] = Val
        return self.q

class one_phase(solver):
    log_init = False
    def __init__(self, param, diffusion, nonlin_solver):
        super(one_phase, self).__init__(param, diffusion)

        self.solver = nonlin_solver
        self.func = self.spec_func(self)

    def init_log(self, show = True):
        self.log_init = True
        
        if type(self.solver).__name__ == 'aspen':
            if show :
                print('ASPEN log initialized')
            self.timelog = self.aspen_log(self.solver.Nd, self.param.Nt)
        elif type(self.solver).__name__ == 'newton':
            if show :
                print('newton log initialized')
            self.timelog = self.newton_log(self.param.Nt)
        else:
            pass

    class aspen_log():
        def __init__(self, Nd = 0, Nt = 0):
            self.domain_iters = np.zeros(Nd)
            self.aspen_iters = np.zeros(Nt)

            self.gb_res = 0

            self.lc_res = np.zeros(Nd)
            self.lc_jac = np.zeros(Nd)
            self.lc_lin = np.zeros(Nd)

            self.gb_jac = 0
            self.gb_lin = 0
        def update(self, solver, nstep):
            self.aspen_iters[nstep] += solver.gb_iters
            self.domain_iters += solver.lc_iters

            self.gb_res += solver.gb_res
            self.gb_jac += solver.gb_jac
            self.gb_lin += solver.gb_lin

            self.lc_res += solver.lc_res
            self.lc_jac += solver.lc_jac
            self.lc_lin += solver.lc_lin

    class newton_log():
        def __init__(self, Nt = 0):
            self.lin = 0
            self.res = 0
            self.jac = 0
            self.kn = np.zeros((Nt))
        def update(self, solver, nstep):
            self.lin += solver.lin
            self.res += solver.res
            self.jac += solver.jac
            self.kn[nstep] += solver.k


    def solve(self):
        return_code = 0
        return_message = 'OK'

        # time settings
        t = 0.0
        dt = 1/self.param.Nt
        dt_min = dt*1e-3

        # initial condition
        self.X[:, 0] = self.x0.flatten()

        # solution process itself
        R0 = 1
        nstep = 0

        crit_abs = np.copy(self.solver.crit_abs)

        X = np.copy(self.x0)
        self.X_cur = np.copy(self.x0)

        while t < 1.0:
            dt = min(dt, self.t[nstep+1]-t)
            self.solver.crit_abs = crit_abs/dt

            if self.log_init:
                self.solver.init_log()

            X, mes = self.solver.solve(self.func, X)

            if not(mes):
                dt /= 4
                if dt < dt_min:
                    return_message = 'Not converged'
                    break
                continue

            self.X_cur = np.copy(X)
            t += dt

            if self.log_init :
                self.timelog.update(self.solver, nstep)
            
            if self.t[nstep+1] == t:
                self.X[:, nstep+1] = self.X_cur.flatten()
                nstep += 1
                dt = 1/self.param.Nt

        if type(self.solver).__name__ == 'aspen':
            self.timelog.domain_iters /= self.param.Nt

        return self.X, return_code, return_message

    def FluxRes(self, X, i):
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

    def locFluxRes(self, X, X_prev, i, start, end):
        val = 0
        if self.bc.first:
            delta = np.zeros((2, 1))
            if start+i == 0:
                delta[0] = X[i]
                delta[1] = self.bc.val[0]
            elif start+i == self.param.Nx:
                delta[0] = self.bc.val[1]
                delta[1] = X[i - 1]
            else:
                if i == 0:
                    delta = np.array([X[0], X_prev[start-1]])
                elif  i == end-start:
                    delta = np.array([X_prev[end], X[i-1]])
                else:
                    delta = np.array([X[i], X[i - 1]])
            x = max(delta[0], delta[1])
            mult = self.D.model(x)
            val = - self.D.val[start+i]*self.param.Nx**2 * mult[0] * (delta[0] - delta[1])
        else:
            pass
        return val

    def FluxJac(self, X, i):
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

    def locFluxJac(self, X, X_prev, i, start, end):
        val = np.zeros((1, 2))
        if self.bc.first:
            delta = np.zeros((2, 1))
            if start+i == 0:
                delta[0] = self.bc.val[1]
                delta[1] = X[i]
            elif start+i == self.param.Nx:
                delta[0] = X[i-1]
                delta[1] = self.bc.val[1]
            else:
                if i == 0:
                    delta[0] = X_prev[start-1]
                    delta[1] = X[0]
                elif i == end-start:
                    delta[0] = X[i-1]
                    delta[1] = X_prev[end]
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

                if start+i == 0:
                    if imax == 0:
                        x_dx = 0
                    if j == 0:
                        d_dx = 0
                elif start+i == self.param.Nx:
                    if imax == 1:
                        x_dx = 0
                    if j == 1:
                        d_dx = 0

                val[0,j] = -self.D.val[start+i] * self.param.Nx**2*(x_dx *   (delta[1]-delta[0])+mult[0]*d_dx)
        else:
            pass
        return val

    class spec_func():

        def __init__(self, outer_instance):

            self.outer = outer_instance
            self.dt = 1/self.outer.param.Nt

        def val(self, X):
            N = self.outer.param.Nx
            Rt = (X - self.outer.X_cur) / self.dt
            Rt = np.reshape(Rt,   (N,   1))
            Rx = np.zeros((N, 1))
            for i in range(N):
                Rx[i] = self.outer.FluxRes(X, i+1) - self.outer.FluxRes(X, i)
            Rs = self.outer.q

            return Rt + Rx + Rs

        def jac(self, X):

            N = self.outer.param.Nx
            Jx = np.zeros((N, N))
            Jt = np.eye(N)/self.dt
            Jf = np.zeros((N+1, 2))
            for i in range(N+1):
                Jf[i, :] = self.outer.FluxJac(X, i)

            for i in range(N):
                if i != 0:
                    J1 = Jf[i, 0]
                    Jx[i, i-1] = - J1
                J1 = Jf[i, 1]
                J2 = Jf[i+1, 0]
                Jx[i, i] = J2-J1
                if i + 1 != N:
                    J2 = Jf[i+1, 1]
                    Jx[i, i+1] = J2

            return Jx + Jt


        def val_loc(self, X, X_prev, start, end):
            Rt = (X - self.outer.X_cur[start:end]) / self.dt
            N = end - start
            Rt = np.reshape(Rt,   (N,   1))
            Rx = np.zeros((N, 1))
            Rs = self.outer.q[start:end]

            for i in range(0, N):
                Rx[i] = self.outer.locFluxRes(X, X_prev, i+1, start, end) \
                    - self.outer.locFluxRes(X, X_prev, i, start, end)

            return Rt + Rx + Rs

        def jac_loc(self, X, X_prev, start, end):

            N = end - start
            Jx = np.zeros((N, N))
            Jt = np.eye(N)/self.dt
            Jf = np.zeros((N+1, 2))
            for i in range(N+1):
                Jf[i, :] = self.outer.locFluxJac(X, X_prev, i, start, end)

            for i in range(N):
                if i != 0:
                    J1 = Jf[i, 0]
                    Jx[i, i-1] = - J1
                J1 = Jf[i, 1]
                J2 = Jf[i+1, 0]
                Jx[i, i] = J2-J1
                if i + 1 != N:
                    J2 = Jf[i+1, 1]
                    Jx[i, i+1] = J2

            return Jx + Jt
            

        def jac_gb(self, X, X_l, domain_borders):
            N = self.outer.param.Nx
            Jx, Dx = np.zeros((N, N)), np.zeros((N, N))
            Jt = np.eye(N)/self.dt

            k = 0
            left_bd = domain_borders[k]
            right_bd = domain_borders[k+1]

            Jf_new = np.zeros((N+ 1, 2))
            for i in range(N+1):
                Jf_new[i, :] = self.outer.FluxJac(X_l, i)

            for i in range(N):
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
                        J1 = self.outer.FluxJac(X, i)[0, 0]
                        Jx[i, i-1] = -J1

                J1 = Jf_new[i, 1]
                J2 = Jf_new[i + 1, 0]
                Jx[i, i] = J2 - J1
                Dx[i, i] = Jx[i, i]

                if i + 1 != N:
                    if i+1 != right_bd:
                        J2 = Jf_new[i+1, 1]
                        Jx[i, i+1] = J2
                        Dx[i, i+1] = Jx[i, i+1]
                    else:
                        J2 = self.outer.FluxJac(X, i+1)[0, 1]
                        Jx[i, i+1] = J2

            return Jx+Jt, Dx+Jt