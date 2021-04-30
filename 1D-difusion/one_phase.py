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
        self.t = np.arange(self.param.Nt+1)/self.param.Nt
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
    def __init__(self, param, diffusion, nonlin_solver, bd_ch = None):
        super(one_phase, self).__init__(param, diffusion)
        self.solver = nonlin_solver
        self.func = self.spec_func(self)
        if bd_ch == None:
            self.dyn_bd = False
        else:
            self.dyn_bd = True
            self.bd_ch = bd_ch
    def init_log(self):
        self.log_init = True
        if type(self.solver).__name__ == 'aspen':
            self.timelog = self.aspen_log(self.solver.Nd, self.param.Nt)
            if(self.dyn_bd):
                self.timelog.borders = np.zeros((self.solver.Nd+1, 5))
                self.timelog.borders[:, 0] = self.solver.partion
        elif type(self.solver).__name__ == 'newton':
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
        return_message = 'OK'
        self.solver.init_func(self.func)

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

            X, mes = self.solver.solve(X)

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
                if self.dyn_bd and (nstep % (self.param.Nt//10) == 0) and nstep != self.param.Nt:
                    self.solver.partion = self.bd_ch(self.X[:, nstep-10:nstep], self.solver.Nd)
                    self.timelog.borders[:, nstep*5//self.param.Nt] = self.solver.partion

        if type(self.solver).__name__ == 'aspen' and self.log_init:
            self.timelog.domain_iters /= self.param.Nt

        self.solver.crit_abs = crit_abs

        return self.X, return_message

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

    class spec_func():

        def __init__(self, outer_instance):

            self.outer = outer_instance

            self.dt = 1/outer_instance.param.Nt
            self.N = outer_instance.param.Nx

            self.Jf = np.zeros((self.N+1, 2))
            self.Jx = np.zeros((self.N, self.N))
            self.Jt = np.eye(self.N)

            self.gb_Jx = np.zeros((self.N, self.N))

            self.Rt = np.zeros((self.N, 1))
            self.Rx = np.zeros((self.N, 1))

            self.res_f = lambda X, i: outer_instance.FluxRes(X, i+1) - outer_instance.FluxRes(X, i)
            self.jac_f = outer_instance.FluxJac

        def val(self, X, st = 0, end = None):
            if end == None:
                end = self.N
            np.subtract(X[st:end], self.outer.X_cur[st:end], out = self.Rt[st:end, :])

            for i in range(st, end):
                self.Rx[i, :] = self.res_f(X, i)

            return self.Rt[st:end]/self.dt + self.Rx[st:end] + self.outer.q[st:end]

        def jac(self, X, st = 0, end = None):
            if end == None:
                end = self.N
            for i in range(st, end+1):
                self.Jf[i, :] = self.jac_f(X, i)

            for i in range(st, end):
                if i != st:
                    J1 = self.Jf[i, 0]
                    self.Jx[i, i-1] = - J1
                J1 = self.Jf[i, 1]
                J2 = self.Jf[i+1, 0]
                self.Jx[i, i] = J2-J1
                if i + 1 != end:
                    J2 = self.Jf[i+1, 1]
                    self.Jx[i, i+1] = J2

            return self.Jx[st:end, st:end] + self.Jt[st:end, st:end]/self.dt
            

        def jac_gb(self, X_gb_prev, domain_borders):

            self.gb_Jx = np.copy(self.Jx)

            for bd in domain_borders[1:-1]:
                tmp = self.jac_f(X_gb_prev, bd)
                self.gb_Jx[bd, bd-1] = -tmp[0][0]
                self.gb_Jx[bd-1, bd] = tmp[0][1]

            return self.gb_Jx+self.Jt/self.dt, self.Jx+self.Jt/self.dt # J & D