import numpy as np
from time import time

class fxbd_1D():
  # Grid parameters
    class Parameters():
        def __init__(self, param):
            self.Nx = param['Nx']
            self.Nt = param['Nt']
    # 1-st boundary condition
    class Boundary():
        def __init__(self, left, right):
            self.val = np.array((left, right))
    log_init = False
    def __init__(self, param, problem):
        self.param = self.Parameters(param)
        self.X = np.zeros((self.param.Nx, self.param.Nt+1))
        self.t = np.arange(self.param.Nt+1)/self.param.Nt
        self.bc = self.Boundary(0, 0)
        self.x0 = np.zeros((self.param.Nx, 1))
        self.q = np.zeros((self.param.Nx, 1))
        # the equation to solve
        self.prob = problem

    def setSolver(self, nl_solver, bd_ch = None):
        self.solver = nl_solver

        if bd_ch == None:
            self.dyn_bd = False
        else:
            self.dyn_bd = True
            self.bd_ch = bd_ch
            self.freq_ch = 5

    def setBoundary(self, left, right):
        self.bc = self.Boundary(left, right)
    def setInitial(self, x0):
      self.x0 = x0

    # sources will add constant term in residual
    def setSources(self, list_pos, list_val):
      self.q = np.zeros((self.param.Nx, 1))
      for Pos, Val in zip(list_pos, list_val):
          index = round(self.param.Nx*Pos)
          self.q[index-1] = Val

    # timelogs for research purposes
    class aspen_log():
        def __init__(self, Nd = 0, Nt = 0):
            self.domain_iters = np.zeros((Nd, Nt), dtype='int')
            self.aspen_iters = np.zeros(Nt, dtype='int')
            self.gb_res = 0
            self.lc_res = np.zeros(Nd)
            self.lc_jac = np.zeros(Nd)
            self.lc_lin = np.zeros(Nd)
            self.gb_jac = 0
            self.gb_lin = 0
        def update(self, solver, nstep):
            self.aspen_iters[nstep] += solver.gb_iters
            self.domain_iters[:, nstep] += solver.lc_iters

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

    def init_log(self):
        self.log_init = True
        if type(self.solver).__name__ == 'aspen':
            self.timelog = self.aspen_log(self.solver.Nd, self.param.Nt)
            if(self.dyn_bd):
                self.timelog.borders = np.zeros((self.solver.Nd+1, 5))
                self.timelog.borders[:, 0] = self.solver.partion
                self.timelog.bd_time = 0
        elif type(self.solver).__name__ == 'newton':
            self.timelog = self.newton_log(self.param.Nt)
        else:
            pass

    # solve function
    def solve(self, tmax=1.0):
        return_message = 'OK'
        self.solver.init_func(self.prob.func)
        # time setting
        t = 0.0
        dt = tmax/self.param.Nt
        self.t = tmax*np.arange(self.param.Nt+1)/self.param.Nt
        dt_min = dt*1e-3
        # initial condition
        self.X[:, 0] = self.x0.flatten()
        # solution process itself
        R0 = 1
        nstep = 0
        crit_abs = np.copy(self.solver.crit_abs)
        X = np.copy(self.x0)
        self.X_cur = np.copy(X)

        while t < tmax:
            dt = min(dt, self.t[nstep+1]-t)
            self.prob.update(dt, self.X_cur, self.bc, self.q)
            self.solver.crit_abs = crit_abs/dt

            if self.log_init:
                self.solver.init_log()


            X, mes = self.solver.solve(X)

            if not(mes):
                #dt /= 4
                # force-locked time step
                # to study perfomance dependece
                # on currant number
                return_message = 'Not converged'
                break
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
                if self.dyn_bd and (nstep % (self.param.Nt//freq_ch) == 0) and nstep != self.param.Nt:
                    self.func.reset_jac(self.solver.partion)
                    bd_t = -time()
                    self.solver.partion = self.bd_ch(self, self.X, nstep, self.solver.Nd)
                    bd_t += time()
                    self.timelog.bd_time += bd_t
                  
                    if self.log_init :
                        self.timelog.borders[:, nstep*freq_ch//self.param.Nt] = self.solver.partion

        self.solver.crit_abs = crit_abs

        return self.X, return_message
