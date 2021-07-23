import numpy as np
from time import time

import matplotlib.pyplot as plt
# one-dimensional fixed borders problem
# for any equation, specified by prob object
class fxbd_1D():

    timelog_init = False

    def __init__(self, param, prob):
        self.param = param
        self.prob = prob

        # grid properties
        self.Nt = param['Nt']
        self.Nx = param['Nx']

        # border condition
        self.bc = (0, 0)

        # solution storage
        self.X = np.zeros((self.Nx, self.Nt+1))
        # source vector
        self.q = np.zeros((self.Nx, 1))

    # set nonlinear solver
    def setSolver(self, nl_solver, bd_ch = None, freq_ch = 5):
        self.nl_solver = nl_solver

        if bd_ch == None:
            self.dyn_bd = False
        else:
            self.dyn_bd = True
            self.bd_ch = bd_ch
            self.freq_ch = freq_ch

    # set Dirichle Conditions on borders
    def setBoundary(self, left, right):
        self.bc = (left, right)

    # set Initial condition on variable
    def setInitial(self, x0):
        self.X[:, 0] = x0

    # sources will add q_i/dx (aprroximation of delta-func) term in residual
    def setSources(self, list_pos, list_val):
        self.q = np.zeros((self.Nx, 1))
        for Pos, Val in zip(list_pos, list_val):
            index = round(self.Nx*Pos)
            self.q[index-1] = Val

    # timelogs for research purposes
    # currently different class for Newton and ASPEN
    # timelogs are written
    class aspen_log():
        def __init__(self, Nd = 0, Nt = 0):
            # iteration number in each domain on each step
            self.domain_iters = np.zeros((Nd, Nt), dtype='int')
            # number of global iterations of ASPEN
            self.aspen_iters = np.zeros(Nt, dtype='int')

            # total resiudal build time on global stage
            # total jacobian build time on global stage
            # total linear solve time on global stage
            self.gb_res = 0
            self.gb_jac = 0
            self.gb_lin = 0
            # total resiudal build time on global stage in each domain
            # total jacobian build time on global stage in each domain
            # total linear solve time on global stage in each domain
            self.lc_res = np.zeros(Nd)
            self.lc_jac = np.zeros(Nd)
            self.lc_lin = np.zeros(Nd)

        # basically, updates the log given nonlinear solver
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
            # total resiudal build time
            # total jacobian build time
            # total linear solve time
            self.lin = 0
            self.res = 0
            self.jac = 0
            # number of Newton iterations on each step
            self.kn = np.zeros((Nt))
        # basically, updates the log given nonlinear solver
        def update(self, solver, nstep):
            self.lin += solver.lin
            self.res += solver.res
            self.jac += solver.jac
            self.kn[nstep] += solver.k

    # initialize the timelog for research purposes
    def init_log(self):
        self.log_init = True
        if type(self.nl_solver).__name__ == 'aspen':
            self.timelog = self.aspen_log(self.nl_solver.Nd, self.Nt)
            # if domains are dynamic they will be written in log too
            if(self.dyn_bd):
                self.timelog.borders = np.zeros((self.nl_solver.Nd+1, self.freq_ch))

                for i in range(self.nl_solver.Nd):
                    self.timelog.borders[i, 0] = self.nl_solver.partion[i][0]
                self.timelog.borders[self.nl_solver.Nd , :] = self.Nx
                self.timelog.bd_time = 0

        elif type(self.nl_solver).__name__ == 'newton':
            self.timelog = self.newton_log(self.Nt)
        else:
            pass

    # solve method
    # given desired end time calculates the solution
    # for grid properties specified in initializer
    def solve(self, tmax=1.0):
        return_message = 'OK'

        # time setting
        t = 0.0
        dt = tmax/self.Nt
        self.t = dt*(np.arange(self.Nt+1))
        dt_min = dt*1e-3

        R0 = 1
        nstep = 0
        crit_abs = np.copy(self.nl_solver.crit_abs)

        self.X_cur = np.copy(self.X[:, 0].reshape(-1, 1))

        self.prob.setBcSr(self.bc, self.q)
        self.nl_solver.init_func(self.prob.func)

        while t < tmax:
            dt = min(dt, self.t[nstep+1]-t)
            self.nl_solver.crit_abs = crit_abs/dt

            if self.log_init:
                self.nl_solver.init_log()

            self.prob.update(self.X_cur, dt)
            self.X_cur, mes = self.nl_solver.solve(self.X_cur)

            if not(mes):
                # dt /= 4
                # force-locked time step
                # to study perfomance dependece

                # to change uncomment this line
                # dt /= 4

                # comment this line
                dt = dt_min *1e-1

                if dt < dt_min:
                    return_message = 'Not converged'
                    break
                continue

            t += dt

            # update log if initialized
            if self.log_init :
                self.timelog.update(self.nl_solver, nstep)
 
            # if on the desired timstep write solution
            if self.t[nstep+1] == t:
                self.X[:, nstep+1] = self.X_cur.flatten()
                nstep += 1
                dt = 1/self.Nt

            # basicly the block for domain partion change
            if self.dyn_bd and self.t[nstep] == t and ((nstep) % (self.Nt//self.freq_ch) == 0) and nstep < self.Nt:

                # we need to zero all non-block elements in jacobian
                # in current lazy realization all elements are zeroed
                # in order to save continuity to multiple dimensions problems
                self.prob.func.reset_jac()

                bd_t = -time()
                partion = self.bd_ch(self.prob, self.X, nstep, self.nl_solver.Nd)
                self.nl_solver.set_partion(partion)
                bd_t += time()
                self.timelog.bd_time += bd_t
              
                if self.log_init :

                    for i in range(self.nl_solver.Nd):
                        self.timelog.borders[i, (nstep)*self.freq_ch//self.Nt] = self.nl_solver.partion[i][0]    
                        
        # return back to avoid bugs while runing same problem several times
        self.nl_solver.crit_abs = crit_abs

        return self.X, return_message
            



    