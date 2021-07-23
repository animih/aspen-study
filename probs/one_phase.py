import numpy as np


# 1D one-phase flow problem
# 4-point stencil is used to solve
class one_phase():

    def __init__(self, D, param):
        self.D = D
        self.Nx = param['Nx']
        self.dt = 1/param['Nt']

        self.func = self.spec_func(self)

    def setBcSr(self, bc, q):
        self.bc = bc
        self.q = q

    def update(self, X_cur, dt):
        self.X_cur = X_cur
        self.dt = dt

    # help function that calculate term that repsonse 
    # to flow ratio between (n-1, n)
    # ... | n-1 <-> n  | n+1 | ...
    # returns k(x_{n-1/2})* mod(max(u_{n}, u_{n-1})) * (u_{n}-u_{n-1})/dx
    def FluxRes(self, X, i):
        val = 0
        delta = np.zeros((2, 1))
        if i == 0:
            delta[0] = X[i]
            delta[1] = self.bc[0]
        elif i == self.Nx:
            delta[0] = self.bc[1]
            delta[1] = X[i - 1]
        else:
            delta = np.array([X[i], X[i - 1]])
        x = max(delta[0], delta[1])
        mult = self.D.model(x)
        val = - self.D.val[i]*self.Nx**2 * mult[0] * (delta[0] - delta[1])
        return val

    # help func that calculate jacobian of term that
    # response to flow ratio between (n-1, n)
    # ... | n-1 <-> n  | n+1 | ...
    # returns 2-size array
    # first element: d(Flow_{n-1, n})/du_n
    # second element: d(Flow_{n-1, n})/du_{n-1}
    def FluxJac(self, X, i):
        val = np.zeros((1, 2))
        delta = np.zeros((2, 1))
        if i == 0:
            delta[0] = self.bc[0]
            delta[1] = X[i]
        elif i == self.Nx:
            delta[0] = X[i-1]
            delta[1] = self.bc[1]
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
            elif i == self.Nx:
                if imax == 1:
                    x_dx = 0
                if j == 1:
                    d_dx = 0

            val[0,j] = -self.D.val[i] * self.Nx**2*(x_dx *   (delta[1]-delta[0])+mult[0]*d_dx)
        return val

    # special func for global stage
    def GFluxJac(self, X_lc, X_gb_prev, i):
        val = np.zeros((1, 2))
        delta = np.zeros((2, 1))
        for j in range(2):
            if j == 0:
                delta[0] = X_lc[i-1]
                delta[1] = X_gb_prev[i]
            if j == 1:
                delta[0] = X_gb_prev[i-1]
                delta[1] = X_lc[i]
            imax = 0
            if delta[1] >= delta[0]:
                imax = 1
            x = delta[imax]
            mult = self.D.model(x)
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
            elif i == self.Nx:
                if imax == 1:
                    x_dx = 0
                if j == 1:
                    d_dx = 0

            val[0,j] = -self.D.val[i] * self.Nx**2*(x_dx *   (delta[1]-delta[0])+mult[0]*d_dx)
        return val

    # returns second i-{ii,jj} -derivate of term responds
    # to flow between n-1, n
    # first el:
    # d^2(Flow_n)/(dU_{n-1})^2
    # second el:
    # d^2(Flow_n)/(dU_n)^2
    # needed for metrics
    def FluxSecJac(self, X, i):
        val = np.zeros((1, 2))
        delta = np.zeros((2, 1))
        if i == 0:
            delta[0] = self.bc[0]
            delta[1] = X[i]
        elif i == self.Nx:
            delta[0] = X[i-1]
            delta[1] = self.bc[1]
        else:
            delta[0] = X[i-1]
            delta[1] = X[i]
        imax = 0
        if delta[1] >= delta[0]:
            imax = 1
        x = delta[imax]
        mult = self.D.model(x)

        for j in range(2):
            x_ddx = 0
            xx_dx = 0
            if imax == j:
                x_ddx = mult[1]
                xx_dx = mult[2]

            d_dx = -1
            if j == 1:
                d_dx = 1

            if i == 0:
                if imax == 0:
                    x_ddx = 0
                    xx_dx = 0
                if j == 0:
                    d_dx = 0
            elif i == self.Nx:
                if imax == 1:
                    x_ddx = 0
                    xx_dx = 0
                if j == 1:
                    d_dx = 0

            val[0,j] = -self.D.val[i] * self.Nx**2*(xx_dx *   (delta[1]-delta[0]) + x_ddx*d_dx+x_ddx*d_dx)
        return val

    class spec_func():

        def __init__(self, outer_instance):

            self.outer = outer_instance

            self.dt = outer_instance.dt
            self.N = outer_instance.Nx

            self.Jf = np.zeros((self.N+1, 2))
            self.Jx = np.zeros((self.N, self.N))
            self.Jx_s = np.zeros((self.N, self.N))
            self.Jt = np.eye(self.N)

            self.Rt = np.zeros((self.N, 1))
            self.Rx = np.zeros((self.N, 1))

            self.res_f = lambda X, i: outer_instance.FluxRes(X, i+1) - outer_instance.FluxRes(X, i)
            self.jac_f = outer_instance.FluxJac
            self.gjac_f = outer_instance.GFluxJac

        def val(self, X, inds = []):
            if np.size(inds, 0) == 0:
                st = 0
                end = self.N
            else:
                st = inds[0]
                end = inds[-1]+1

            np.subtract(X[st:end], self.outer.X_cur[st:end], out = self.Rt[st:end])

            for i in range(st, end):
                self.Rx[i, :] = self.res_f(X, i)

            return self.Rt[st:end]/self.dt + self.Rx[st:end] + self.outer.q[st:end]*self.outer.Nx

        def jac(self, X, inds = []):
            if np.size(inds, 0) == 0:
                st = 0
                end = self.N
            else:
                st = inds[0]
                end = inds[-1]+1

            for i in range(st, end+1):
                self.Jf[i, :] = self.jac_f(X, i)

            for i in range(st, end):
                if i != st:
                    J1 = self.Jf[i, 0]
                    self.Jx_s[i, i-1] = - J1
                J1 = self.Jf[i, 1]
                J2 = self.Jf[i+1, 0]
                self.Jx_s[i, i] = J2-J1
                if i + 1 != end:
                    J2 = self.Jf[i+1, 1]
                    self.Jx_s[i, i+1] = J2

            self.Jx[st:end, st:end] = self.Jx_s[st:end, st:end]

            return self.Jx_s[st:end, st:end] + self.Jt[st:end, st:end]/self.dt

        def reset_jac(self):
            self.Jx_s = np.zeros((self.N, self.N))
            

        def jac_gb(self, X_gb_prev, X, domains):

            borders = [domain[0] for domain in domains[1:]]
            
            for bd in borders:
                tmp = self.gjac_f(X, X_gb_prev, bd)
                self.Jx[bd, bd-1] = -tmp[0][0]
                self.Jx[bd-1, bd] = tmp[0][1]
            
            return self.Jx+self.Jt/self.dt, self.Jx_s+self.Jt/self.dt
    # used for metrics
    def precompute_Jf(self, X, N, step=1):
        Jf = np.zeros((N//step+1, 2))
        for i in range(0, N+1, step):
            Jf[i//step, :] = self.FluxJac(X, i)
        return Jf

# some of proposed metrics
# for one_phase diffusion only

# calulates cross derivative
def m1(Jf, i, j):
    if i - j == 1:
        return np.abs(Jf[i, 0]) + np.abs(Jf[i, 1])
    else:
        return 0

# calculates cross correlation
def m2(X, i, j, x_step = 1,t_step = 1):
    fr = X[i*x_step, ::t_step].T
    sc = X[j*x_step, ::t_step].T
    tmp = np.stack((fr, sc), axis = 0)
    cov = np.cov(tmp)
    val = np.abs(cov[0][1]/np.sqrt(cov[0][0]*cov[1][1]))
    #val = cov[0][1]
    return val

# calculates cross second derivative, or smth...
# don't remeber quite much how derived it...
def m3(solver, X, Jf, i, j, Nt, step = 1):
    if i - j == 1:
        dt = 1/Nt
        Sec_i = solver.FluxSecJac(X, i*step)
        Sec_j = solver.FluxSecJac(X, j*step)
        pp_i = np.abs(Sec_i[0][1])
        p_i = (1/dt+Jf[i+1, 0] - Jf[i, 1])**2
        pp_j = np.abs(Sec_j[0][1])
        p_j = (1/dt+Jf[j+1, 0] - Jf[j, 1])**2

        t1 = Jf[i, 1]**2
        t2 = Jf[i, 0]**2
        t3 = np.abs(Sec_i[0][0])
        return (t3*(1/t1 + 1/t2))+np.abs(pp_i)/(p_i)+np.abs(pp_j)/(p_j)
    else:
        return 0