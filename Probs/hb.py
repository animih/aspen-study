import numpy as np

# implict angle scheme for nonlin transer eq

class transfer():

    def __init__(self, c, param):
        self.c = c
        self.Nx = param['Nx']
        self.dt = 1/param['Nt']

        self.func = self.spec_func(self)

    def update(self, dt, X_cur, bc, q):
        self.dt = dt
        self.X_cur = X_cur
        self.bc = bc
        self.q = q

    # finds -a(x, u)*(u_{l+1}-u_l)/h
    def Res(self, X, i):
        val = 0
        delta = np.zeros((2, 1))
        if i == self.Nx-1:
            delta[0] = self.bc.val[1]
            delta[1] = X[i]
        else:
            delta = np.array([X[i+1], X[i]])
        mult = self.c.model(delta[1])
        val = -self.c.val[i]*self.Nx * mult[0] * (delta[0] - delta[1])
        return val

    # returns two elements
    # first - d (R_i) / du_{i}
    # second - d (R_i) / du_{i+1}
    def Jac(self, X, i):
        val = np.zeros((1, 2))
        delta = np.zeros((2, 1))
        if i == self.Nx-1:
            delta[0] = self.bc.val[1]
            delta[1] = X[i]
        else:
            delta[0] = X[i+1]
            delta[1] = X[i]

        mult = self.c.model(X[i])

        for j in range(2):
            x_dx = 0
            if j == 0:
                x_dx = mult[1]

            d_dx = -1
            if j == 1:
                d_dx = 1

            val[0,j] = -self.c.val[i]*self.Nx * (x_dx * (delta[0]-delta[1])+mult[0]*d_dx)
        return val

    # special func for global stage
    def GJac(self, X_lc, X_gb_prev, i):
        delta = np.zeros((2, 1))
        delta[0] = X_gb_prev[i+1]
        delta[1] = X_lc[i]

        mult = self.c.model(X_lc[i])

        x_dx = 0

        d_dx = 1

        val = -self.c.val[i]*self.Nx * (x_dx * (delta[0]-delta[1])+mult[0]*d_dx)
        return val
        

    class spec_func():

        def __init__(self, outer_instance):

            self.outer = outer_instance

            self.dt = outer_instance.dt
            self.N = outer_instance.Nx

            self.Jf = np.zeros((self.N, 2))
            self.Jx = np.zeros((self.N, self.N))
            self.Jx_s = np.zeros((self.N, self.N))
            self.Jt = np.eye(self.N)

            self.Rt = np.zeros((self.N, 1))
            self.Rx = np.zeros((self.N, 1))

            self.res_f = lambda X, i: outer_instance.Res(X, i)
            self.jac_f = outer_instance.Jac
            self.gjac_f = outer_instance.GJac


        def val(self, X, st = 0, end = None):
            if end == None:
                end = self.N
            np.subtract(X[st:end], self.outer.X_cur[st:end], out = self.Rt[st:end, :])

            for i in range(st, end):
                self.Rx[i, :] = self.res_f(X, i)

            return self.Rt[st:end]/self.dt + self.Rx[st:end] + self.outer.q[st:end]*self.outer.Nx

        def jac(self, X, st = 0, end = None):
            if end == None:
                end = self.N
            for i in range(st, end):
                self.Jf[i, :] = self.jac_f(X, i)

            for i in range(st, end):
                if i != end-1:
                    J1 = self.Jf[i, 1]
                    self.Jx_s[i, i+1] = J1
                J1 = self.Jf[i, 0]
                self.Jx_s[i, i] = J1

            self.Jx[st:end, st:end] = self.Jx_s[st:end, st:end]

            return self.Jx_s[st:end, st:end] + self.Jt[st:end, st:end]/self.dt

        def reset_jac(self, borders):
            for i in range(borders.shape[0]-1):
                bg = borders[i]
                end = borders[i+1]
                self.Jx_s[bg:end, bg:end] = 0
            

        def jac_gb(self, X_gb_prev, X, domain_borders):
            
            for bd in domain_borders[1:-1]:
                tmp = self.gjac_f(X, X_gb_prev, bd)
                self.Jx[bd, bd+1] = tmp
            
            return self.Jx+self.Jt/self.dt, self.Jx_s+self.Jt/self.dt


    # used for metrics
    def precompute_Jf(solver, X, N):
        Jf = np.zeros((N+1, 2))
        for i in range(N+1):
            Jf[i, :] = solver.FluxJac(X, i)
        return Jf