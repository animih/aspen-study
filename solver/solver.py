import numpy as np
import time

class solver():
    class Parameters():
        def __init__(self, param):
            raise NotImplemented
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

class classic_solver(solver):
    def __init__(self, param, diffusion):
        super(classic_solver, self).__init__(param, diffusion)
        self.timelog = self.TimeLog(Nt=self.param.Nt)
    class Parameters():
        def __init__(self, param):
            self.Nx = param['Nx']
            self.Nt = param['Nt']
    class TimeLog():
        def __init__(self, linsol=0, resbld=0, jacbld=0, Nt=0):
            self.linsol = linsol
            self.resbld = resbld
            self.jacbld = jacbld
            self.kn = np.zeros((Nt))
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

    def calcFluxJac(self, X, i):
        val = np.zeros((1, 2))
        if self.bc.first:
            delta = np.zeros((2, 1))
            if i == 0:
                delta[0] = self.bc.val[1]
                delta[1] = X[i]
            elif i == self.param.Nx:
                delta[0] = X[i-1]
                delta[1] = self.bc.val[0]
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


    def buildFlow(self, X, dt):
        Jt = np.eye(self.param.Nx)/dt
        Jf = np.zeros((self.param.Nx + 1, 2))
        mask = range(self.param.Nx + 1)
        for i in mask:
            Jf[i, :] = self.calcFluxJac(X[:, 1], i)
        return Jt, Jf


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

    def solve(self):
        return_code = 0
        return_message = 'OK'

        # time settings
        t = 0.0
        dt = 1./self.param.Nt
        dt_min = dt*1e-3

        # initial condition
        X = np.concatenate((self.x0, self.x0), axis=1)

        # Newton iteration parameters
        kmax = 15
        kmin = 4
        crit_rel = 1e-2
        crit_abs = 1e-3

        #initial condition
        self.X[:, 0] = self.x0.flatten()

        # solution process
        R0 = 1
        nstep = 0
        while t < 1.0:
            dt = min(dt, self.t[nstep+1] - t)
            X[:, 1] = X[:, 0]

            is_converged = False
            for k in range(kmax):
                # Residual
                t_res = -time.time()
                Rt, W = self.buildResidual(X, dt)
                t_res += time.time()
                self.timelog.resbld += t_res
                Rx = W[1:self.param.Nx+1] - W[0:self.param.Nx]
                Rs = self.q
                R = Rt + Rx + Rs

                # convergence
                delta = np.linalg.norm(R)
                if k == 0:
                    R0 = delta

                is_conv_abs = (delta * dt) <= crit_abs
                is_conv_rel = delta <= (crit_rel*R0)
                is_converged = is_conv_abs or is_conv_rel

                if is_converged:
                    self.timelog.kn[nstep] += k
                    t += dt
                    X[:, 0] = X[:, 1]
                    break
                elif (k + 1 == kmax):
                    dt /= 4
                    break

                # Jacobian
                t_jac = -time.time()
                Jt, Jf = self.buildFlow(X,  dt)
                Jx = self.buildJacobian(Jf,  self.param.Nx)
                t_jac += time.time()
                self.timelog.jacbld += t_jac
                J = Jt + Jx

                # linear system solve
                t_lin = -time.time()
                # fixed
                dX = np.linalg.solve(J,   -R)
                t_lin += time.time()
                self.timelog.linsol += t_lin

                is_weak = np.sum(np.isnan(dX)) + np.sum(np.isinf(dX))
                if is_weak > 0:
                    dX[:] = 0

                # update
                X[:, 1] += np.reshape(dX,   self.param.Nx)

            if (not is_converged):
                if dt < dt_min:
                    return_message = 'Not converged'
                    break

            if self.t[nstep+1] == t:
                self.X[:, nstep+1] = X[:, 0]
                nstep += 1
                # not sure about next step...
                dt = 1./self.param.Nt

        return self.X, return_code, return_message