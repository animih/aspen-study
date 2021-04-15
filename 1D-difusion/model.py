import sys
sys.path.append('../nonlin')
sys.path.append('../partion')
sys.path.append('../one_phase')

from aspen import aspen
from newton import newton

import partion

from one_phase import one_phase
import numpy as np


def model(u):
    return np.array([u**2+1, 2*max(u, 0)])

class Diffusion():
    def __init__(self, Nx, Amp=0, Period=0, Scale=1, model=None):
        self.val = np.zeros((Nx+1, 1))
        for i in range(Nx+1):
            self.val[i] = (
                Amp * np.sin(Period * 2*np.pi * (i+1) / (Nx+1)) + 1) * Scale
        self.model = model


if __name__ == "__main__":
    Nx = 100
    Nt = 10
    Nd = 4
    param = {
        'Nx': Nx, 
        'Nt': Nt
    }

    # diffuison init
    D = Diffusion(Nx, 0.9, 1, 0.5e-1, model=model)

    # aspen init
    domains = partion.partion_equally(Nx, Nd)
    nl_solver = newton(1e-2, kmax = 14, crit_abs = 1e-3)


    # solver init

    solver = one_phase(param, D, nl_solver)
    solver.setBoundary(1, 1)
    solver.setInitial(0.5, 2, 1)
    solver.setSources([0.6], [0])

    X, mes, code= solver.solve()
    print(X, mes, code)
