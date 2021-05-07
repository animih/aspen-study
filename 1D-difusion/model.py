import sys
import torch
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
class border_changer():
    def __init__(self, Nx, Nd):
        self.init_X = 1/2-torch.rand((Nx, Nd), dtype=torch.double)
        self.init_X[:, 0] = 1
    def func(self, solver, X, Nd):
        Jf = partion.precompute_Jf(solver, X[:, -1], Nx)
        Jf_n = partion.precompute_Jf(solver, X[:, -1]+1e-2, Nx)
        func1 = lambda k, m: partion.m3(solver, Jf, Jf_n, k, m)
        A1 = partion.adj_matrix(func1, Nx)
        borders = partion.domain_builder2(A1, Nd,
            inv = False, k =Nd)
        return borders

if __name__ == "__main__":
    Nx = 1000
    Nt = 100
    Nd = 6
    param = {
        'Nx': Nx, 
        'Nt': Nt
    }
    # inital conditions
    x = np.linspace(0, 1, Nx)
    x0 = np.exp(-x/5)*np.cos(3*np.pi/2*x)**2
    x0 = x0[::-1]
    x0 = np.reshape(x0, (-1, 1))
    # diffuison init
    D = Diffusion(Nx, 0.9, 1, 0.5e-1, model=model)

    border_changer = border_changer(Nx, Nd)

    # aspen init
    domains = partion.partion_equally(Nx, Nd)
    nl_solver = aspen(Nd, domains, 1e-2, crit_abs = 1e-3)


    # solver init
    solver = one_phase(param, D, nl_solver, bd_ch = border_changer.func)
    solver.setBoundary(0, 1)
    solver.init_log()
    solver.x0 = np.copy(x0)

    X, mes= solver.solve()
    print(X, mes)
