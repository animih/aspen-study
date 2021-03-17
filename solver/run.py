import solver
import numpy as np


def model(u):
    return np.array([u**2+1, 2*max(u, 0)])


Nt = 100
Nx = 1000
x0 = solver.setInitial(Nx, 0.5, 2, 1)
bc = solver.Boundary(0, 0)
D = solver.Diffusion(Nx, 0.9, 1, 0.5e-1, model=model)
q = solver.setSource(Nx, 0.6, 0)
param = solver.Parameters(
    Nx=Nx,
    Nt=Nt,
    D=D,
    bc=bc,
    src=q,
    x0=x0)
sol, code, message = solver.solve(param)
X = sol.X
print(X)
