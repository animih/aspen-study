from solver import classic_solver
from aspen import aspen_solver
import partion
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
	Nx = 1000
	Nt = 100
	Nd = 6
	param = {
		'Nx': Nx, 
		'Nt': Nt,
		'Nd': Nd
	}
	D = Diffusion(Nx, 0.9, 1, 0.5e-1, model=model)

	part = partion.partion_equally
	solver = aspen_solver(param, D, part)
	solver.setBoundary(1, 1)
	solver.setInitial(0.5, 2, 1)
	solver.setSources([0.6], [0])

	X, code, message = solver.solve(debug = True)
	print(X)
