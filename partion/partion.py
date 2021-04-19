import numpy as np

# simply two aproaches for minimization functions
def metrics1(borders, cut, alpha = 1):
    f_cut = np.vectorize(cut)(borders[1:-1])
    reg = (borders[1:]-borders[:-1])**alpha

    return np.sum((f_cut[1:]+f_cut[:-1])/reg[1:-1] \
        + f_cut[-1]/reg[-1] + f_cut[0]/reg[0])

def metrics2(borders, cut, weights, alpha = 1):
    f_cut = np.vectorize(cut)(borders[1:-1])
    reg = np.array([np.sum(weights[bd1:bd2, bd1:bd2]) 
        for bd1, bd2 in zip(borders[:-1], borders[1:])])**alpha

    return np.sum((f_cut[1:]+f_cut[:-1])/reg[1:-1] \
        + f_cut[-1]/reg[-1] + f_cut[0]/reg[0])

# some of proposed metrics
def m1(solver, x, bd):
    value  = (np.abs(solver.FluxJac(x, bd)[0, 1]) \
                +np.abs(solver.FluxJac(x, bd)[0, 0]))
    return value
def m2(solver, X, bd, Nt, t_step):
    fr = X[bd-1, ::t_step].T
    sc = X[bd, ::t_step].T
    tmp = np.stack((fr, sc), axis = 0)
    cov = np.cov(tmp)
    val = np.abs(cov[0][1]/np.sqrt(cov[0][0]*cov[1][1]))
    return val
def m3(solver, x, bd, Nt):
    dt = 1/Nt

    kkk = solver.FluxJac(x, bd)
    val = np.abs(kkk[0, 1]) \
        + np.abs(kkk[0, 0])
    tmp = np.abs((1/dt+kkk[0, 0] \
        -solver.FluxJac(x, bd-1)[0, 1]))\
        +np.abs((1/dt+solver.FluxJac(x, bd+1)[0, 0]\
        -kkk[0, 1]))
    
    return tmp/val

# only 1D is considered

# equally partion, nice start
def partion_equally(Nx, Nd):
    res = np.zeros(Nd+1, dtype='int')
    res[1:Nd] = Nx//Nd*np.arange(1, Nd, 1, dtype='int')
    res[Nd] = Nx

    return res

# spectral bisection (proves to be bad in out prob)
def spec_bis(A):
	N = A.shape[0]
	D = np.diag(A @ np.ones(N))
	L = D - A

	w, v = np.linalg.eig(L)
	w[np.argmin(w)] = np.inf
	ind = np.argmin(w)
	solution = (v[ind] > 0)

	return v[ind]

# local search
# heating imitation 1D
class heat_1D():
    def __init__(self, T0 = 9600, alpha =0.3):
        self.T0 = T0
        self.T = T0
        self.alpha = alpha
    def find(self, borders, m_simple, metrics, steps=50, cl = True):
        nd = borders.shape[0]-1

        prev_val = metrics(borders)
        prev_borders = np.copy(borders)

        best_borders = np.copy(borders)
        best_val = np.copy(prev_val)

        for k in range(steps):
            for i in range(1, nd):

            	for k in range(2):
	                xi = np.random.randint(-1, 1)
	                val_prev = m_simple(borders[i])

	                if borders[i]+xi != borders[i+1] and borders[i]+xi != borders[i-1]:
	                    val = m_simple(borders[i]+xi)
	                    if val < val_prev:
	                        borders[i] += xi
	                        break
	                    elif np.random.random() < np.exp(-(val-val_prev)/np.abs(val_prev*self.T)):
	                        borders[i] += xi
	                        break
	                    
	                xi = - xi

            val = metrics(borders)

            if val < prev_val:
                prev_val = np.copy(val)
                prev_borders = np.copy(borders)
            elif np.random.random() < np.exp(-(val-prev_val)/np.abs(prev_val*self.T)):
                prev_val = np.copy(val)
                prev_borders = np.copy(borders)
            else:
                borders = np.copy(prev_borders)

            if val < best_val:
                best_borders = np.copy(borders)
                best_val = np.copy(val)
            if cl:
            	self.decrease()
            else:
            	self.increase()

        return prev_borders

    def decrease(self):
        self.T /= (1+self.alpha)
    def increase(self):
        self.T *= (1+self.alpha)


def generate_random(Nx, Nd, min_size=10):
    size = 0
    while(size < min_size):
        choice = 10+np.random.choice(Nx-10, Nd-1)
        choice = np.sort(choice)
        size = np.min(choice[1:] - choice[:-1])
    return np.concatenate(([0], choice , [Nx]), axis = 0)