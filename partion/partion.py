import numpy as np

# simply two aproaches for minimization functions
def metrics1(borders, weights):
    Nd = borders.shape[0]-1
    val = 0
    for i in range(Nd):
        beg = borders[i]
        end = borders[i+1]
        val += np.sum(weights[end:, beg:end])/(end-beg)

    return val

def metrics2(borders, weights):
    Nd = borders.shape[0]-1
    val = 0
    for i in range(Nd):
        beg = borders[i]
        end = borders[i+1]
        val += np.sum(weights[end:, beg:end])/np.sum(weights[beg:end, beg:end])

    return val

# some of proposed metrics
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

def precompute_Jf(solver, X, N):
    Jf = np.zeros((N+1, 2))
    for i in range(N+1):
        Jf[i, :] = solver.FluxJac(X, i)
    return Jf

def m2(solver, X, i, j, t_step = 5):
    fr = X[i, ::t_step].T
    sc = X[j, ::t_step].T
    tmp = np.stack((fr, sc), axis = 0)
    cov = np.cov(tmp)
    val = np.abs(cov[0][1]/np.sqrt(cov[0][0]*cov[1][1]))
    #val = cov[0][1]
    return val

def m1(solver, Jf, i, j):
    if i == j:
        return solver.param.Nt+Jf[i+1, 0] \
            - Jf[i, 1]
    elif i - j == 1:
        return - Jf[i, 0]
    elif j - i == 1:
        return Jf[i+1, 1]
    else:
        return 0

# a fancy func to construct adjencity matrix
# also chain rule can be used to dense the matrix
def adj_matrix(m_ij, Nx , sc = 0, EPS = 1e-3):
    f = np.vectorize(m_ij)
    if sc:
        main = f(np.arange(Nx), np.arange(Nx))
        tmp1 = f(np.arange(Nx-1), np.arange(1, Nx))
        tmp2 = f(np.arange(1, Nx), np.arange(Nx-1))
        out =  np.diag(main) + np.diag(tmp1, k = 1) + \
            np.diag(tmp2, k = -1)
        if sc > 1:
            tmp =  -np.linalg.inv(out) @ out.T
            out = out @ tmp.T
    else:
        out = np.diag(np.vectorize(m_ij)(np.arange(Nx), np.arange(Nx)))

        for i in range(Nx-1):
            out[i, i+1:] = 2*np.vectorize(m_ij)(i*np.ones(Nx-i-1, dtype='int'),
                np.arange(Nx-i-1))

    # to work with nondirected, positive weighted graph
    out = np.abs(out)
    out = 1/2*(out+out.T)

    return out


# only 1D is considered

# equally partion, nice start
def partion_equally(Nx, Nd):
    res = np.zeros(Nd+1, dtype='int')
    res[1:Nd] = Nx//Nd*np.arange(1, Nd, 1, dtype='int')
    res[Nd] = Nx

    return res

# spectral bisection
def spec_bis(A, inv = False):
    N = A.shape[0]
    D = np.diag(A @ np.ones(N))
    L = D - A

    if inv:
        Dinv = np.linalg.inv(D)
        w, v = np.linalg.eig(Dinv@L)
        #w = np.abs(w)
    else:
        w, v = np.linalg.eig(L)
    w[np.argmin(w)] = np.inf
    ind = np.argmin(w)

    return v[ind]

# local search
# heating imitation 1D
class heat_1D():
    def __init__(self, T0 = 9600, alpha =0.3):
        self.T0 = T0
        self.T = T0
        self.alpha = alpha
    def find(self, borders, metrics, steps=50, cl = True):
        nd = borders.shape[0]-1

        prev_val = metrics(borders)
        prev_borders = np.copy(borders)

        best_borders = np.copy(borders)
        best_val = np.copy(prev_val)

        for k in range(steps):
            for i in range(1, nd):

                for k in range(2):
                    xi = np.random.randint(-1, 1)

                    if borders[i]+xi != borders[i+1] and borders[i]+xi != borders[i-1]:
                        borders[i]+= xi
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