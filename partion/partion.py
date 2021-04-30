import numpy as np
import torch
from heapq import heappop, heappush
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

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
        return 0
    elif i - j == 1:
        return np.abs(1/Jf[i, 0]) + np.abs(1/Jf[i, 1])
    elif j - i == 1:
        return np.abs(1/Jf[i+1, 1]) + np.abs(1/Jf[i+1, 0])
    else:
        return 0

def m3(solver, Jf, Jf_n, i, j):
    if i - j == 1:
        dt = 1/solver.param.Nt
        t1 = np.abs(1/dt+Jf[i+1, 0] - Jf[i, 1])
        t2 = np.abs(Jf[i+1, 0] - Jf[i, 1] \
            -Jf_n[i+1, 0] + Jf_n[i, 1])
        t3 = np.abs(Jf[i, 0]) + np.abs(Jf[i, 1])
        t4 = np.abs(Jf[j+1, 0] - Jf[j, 1] \
            -Jf_n[j+1, 0] + Jf_n[j, 1])
        return np.abs(Jf[i, 1])*t2/t1**2\
            + np.abs(Jf[i, 0])*t4/t3**2
    if i == j:
        dt = 1/solver.param.Nt
        t1 = np.abs(1/dt+Jf[i+1, 0] - Jf[i, 1])
        t2 = np.abs(Jf[i+1, 0] - Jf[i, 1] \
            -Jf_n[i+1, 0] + Jf_n[i, 1])
        return (t2**2/t1**4)
    else:
        return 0


# a fancy func to construct adjencity matrix
# also chain rule can be used to dense the matrix
def adj_matrix(m_ij, Nx, dense = 0):
    f = np.vectorize(m_ij)

    main = f(np.arange(Nx), np.arange(Nx))
    tmp1 = f(np.arange(1, Nx), np.arange(Nx-1))
    out =  np.diag(tmp1, k = -1)+np.diag(main)

    if dense:
        #main = f(np.arange(Nx), np.arange(Nx))
        #out += np.diag(main)
        for k in range(2, dense+2):
            tmp1 = f(np.arange(Nx-k), np.arange(k, Nx))
            out +=  np.diag(tmp1, k = -k)

    out += out.T

    return out


# only 1D is considered

# equally partion, nice start
def partion_equally(Nx, Nd):
    res = np.zeros(Nd+1, dtype='int')
    res[1:Nd] = Nx//Nd*np.arange(1, Nd, 1, dtype='int')
    res[Nd] = Nx

    return res

# spectral bisection
def spec_bis(A, inv = False, k = 2):
    Ad = torch.from_numpy(A).double()
    N = Ad.shape[0]
    D = np.diag(A @ np.ones(N))
    D = torch.from_numpy(D).double()
    L = D - Ad
    
    if inv:
        w, v = torch.lobpcg(L, k = k, B = D, largest = False)
        #w = np.abs(w)
    else:
        w, v = torch.lobpcg(L, k = k, largest = False)
    #w[np.argmin(w)] = np.inf

    return w, v

def domain_builder1(A, Nd, inv = False):
    Nx = A.shape[0]
    borders = [0, Nx]
    h = []
    heappush(h, (-Nx, (0, Nx)))
    k = 0
    while k != Nd-1:
        _, (beg, end) = heappop(h)
        w_list, v_list = spec_bis(A[beg:end, beg:end], inv = inv, k = 2)
        s = np.sign(v_list[:, 1])
        #plt.plot(v_list[:, 1])
        tmp = [beg]
        for i in range(1, end-beg):
            if s[i] != s[i-1]:
                bd = i + beg
                tmp.append(bd)
                borders.append(bd)
                k += 1

                if k == Nd-1:
                    break
        tmp.append(end)

        for i in range(len(tmp)-1):
            beg = tmp[i]
            end = tmp[i+1]
            heappush(h, (-(end-beg), (beg, end)))

    borders = np.sort(borders)
    return borders

def domain_builder2(A, Nd, inv = False, k = 5):
    Nx = A.shape[0]

    w, v = spec_bis(A, inv = inv, k = k+1)
    # dropout trivial solution
    v = v.numpy()

    v[:, 0] = np.linspace(-1, 1, Nx)

    for i in range(0, k):
        v[:, i] /= np.sqrt(np.sum(v[:, i]**2/Nx))

    # (optional) uncomment for visualization
    # usally smth like deformed Lissague figures appear
    '''
    for v1, v2 in zip(v[:, :-1].T, v[:, 1:].T):
        plt.scatter(v1.T, v2.T)
        plt.show()
    '''
    
    kmeans = KMeans(n_clusters=Nd, random_state=0).fit(v)
    lb = kmeans.labels_
    #print(kmeans.cluster_centers_)
    
    borders = [0]
    j = 0
    flag = True
    for i in range(1, Nx):
        if lb[i] != lb[i-1]:
            if j < Nd-1 or True:
                borders.append(i)
            j += 1
        if j > Nd-1 and flag:
            print('clustering alg returned more borders than declared, more domains should be considered')
            flag = False
            #break

    borders.append(Nx)
    borders = np.sort(borders)
    
    return borders

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


def generate_random(Nx, Nd, min_size=20):
    size = 0
    while(size < min_size):
        choice = 10+np.random.choice(Nx-10, Nd-1)
        choice = np.sort(choice)
        size = np.min(choice[1:] - choice[:-1])
    return np.concatenate(([0], choice , [Nx]), axis = 0)