import numpy as np
import torch
from heapq import heappop, heappush
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from time import time

# some of proposed ways to regulize

# reg by domian size |V|
def metrics1(borders, weights):
    Nd = borders.shape[0]-1
    val = 0
    for i in range(Nd):
        beg = borders[i]
        end = borders[i+1]
        val += np.sum(weights[end:, beg:end])/(end-beg)

    return val

# reg by domain weight S(V)
def metrics2(borders, weights):
    Nd = borders.shape[0]-1
    val = 0
    for i in range(Nd):
        beg = borders[i]
        end = borders[i+1]
        val += np.sum(weights[end:, beg:end])/np.sum(weights[beg:end, beg:end])

    return val


# a fancy func to construct adjencity matrix
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

def generate_fr_eq(Nx, Nd):
    borders = partion_equally(Nx, Nd)

    max_step = Nx//Nd-10

    if Nd == 2:
        borders[1] += np.random.randint(-max_step, max_step)
    else:
        borders[1] += np.random.randint(-max_step, max_step//2)
        borders[-2] += np.random.randint(-max_step//2, max_step)

        for i in range(2, Nd-1):
            borders[i] += np.random.randint(-max_step//2, max_step//2)

    return borders


# spectral bisection
def spec_bis(A, inv = False, k = 2, X = None):
    Ad = torch.from_numpy(A).double()
    N = Ad.shape[0]
    D = np.diag(A @ np.ones(N))
    D = torch.from_numpy(D).double()
    L = D - Ad
    
    if inv:
        w, v = torch.lobpcg(L, k = k, B = D, X=X, largest = False)
        #w = np.abs(w)
    else:
        w, v = torch.lobpcg(L, k = k, X=X, largest = False)
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

def domain_builder2(A, Nd, inv = False, k = 5, X = None):
    Nx = A.shape[0]
    w, v = spec_bis(A, inv = inv, k = k+1, X = X)
    # dropout trivial solution
    tmp = v.numpy()

    tmp[:, 0] = np.linspace(-1, 1, Nx)

    for i in range(0, k):
        tmp[:, i] /= np.sqrt(np.sum(tmp[:, i]**2/Nx))

    # (optional) uncomment for visualization
    # usally smth like deformed Lissague figures appear
    '''
    for v1, v2 in zip(v[:, :-1].T, v[:, 1:].T):
        plt.plot(v2.T)
        plt.show()
    '''
    
    kmeans = KMeans(n_clusters=Nd, random_state=0).fit(tmp)
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
    if X != None:
        return v, borders
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