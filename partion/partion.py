import numpy as np
import torch
from heapq import heappop, heappush
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from time import time

# some of proposed ways to regulize edgecut

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

    # dense is needed to restrict the maximum degree of node
    if dense:
        for k in range(2, dense+2):
            tmp1 = f(np.arange(Nx-k), np.arange(k, Nx))
            out +=  np.diag(tmp1, k = -k)

    # we consider only nondirected graphs
    out += out.T

    return out

# equally partion for 1D problem
def partion_equally1D(Nx, Nd):
    res = [ np.arange(Nx//Nd*i, Nx//Nd*(i+1), 1, dtype = 'int') for i in range(Nd-1) ]
    res.append(np.arange(Nx//Nd*(Nd-1), Nx, 1, dtype = 'int'))

    return res

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

# first approach (not used)
# that solves multiple domain problem
# by perofroming iteratively bisection
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

# second approach (recommended)
# that solves multiple domain problem
# by clusterizing in spectral embedded space
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