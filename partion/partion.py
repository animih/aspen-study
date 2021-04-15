import numpy as np

def partion_equally(Nx, Nd):
    res = np.zeros(Nd+1, dtype='int')
    res[1:Nd] = Nx//Nd*np.arange(1, Nd, 1, dtype='int')
    res[Nd] = Nx

    return res