import numpy as np
import pdb

def randomSort(x):
    #x is assumed to be a vector
    x_uni = np.unique(x)
    #print(x_uni)
    x_uni = np.sort(x_uni)
    deltavec = x_uni[1:] - x_uni[0:-1]
    delta = np.min(np.abs(deltavec), axis=0)
    #pdb.set_trace()
    rand = (delta/2)*np.random.uniform(low=-1.0, high=1.0, size=x.shape)
    x_prt = x + rand
    ix = np.argsort((x_prt).flatten())
    #pdb.set_trace()
    return x[ix, :], ix, x_prt[ix, :]
