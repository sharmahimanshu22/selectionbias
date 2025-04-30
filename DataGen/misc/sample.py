import numpy as np
from scipy.stats import bernoulli
import pdb


def posterior_sample(x, posterior):
    # p = posterior(x)
    p = posterior
    ix = bernoulli.rvs(p)
    ix = np.cast['bool'](ix).flatten()
    xx = x[ix, 0:np.shape(x)[1]]
    return xx, ix


def posterior_sample_n(x, posterior, n):
    # p = posterior(x)
    p = posterior
    ix = bernoulli.rvs(p)
    ix = np.cast['bool'](ix).flatten()
    ixx = np.arange(np.size(ix))[ix]
    ixx = np.random.choice(ixx, n, replace=True)
    xx = x[ixx, :]
    return xx, ixx


def batch(x, y, n_p, n_u):
    x_p, ix_p = batchPos(x, y, n_p)
    x_u, ix_u = batchUL(x, y, n_u)
    xx = np.concatenate((x_p, x_u), axis=0)
    ix = np.concatenate((ix_p, ix_u), axis=0)
    return xx, y[ix, :], x_p, x_u, ix


def batchPos(x, y, n_p):
    return batchY(x, y, 1, n_p)


def batchUL(x, y, n_u):
    return batchY(x, y, 0, n_u)


def batchY(x, y, value, n, *args):
    ix = (y == value).flatten( )
    ix_all = np.arange(np.size(y))
    ix = ix_all[ix]
    if args:
        p = args[0].flatten()
        p = p[ix]
        ix_p = bernoulli.rvs(p)
        ix_p = np.cast['bool'](ix_p)
        ix = ix[ix_p]
    ix = np.random.choice(ix, n, replace=True)
    xx = x[ix, :]
    return xx, ix


def batchTrueAndFakePos(x, y, posterior, n_tp, n_fp):
    x_tp, ix_tp = batchPos(x, y, n_tp)
    x_fp, ix_fp = batchY(x, y, 0, n_fp, posterior)
    xx = np.concatenate((x_tp, x_fp), axis=0)
    ix = np.concatenate((ix_tp, ix_fp), axis=0)
    return xx, y[ix, :], x_tp, x_fp, ix

