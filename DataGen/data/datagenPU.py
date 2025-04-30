from scipy.stats import norm
from scipy.stats import uniform
from random import seed
from random import randint
from random import random
import numpy as np
from data.distributions import mixture
from sklearn.datasets import make_spd_matrix as spd
from scipy.stats import dirichlet
from sklearn import metrics
from scipy.stats import multivariate_normal as mvn
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import pdb as pdb
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


class DataGenerator:

    def __init__(self, dist_p, dist_n, alpha, n_p, n_u):
        self.dist_p = dist_p
        self.dist_n = dist_n
        self.alpha = alpha
        self.n_p = n_p
        self.n_u = n_u
        self.n_up = np.cast['int32'](np.floor(n_u * alpha))
        self.n_un = self.n_u - self.n_up
        
    def _shuffle_in_unison(self, a, b):
        p = np.random.permutation(len(a))
        return a[p],b[p]
        
    def data_pos(self, n):
        #pdb.set_trace()
        return np.reshape(self.dist_p.rvs(size=n), newshape=(n, -1))

    def data_neg(self, n):
        return np.reshape(self.dist_n.rvs(size=n), newshape=(n, -1))

    def data_ul(self, n, alpha):
        n_up = np.cast['int32'](np.floor(n * alpha))
        n_un = n - n_up
        x_up = self.data_pos(n_up)
        x_un = self.data_neg(n_un)
        x = np.concatenate((x_up, x_un), axis=0)
        y = np.zeros([n, 1])
        y[np.arange(x_up.shape[0]), 0] = 1
        return x, y

    def pu_data(self):
        x_p = self.data_pos(self.n_p)
        x_u, y_u = self.data_ul(self.n_u, self.alpha)
        x_pu = np.concatenate((x_p, x_u), axis=0)
        y_pu = np.zeros([x_pu.shape[0], 1])
        y_pu[np.arange(self.n_p), 0] = 1
        y_pn = np.concatenate((np.ones([self.n_p, 1]), y_u), axis=0)
        # y_pn = y_pu
        # y_pn[x_pu.size(0):(self.n_p - 1):-1, 0] = y_u
        return x_pu, y_pu, y_pn
        
    def pn_data(self, n, alpha):
        x_p = self.data_pos(int(alpha*n))
        x_n = self.data_neg(int((1-alpha)*n))
        y_p = np.ones((x_p.shape[0], 1))
        y_n = np.zeros((x_n.shape[0], 1))
        x = np.vstack((x_p, x_n))
        y = np.vstack((y_p, y_n))
        x, y = self._shuffle_in_unison(x, y)
        return x, y
    
    def dens_pos(self, x):
        return self.dist_p.pdf(x)

    def dens_neg(self, x):
        return self.dist_n.pdf(x)

    def dens_mix(self, x, a):
        return a * self.dens_pos(x) + (1 - a) * self.dens_neg(x)

    def pn_posterior(self, x, a):
        return a * self.dens_pos(x) / self.dens_mix(x, a)

    def pu_posterior(self, x):
        c1 = self.n_p / (self.n_u + self.n_p)
        c2 = (self.n_up + self.n_p) / (self.n_u + self.n_p)
        return c1 * self.dens_pos(x) / self.dens_mix(x, c2)

    def pn_posterior_sts(self, x):
        c = (self.n_up + self.n_p) / (self.n_u + self.n_p)
        return self.pn_posterior(x, c)

    def pn_posterior_cc(self, x):
        return self.pn_posterior(x, self.alpha)

    def pn_posterior_balanced(self, x):
        return self.pn_posterior(x, 0.5)


class GaussianDG(DataGenerator):

    def __init__(self, mu, sig, alpha, n_p, n_u):
        dist_p = norm(loc=0, scale=1)
        dist_n = norm(loc=mu, scale=sig)
        super(GaussianDG, self).__init__(dist_p=dist_p, dist_n=dist_n, alpha=alpha, n_p=n_p, n_u=n_u)


class UniformDG(DataGenerator):

    def __init__(self, mu, sig, alpha, n_p, n_u):
        dist_p = uniform(loc=0, scale=1)
        dist_n = uniform(loc=mu, scale=sig)
        super(UniformDG, self).__init__(dist_p=dist_p, dist_n=dist_n, alpha=alpha, n_p=n_p, n_u=n_u)


class NormalMixDG(DataGenerator):

    def __init__(self, mu_pos, sig_pos, p_pos, mu_neg, sig_neg, p_neg, alpha, n_pos, n_ul):
        components_pos = [norm(loc=mu, scale=sig) for (mu, sig) in zip(mu_pos, sig_pos)]
        components_neg = [norm(loc=mu, scale=sig) for (mu, sig) in zip(mu_neg, sig_neg)]
        dist_pos = mixture(components_pos, p_pos)
        dist_neg = mixture(components_neg, p_neg)
        super(NormalMixDG, self).__init__(dist_p=dist_pos, dist_n=dist_neg, alpha=alpha, n_p=n_pos, n_u=n_ul)


class MVNormalMixDG(DataGenerator):

    def __init__(self, mu_pos, sig_pos, p_pos, mu_neg, sig_neg, p_neg, alpha, n_pos, n_ul):
        components_pos = [mvn(mean=mu, cov=sig) for (mu, sig) in zip(mu_pos, sig_pos)]
        components_neg = [mvn(mean=mu, cov=sig) for (mu, sig) in zip(mu_neg, sig_neg)]
        dist_pos = mixture(components_pos, p_pos)
        dist_neg = mixture(components_neg, p_neg)
        super(MVNormalMixDG, self).__init__(dist_p=dist_pos, dist_n=dist_neg, alpha=alpha, n_p=n_pos, n_u=n_ul)


