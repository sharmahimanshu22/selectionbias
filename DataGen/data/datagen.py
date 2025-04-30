from scipy.stats import norm
from scipy.stats import uniform
from random import seed
from random import randint
from random import random
import numpy as np
from DataGen.data.distributions import mixture
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

    def __init__(self, dist_p, dist_n, alpha):
        self.dist_p = dist_p
        self.dist_n = dist_n
        self.alpha = alpha

        
    def data_pos(self, n):
        #pdb.set_trace()
        if isinstance(self.dist_p, mixture):
            x, c = self.dist_p.rvsCompInfo(size=n)
            x = np.reshape(x, newshape=(n, -1))
        else:
            #pdb.set_trace()
            x = np.reshape(self.dist_p.rvs(size=(n,1)), newshape=(n, -1))
            c = np.ones((x.shape[0], 1))
        return x, c

    def data_neg(self, n):
        if isinstance(self.dist_n, mixture):
            x, c = self.dist_n.rvsCompInfo(size=n)
            x = np.reshape(x, newshape=(n, -1))
        else:
            x = np.reshape(self.dist_n.rvs(size=(n, 1)), newshape=(n, -1))
            c = np.ones((x.shape[0], 1))
        return x, c

    def data_pos_compInfo(self, n):
        #pdb.set_trace()
        x, c = self.dist_p.rvsCompInfo(size=n)
        x = np.reshape(x, newshape=(n, -1))
        return x, c

    def data_neg_compInfo(self, n):
        x, c = self.dist_n.rvsCompInfo(size=n)
        x = np.reshape(x, newshape=(n, -1))
        return x, c

    # def data_ul(self, n, alpha=None):
    #     if alpha == None:
    #         alpha = self.alpha
    #     n_up = np.cast['int32'](np.floor(n * alpha))
    #     n_un = n - n_up
    #     x_up = self.data_pos(n_up)
    #     x_un = self.data_neg(n_un)
    #     x = np.concatenate((x_up, x_un), axis=0)
    #     y = np.zeros([n, 1])
    #     y[np.arange(x_up.shape[0]), 0] = 1
    #     return x, y

    def pu_data(self, n_p, n_u, alpha = None):
        if alpha == None:
            alpha = self.alpha
        x_p, c_p = self.data_pos(n_p)
        x_u, y_u, c_u = self.pn_data(n_u, alpha)
        x_pu = np.concatenate((x_p, x_u), axis=0)
        y_pu = np.zeros([x_pu.shape[0], 1])
        y_pu[np.arange(n_p), 0] = 1
        y_pn = np.vstack((np.ones([n_p, 1]), y_u))
        c_pu = np.vstack((c_p, c_u))
        # y_pn = y_pu
        # y_pn[x_pu.size(0):(self.n_p - 1):-1, 0] = y_u
        return x_pu, y_pu, y_pn, c_pu, x_p, x_u, y_u, c_p, c_u
        
    def pn_data(self, n, alpha=None):
        if alpha == None:
            alpha = self.alpha
        n_p = np.cast['int32'](np.floor(n * alpha))
        n_n = n - n_p
        x_p, c_p = self.data_pos(n_p)
        x_n, c_n = self.data_neg(n_n)
        y_p = np.ones((x_p.shape[0], 1))
        y_n = np.zeros((x_n.shape[0], 1))
        x = np.vstack((x_p, x_n))
        y = np.vstack((y_p, y_n))
        c = np.vstack((c_p, c_n))
        return x, y, c, x_p, x_n, c_p, c_n
    
    def dens_pos(self, x):
        return self.dist_p.pdf(x)

    def dens_neg(self, x):
        return self.dist_n.pdf(x)

    def dens_mix(self, x, alpha = None):
        if alpha == None:
            alpha = self.alpha
        return alpha * self.dens_pos(x) + (1 - alpha) * self.dens_neg(x)

    def pn_posterior(self, x, alpha = None):
        if alpha == None:
            alpha = self.alpha
        return alpha * self.dens_pos(x) / self.dens_mix(x, alpha)

    def pu_posterior(self, x, n_p, n_u, alpha=None):
        if alpha == None:
            alpha = self.alpha
        n_up = np.cast['int32'](np.floor(n_u * alpha))
        c1 = n_p / (n_u + n_p)
        c2 = (n_up + n_p) / (n_u + n_p)
        return c1 * self.dens_pos(x) / self.dens_mix(x, c2)

    def pn_posterior_sts(self, x, n_p, n_u, alpha = None):
        if alpha == None:
            alpha = self.alpha
        n_up = np.cast['int32'](np.floor(n_u * alpha))
        c = (n_up + n_p) / (self.n_u + self.n_p)
        return self.pn_posterior(x, c)

    def pn_posterior_cc(self, x):
        return self.pn_posterior(x, self.alpha)

    def pn_posterior_balanced(self, x):
        return self.pn_posterior(x, 0.5)
    



class GaussianDG(DataGenerator):

    def __init__(self, mu, sig, alpha):
        dist_p = norm(loc=0, scale=1)
        dist_n = norm(loc=mu, scale=sig)
        super(GaussianDG, self).__init__(dist_p=dist_p, dist_n=dist_n, alpha=alpha)


class UniformDG(DataGenerator):

    def __init__(self, mu, sig, alpha):
        self.dist_p = uniform(loc=0, scale=1)
        self.dist_n = uniform(loc=mu, scale=sig)
        super(UniformDG, self).__init__(dist_p=self.dist_p, dist_n=self.dist_n, alpha=alpha)


class MixtureDG(DataGenerator):
    def __int__(self, dist_p, dist_n, alpha):
        super(MixtureDG, self).__init__(dist_p=dist_p, dist_n=dist_n, alpha=alpha)
    def responsibility(self,x):
        comps = self.dist_p.comps + self.dist_n.comps
        mProp = np.hstack((self.alpha*self.dist_p.mixProp, (1-self.alpha)*self.dist_n.mixProp))
        mix = mixture(comps, mProp)
        R = mix.responsibility(x)
        return R
    def updateMixProps(self, alpha=None, p_pos=None, p_neg=None):
        if p_pos is not None:
            self.dist_p = mixture(self.dist_p.comps, p_pos)
        if p_neg is not None:
            self.dist_n = mixture(self.dist_n.comps, p_neg)
        if alpha is not None:
            self.alpha = alpha

class NormalMixDG(MixtureDG):

    def __init__(self, mu_pos, sig_pos, p_pos, mu_neg, sig_neg, p_neg, alpha):
        components_pos = [norm(loc=mu, scale=sig) for (mu, sig) in zip(mu_pos, sig_pos)]
        components_neg = [norm(loc=mu, scale=sig) for (mu, sig) in zip(mu_neg, sig_neg)]
        self.dist_pos = mixture(components_pos, p_pos)
        self.dist_neg = mixture(components_neg, p_neg)
        super(NormalMixDG, self).__init__(dist_p=self.dist_pos, dist_n=self.dist_neg, alpha=alpha)


class MVNormalMixDG(MixtureDG):

    def __init__(self, mu_pos, sig_pos, p_pos, mu_neg, sig_neg, p_neg, alpha):
        components_pos = [mvn(mean=mu, cov=sig) for (mu, sig) in zip(mu_pos, sig_pos)]
        components_neg = [mvn(mean=mu, cov=sig) for (mu, sig) in zip(mu_neg, sig_neg)]
        dist_p = mixture(components_pos, p_pos)
        dist_n = mixture(components_neg, p_neg)
        super(MVNormalMixDG, self).__init__(dist_p=dist_p, dist_n=dist_n, alpha=alpha)



