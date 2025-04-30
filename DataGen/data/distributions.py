from scipy.stats import norm
from scipy.stats import uniform
import numpy as np
import scipy.stats as stats
import pdb as pdb
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


class mixture:

    def __init__(self, components, mixing_proportion):
        self.comps = components
        self.mixProp = mixing_proportion
        # self.sigmoid = activations.sigmoid()

    def pdf(self, x):
        return np.sum((p * comp.pdf(x) for (comp, p) in zip(self.comps, self.mixProp)), axis=1)

    def cdf(self, x):
        return np.sum((p * comp.cdf(x) for (comp, p) in zip(self.comps, self.mixProp)), axis=1)

    def rvs(self, size):
        sizes = np.cast['int32'](np.floor(size * self.mixProp))
        #pdb.set_trace()
        delta = np.cast['int32'](size - np.sum(sizes))
        ix = np.random.choice(np.size(self.mixProp), size=delta, p=self.mixProp)
        for ii in ix:
            sizes[ii] = sizes[ii] + 1
        dim = np.size(self.comps[0].rvs(size=1))
        x = np.empty([0, dim])
        for (s, comp) in zip(sizes, self.comps):
            rvs = comp.rvs(size=[s, 1])
            if s == 1:
                rvs = np.expand_dims(rvs, axis=0)
            if dim == 1:
                rvs = rvs.reshape((np.size(rvs), 1))
            x = np.concatenate((x, rvs), axis=0)
        #pdb.set_trace()
        return x

    def rvsCompInfo(self, size):
        sizes = np.cast['int32'](np.floor(size * self.mixProp))
        #pdb.set_trace()
        delta = np.cast['int32'](size - np.sum(sizes))
        ix = np.random.choice(np.size(self.mixProp), size=delta, p=self.mixProp)
        for ii in ix:
            sizes[ii] = sizes[ii] + 1
        dim = np.size(self.comps[0].rvs(size=1))
        x = np.empty([0, dim])
        y = np.empty([0, 1])
        k = 0
        for (s, comp) in zip(sizes, self.comps):
            rvs = comp.rvs(size=[s, 1])
            if s == 1:
                rvs = np.expand_dims(rvs, axis=0)
            if dim == 1:
                rvs = rvs.reshape((np.size(rvs), 1))
            x = np.concatenate((x, rvs), axis=0)
            #x = np.concatenate((x, comp.rvs(size=[s, 1])), axis=0)
            y = np.concatenate((y, np.zeros([s, 1]) + k), axis=0)
            k = k + 1
        #pdb.set_trace()
        return x, y

    def component_pdfs(self, x):
        return (comp.pdf(x) for comp in self.comps)

    def responsibility(self, x):
        R = self.component_pdfs(x)
        #pdb.set_trace()
        R = [a * d for (a, d) in zip(self.mixProp, R)]
        denom = sum(R)[:, None]
        R = np.hstack([r[:, None]/denom for r in R])
        return R