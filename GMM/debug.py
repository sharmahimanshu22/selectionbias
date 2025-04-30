import matplotlib.pyplot as plt

from misc import sortedplot as sp
from IPython.display import display
import numpy as np
from scipy.stats import norm
from scipy.stats import bernoulli
from GaussianMixEmbedding.debug import Debug as DBBase
from GMM.utils import matchComponents

import itertools
#from MassSpec.I2GivenI1_NN2.model import I2GivenI1_NN2 as model
import pdb
from  misc.randomSort import randomSort


class Debug:
    def __init__(self):
        self.ll = []

    def attachData(self, X, DG):
        self.X = X
        self.DG = DG
        self.postTrue = [dg.pn_posterior(x) for (x, dg) in zip(self.X, DG)]
        #pdb.set_trace()
        self.RTrue = [dg.responsibility(x) for (x, dg) in zip(self.X, DG)]
        self.nMix = len(X)

    def attachGMM(self, gmm):
        self.gmm = gmm

    def llPlot(self):
        self.ll.append(self.gmm.logLikelihood(self.X))
        sp.sortedplot(self.ll, label='ll', ax=self.axs[0,0])

    def responsibilityError(self):
        #pdb.set_trace()
        for i, (x, rTrue, r) in enumerate(zip(self.X, self.RTrue, self.gmm.responsibility(self.X))):
            permTrue, perm = matchComponents(x, rTrue, r)
            [self.axs[i+1, 0].scatter(rTrue[:,pT], r[:,p], label='resp') for (pT,p) in zip(permTrue, perm)]

    def scatterPlot(self):
        dim = self.gmm.dim
        rDims = np.random.choice(dim, (1,2), replace=True)
        for s, (x, RTrue, R) in enumerate(zip(self.X, self.RTrue, self.gmm.responsibility(self.X))):
            IXTrue = [bernoulli.rvs(r).astype('bool').flatten() for r in RTrue.T]
            IX = [bernoulli.rvs(r).astype('bool').flatten() for r in R.T]
            for (i, d) in enumerate(rDims):
                d0 = d[0]
                d1 = d[1]
                #pdb.set_trace()
                [self.axs[s+1, (2*i)+1].scatter(x[ix, d0], x[ix, d1]) for ix in IXTrue]
                self.axs[s+1, (2*i)+1].set_title("true")
                [self.axs[s+1, (2*i+1)+1].scatter(x[ix, d0], x[ix, d1]) for ix in IX]
                self.axs[s+1, (2*i+1)+1].set_title("predicted")
                [self.axs[s+1, (2*i+k)+1].set_xlabel('dim: ' + str(d0)) for k in [0,1]]
                [self.axs[s+1, (2*i+k)+1].set_ylabel('dim: ' + str(d1)) for k in [0,1]]




    def afterUpdate(self):
        print('after Update')
        fig, axs = sp.subplots(self.nMix + 1, 3, figsize=(10, 12))
        self.fig = fig
        self.axs = axs

        self.llPlot()
        self.responsibilityError()
        self.scatterPlot()
        self.displayPlots()
        # sp.show()

    def beforeUpdate(self, iter):
        if np.remainder(iter, 10) == 0:
            print('Iteration' + str(iter))
        return




    def displayPlots(self):
        for axs in self.axs.reshape(-1):
            #pdb.set_trace()
            axs.legend( )
        display(self.fig)
        sp.close( )
        for axs in self.axs.reshape(-1):
            axs.clear( )





# class Debug:
#
#     def __init__(self, nMix=2):
#         self.DB = [DBBase() for _ in range(nMix)]
#
#         #self.model = model
#
#
#     def attachData(self, data, DG):
#         self.X = data['X']
#         [db.attachData({'x':x}, dg) for (db, x, dg) in zip(self.DB, self.X, DG)]
#
#
#     def attachNets(self, autoEncoders):
#         self.autoEncoders = autoEncoders
#
#
#
#     def lossPlot(self, db, ae):
#         db.lossPlot(ae)
#
#
#     def reconstructionPlot(self, db, ae):
#         db.reconstructionPlot(ae)
#
#     def gaussianPlot(self, db, ae):
#         db.gaussianPlot(ae)
#
#
#     def scatterPlot(self, db, ae):
#         db.scatterPlot(ae)
#
#
#
#
#     def afterUpdate(self):
#         print('after Update')
#         #pdb.set_trace()
#         for i, (db, ae) in enumerate(zip(self.DB, self.autoEncoders[-1].gMixAE)):
#             print('Mixture:' + str(i))
#             db.afterUpdate(ae)
#         # AE = self.autoEncoders[-1]
#         # self.FIG = []
#         # self.AXS = []
#         # for (db, ae) in zip(self.DB, AE.gMixAE):
#         #     fig, axs = sp.subplots(self.nComps + 2, 4, figsize=(10, 12))
#         #     self.FIG.append(fig)
#         #     self.AXS.append(axs)
#         #     self.nComps = ae.nComps
#         #     self.lossPlot()
#         #     self.reconstructionPlot()
#         #     self.gaussianPlot()
#         #     self.scatterPlot()
#         #     self.displayPlots()
#         # sp.show()
#
#     def beforeUpdate(self, iter):
#         if np.remainder(iter, 10) == 0:
#             print('Iteration' + str(iter))
#         return
#
#     def beforeTraining(self, par):
#         print('before Training')
#         self.plotllHistory(par)
#         return
#
#
#     def displayPlots(self):
#         [db.displayPlots() for db in self.DB]
#
#
#         # for (Fig, Axs) in zip(self.FIG, self.AXS):
#         #     for axs in Axs.reshape(-1):
#         #         #pdb.set_trace()
#         #         axs.legend( )
#         #     display(Fig)
#         #     sp.close( )
#         # for Axs in self.AXS:
#         #     for axs in Axs.reshape(-1):
#         #         #pdb.set_trace()
#         #         axs.clear( )
#
