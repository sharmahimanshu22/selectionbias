import pdb

import numpy as np
from scipy.stats import norm
from scipy.stats import mvn
from sklearn.cluster import KMeans
from GMM.compModel import MVN
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from scipy.special import logsumexp
from GMM.distributions import mixture
from misc.dictUtils import safeUpdate
from GMM.utils import ellipse_radius_along_vector
import itertools


class GMM:
    def __init__(self, nComps, dim, maxIter=1000, compDist=None, nMix=1, cMemPerSample=None, equalPropComps=False, identicalCov=False):
        self.nComps = nComps
        self.dim = dim
        self.maxIter = maxIter
        self.iter = 0
        self.nMix = nMix
        #pdb.set_trace()
        if cMemPerSample is None:
            self.cMemPerSample = [np.arange(nComps) for _ in range(nMix)]
        else:
            assert len(cMemPerSample) == nMix
            self.cMemPerSample = cMemPerSample
        self.cMem2SMem(self.cMemPerSample)
        self.mProp = [np.repeat(1/np.size(cMemS), np.size(cMemS)) for cMemS in self.cMemPerSample]
        if compDist is None:
            self.compDist = [MVN(dim=dim, spherical=False) for _ in range(nComps)]
        else:
            self.compDist = compDist
        self.mixture = [mixture([self.compDist[i] for i in cMemS], mp) for (cMemS, mp)
                        in zip(self.cMemPerSample, self.mProp)]
        self.lls = []
        self.initParRan = False
        self.equalPropComps = equalPropComps
        self.identicalCov = identicalCov


    
    def attachDebugger(self, debug):
        self.debug = debug

    # si : sample index
    # ci : component index
    def comp_posterior(self, si, ci, x):
        mix = self.mixture[si]
        return mix.component_posterior(ci,x)
        
        

    def cMem2SMem(self, cMemPerSample):
        self.sMemPerComp = [[] for _ in range(self.nComps)]
        self.cIndPerComp = [[] for _ in range(self.nComps)]
        # Iterate over the membership variable and update the component membership and indexes variables
        for i, mix in enumerate(cMemPerSample):
            for j, component in enumerate(mix):
                self.sMemPerComp[component].append(i)    # component to sample indices
                self.cIndPerComp[component].append(j)    # looks so useless and meaningless

    def separateData(self, X):
        Fits = [KMeans(n_clusters=len(gci_mix), init='k-means++').fit(x) for (x, gci_mix) in zip(X, self.cMemPerSample)]
        Labels = [fit.labels_ for fit in Fits]
        Centers =[fit.cluster_centers_ for fit in Fits]

        # those component index for which a sample containing it has been processed are stored in comps
        gci_processed = np.array([], dtype='int64')
        centers = np.ones((self.nComps,self.dim))*np.inf
        counts = np.zeros(self.nComps)
        C = []
        for (l, mu, gci_mix) in zip(Labels, Centers, self.cMemPerSample):
            #by the end of the iteration newl will contain updated labels, such that some of the labels are uniquely
            # mapped to the components the sample is supposed to contain and is already present in comps by iteratively
            # finding the closest cluster - component pair. The remaining clusters are given a label corresponding to the
            # component that the sample is suppose to have, but not yet seen in the data processed so far.
            c = np.copy(l)
            # contains global index of the components already processed and also contained in the current sample
            gci_existing = np.intersect1d(gci_mix, gci_processed)
            # contains global index of the unprocessed components contained in the current sample
            gci_new = np.setdiff1d(gci_mix, gci_existing)
            # keeps track of the local component index mapped to already processed components
            lci_assigned = np.array([])
            if gci_existing.shape[0] > 0:
                dist = pairwise_distances(mu, centers)
                #pdb.set_trace()
                dist[:, np.setdiff1d(gci_processed, gci_existing)] = np.inf
                while np.any(np.isfinite(dist)):
                    ix = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
                    lci = ix[0]
                    gci = ix[1]
                    c[l==lci] = gci
                    dist[:, gci] = np.inf
                    dist[lci, :] = np.inf
                    oldCenter = centers[gci,:]
                    oldCount = counts[gci]
                    newCount = oldCount + np.sum(l==lci)
                    newCenter = (oldCenter*oldCount + mu[lci,:] * np.sum(l==lci))/newCount
                    centers[gci,:] = newCenter
                    counts[gci] = newCount
                    lci_assigned = np.hstack((lci_assigned, lci))
            lci_unassigned = np.setdiff1d(np.arange(len(gci_mix)), lci_assigned)
            for (lci, gci) in zip(lci_unassigned, gci_new):
                c[l == lci] = gci
                #pdb.set_trace()
                centers[gci,:] = mu[lci,:]
                counts[gci] = np.sum(l==lci)
                gci_processed = np.hstack((gci_processed, gci))
            C.append(c)
        return X, C

    def initPar(self, X):
        X, C = self.separateData(X)
        #pdb.set_trace()
        [cDist.fit(X, [(c == i).astype('int32') for c in C]) for i, cDist in enumerate(self.compDist)]
        self.mProp = [np.array([np.sum((c == i).astype('int32'))/c.shape[0] for i in cMemS]) for (cMemS, c) in zip(self.cMemPerSample, C)]
        self.initParRan = True
        #pdb.set_trace()

    def refit(self, X, maxIter=None):
        self.iter = 0
        self.fit(X, maxIter)

    def fit(self, X, maxIter=None):
        #if not self.initParRan or self.dataOOD(X):
        if not self.initParRan:
            self.initPar(X)
        if maxIter is None:
            maxIter = self.maxIter
        while self.iter < maxIter:
            #self.beforeUpdate()
            self.run(X)
            #self.afterUpdate()
            self.iter+=1

    def dataOOD(self, X):
        maxDensPerCmp = np.array([np.max(np.vstack([cmp.pdf(x) for x in  X])) for cmp in self.compDist])
        max = np.max(maxDensPerCmp)
        min = np.min(maxDensPerCmp)
        oodFlag = min/max < 10**-3
        self.oddFlag = oodFlag
        if self.oddFlag:
            print('OOD input to GMM: Reinitializing GMM parameters')
        return oodFlag


    def run(self, X):
        R = self.responsibility(X)
        if self.equalPropComps:
            R = self.adjustResponsibility(R)
        self.mProp = [np.array([np.sum(r)/x.shape[0] for r in rr.T]) for (rr,x) in zip(R,X)]
        [cDist.fit([X[s] for s in sMem], [R[s][:,c] for (s,c) in zip(sMem, cInd)]) for (cDist, sMem, cInd) in
         zip(self.compDist, self.sMemPerComp, self.cIndPerComp)]
        if self.identicalCov:
            cov = self.__updateCovs__()
            self.__separateMeans__(cov)


    def __updateCovs__(self):
        covs = [cDist.cov for cDist in self.compDist]
        cov = np.mean(covs, axis=0)
        for cdist in self.compDist:
            cdist.cov = cov
        return cov

    def __separateMeans__(self, cov):
        means = np.vstack([cDist.mu for cDist in self.compDist])
        for i, j in itertools.combinations(range(means.shape[0]), 2):        
            diff=means[i] - means[j]
            length = np.linalg.norm(diff)
            diff = diff/length
            r = ellipse_radius_along_vector(cov, n_std=1.0, v=diff)
            if  length < r/4:
                displacement_vec = (1/8)*r*diff
                means[i] = means[i] + displacement_vec
                means[j] = means[j] - displacement_vec 
        for i, mean in enumerate(means):
            self.compDist[i].mu = mean


    def adjustResponsibility(self, Resposnibilities):
        Resposnibilities = [self.__adjustResponsibility__(resps) for resps in Resposnibilities]
        return Resposnibilities
       
    def __adjustResponsibility__(self, responsibilities):
        avgResps = [np.mean(resp, axis=0) for resp in responsibilities.T]
        expectedAvg = 1/responsibilities.shape[1]
        deltaAvgResps = [avgResp - expectedAvg for  avgResp in avgResps]
        excessAvgResps = [max(delAResps,0) for delAResps in deltaAvgResps]
        #deficientAvgResps = [-min(delAResps,0) for delAResps in deltaAvgResps]
        excessResp = np.sum(np.hstack([(EAResp/AResp)*resp.reshape(-1,1) for (EAResp, AResp, resp) 
                              in zip(excessAvgResps, avgResps, responsibilities.T)]), axis=1, keepdims=True)
        excessAvgResp = np.mean(excessResp, axis=0)
        responsibilities_adj = np.hstack([resp.reshape(-1,1) - (delAResp/excessAvgResp)*excessResp 
                            for (delAResp, resp) in zip(deltaAvgResps, responsibilities.T)])
        assert np.allclose(np.sum(responsibilities_adj, axis=1), 1)
        assert np.allclose(np.mean(responsibilities_adj, axis=0), expectedAvg)
        return responsibilities_adj
        
        
    def responsibility(self, X, sample_index=None):
        if sample_index is None:
            R = [self.__responsibility__(x, i)[0] for i, x in enumerate(X)]
            return R
        else:
            return self.__responsibility__(X, sample_index)[0]

    def __responsibility__(self, x, sample_ix):
        logCPdf = np.hstack([self.compDist[j].logpdf(x)[:, None] for j in self.cMemPerSample[sample_ix]])
        logCPdf_w = logCPdf + np.log(self.mProp[sample_ix])
        logMPdf = logsumexp(logCPdf_w, axis=1, keepdims=True)
        R = np.exp(logCPdf_w - logMPdf)
        return R, logMPdf


    def logLikelihood(self, X, equallyWeightedSamples=False):
        logMPdf = [self.__responsibility__(x, i)[1] for i, x in enumerate(X)]
        mixLL = [np.mean(lMPdf) for lMPdf in logMPdf]
        if not equallyWeightedSamples:
            ss = [x.shape[0] for x in X]
            ll = sum([s*mLL for(s, mLL) in zip(ss, mixLL)])/sum([s for s in ss])
        else:
            ll = sum([mLL for(mLL) in zip(mixLL)])
        return ll

    def copy(self):
        compDist = [cDist.copy() for cDist in self.compDist]
        gmm = GMM(self.nComps, self.dim, self.maxIter, compDist, self.nMix, self.cMemPerSample)
        gmm.mProp = np.copy(self.mProp)
        gmm.initParRan = self.initParRan
        return gmm

    def attachDebugger(self, debug):
        self.debug = debug
        self.debug.attachGMM(self)

    def beforeUpdate(self):
        if hasattr(self, 'debug'):
            self.debug.beforeUpdate(self.iter)

    def afterUpdate(self):
        if hasattr(self, 'debug'):
            self.debug.afterUpdate()

