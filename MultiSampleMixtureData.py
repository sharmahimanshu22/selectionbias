from DataGen.data.randomParameters import NormalMixPNParameters2 as NMixPar
from scipy.stats import dirichlet
from DataGen.data.utils import AUCFromDistributions
from scipy.stats import dirichlet
import numpy as np
from DataGen.data.distributions import mixture
import math
import copy
from scipy.stats import multivariate_normal as mvn
import os
import json




class MultiSampleGaussianMixData:

    @staticmethod
    def load_from(dirname):
        i = 0
        X = []
        y = []
        samplefilenamex = "xarray_sample_" + str(i) + ".txt"
        samplefilenamey = "yarray_sample_" + str(i) + ".txt"
        while(os.path.exists(os.path.join(dirname,samplefilenamex))):
            X.append(np.loadtxt(os.path.join(dirname,samplefilenamex)))
            y.append(np.loadtxt(os.path.join(dirname,samplefilenamey)).reshape(-1, 1))
            i = i+1
            samplefilenamex = "xarray_sample_" + str(i) + ".txt"
            samplefilenamey = "yarray_sample_" + str(i) + ".txt"
        
        
        n_samples = i
        pos_mus = []
        neg_mus = []
        with open(os.path.join(dirname, "pos_mu.txt")) as f:
            c = f.read()
            c = c.strip().split("\n")
            for e in c:
                mu = np.array([float(a) for a in e.split(",")])
                pos_mus.append(mu)

        with open(os.path.join(dirname, "neg_mu.txt")) as f:
            c = f.read()
            c = c.strip().split("\n")
            for e in c:
                mu = np.array([float(a) for a in e.split(",")])
                neg_mus.append(mu)
            

       
        pos_covs = []
        neg_covs = []
        
        with open(os.path.join(dirname, "pos_cov.txt")) as f:
            pos_cov_strs = f.read().strip().split("\n\n")
            for ms in pos_cov_strs:
                pos_covs.append( np.array([[float(j) for j in e.split(',')] for e in ms.splitlines()]) )
            f.close()

        with open(os.path.join(dirname, "neg_cov.txt")) as f:
            neg_cov_strs = f.read().strip().split("\n\n")
            for ms in neg_cov_strs:
                neg_covs.append( np.array([[float(j) for j in e.split(',')] for e in ms.splitlines()]) )
            f.close()
        
        sample_to_Component_Idces_and_proportions_dict = None
        sample_to_Component_Idces = []
        component_mixProp_in_sample = []
        with open(os.path.join(dirname,"sample_to_component_indices_and_proportions.txt"), 'r') as file:
            sample_to_Component_Idces_and_proportions_dict = json.load(file)
            file.close()

        for i in range(n_samples):
            sample_to_Component_Idces.append((sample_to_Component_Idces_and_proportions_dict[str(i)]["pos_comp_indices"], 
                                         sample_to_Component_Idces_and_proportions_dict[str(i)]["neg_comp_indices"]))
            component_mixProp_in_sample.append((sample_to_Component_Idces_and_proportions_dict[str(i)]["pos_comp_proportion"], 
                                         sample_to_Component_Idces_and_proportions_dict[str(i)]["neg_comp_proportion"]))
        
        i = 0
        resps = []
        samplefilenameresp = "responsibilities_sample_" + str(i) + ".txt"
        while(os.path.exists(samplefilenameresp)):
            resps.append(np.loadtxt(samplefilenameresp))
            i = i+1
            samplefilenameresp = "responsibilities_sample_" + str(i) + ".txt"


        return MultiSampleGaussianMixData(X, y, resps, sample_to_Component_Idces, component_mixProp_in_sample, pos_mus, neg_mus, pos_covs, neg_covs)




    def save_this(self, dirname):
        os.makedirs(dirname, exist_ok=True)
        for i in range(len(self.X)):
            np.savetxt(os.path.join(dirname, "xarray_sample_" + str(i) + ".txt"),self.X[i])
        for i in range(len(self.y)):
            np.savetxt(os.path.join(dirname, "yarray_sample_" + str(i) + ".txt"),self.y[i])

        np.savetxt(os.path.join(dirname,"pos_mu.txt"), np.asarray(self.all_components_Mu[0]), delimiter=',', fmt='%.3f')
        np.savetxt(os.path.join(dirname,"neg_mu.txt"), np.asarray(self.all_components_Mu[1]), delimiter=',', fmt='%.3f')

        with open(os.path.join(dirname,"pos_cov.txt"), 'w') as f:
            for m in self.all_components_Cov[0]: # all positive cov matrices
                s = "\n".join([",".join([str(e) for e in l]) for l in m])
                f.write(s)
                f.write("\n\n")
            f.close()

        with open(os.path.join(dirname,"neg_cov.txt"), 'w') as f:
            for m in self.all_components_Cov[1]: # all positive cov matrices
                s = "\n".join([",".join([str(e) for e in l]) for l in m])
                f.write(s)
                f.write("\n\n")
            f.close()

        sample_to_Component_Idces_and_proportions_dict = {}
        for i in range(len(self.sample_to_Component_Idces)):
            info = {"pos_comp_indices":self.sample_to_Component_Idces[i][0],
                    "neg_comp_indices":self.sample_to_Component_Idces[i][1],
                    "pos_comp_proportion": self.sample_to_component_Proportions[i][0],
                    "neg_comp_proportion": self.sample_to_component_Proportions[i][1],}
            sample_to_Component_Idces_and_proportions_dict[i] = info
        with open(os.path.join(dirname, "sample_to_component_indices_and_proportions.txt"), 'w') as f:
            json.dump(sample_to_Component_Idces_and_proportions_dict, f)
            f.close()

        
        for i in range(len(self.Responsibilities)):
            np.savetxt(os.path.join(dirname + "responsibilities_sample_" + str(i) + ".txt"),self.Responsibilities[i])





        
    def concatenate_x_y(self, X, y):
        Xy = np.column_stack([X, y])
        return Xy
            

    def __init__(self, X, y, Responsibilities, sample_to_Component_Idces, sample_to_component_Proportions, pos_Mu, neg_Mu, pos_Cov, neg_Cov, ):

        self.X = X
        self.y = y
        print(X, y, "x and y")
        self.Responsibilities = Responsibilities     # only for components present in the sample
        self.sample_to_Component_Idces = sample_to_Component_Idces  # list of tuples. each element in list corresponds to sample. each tuple has two lists. First for positive components, second for negative components
        self.all_components_Mu = (pos_Mu, neg_Mu) # tuple of two lists. first element of tuple is list of positive Mu, second element of tuple is list of negative Mu
        self.all_components_Cov = (pos_Cov, neg_Cov) # tuple of two lists. first element of tuple is list of positive Cov, second element of tuple is list of negative Cov
        self.sample_to_component_Proportions = sample_to_component_Proportions # list of tuple. each element of list is a tuple of two list describing proportions of (only those) components present in the sample.  first element of tuple corresponds to positive components, second element of tuple cooresponds to negative components
        
        n_pos_comps = len(pos_Mu)
        self.sample_to_Component_Idces_Combined = []
        for e in self.sample_to_Component_Idces:
            v = copy.copy(e[0])   # these are positive components present
            for f in e[1]:  # these are the negative components present
                v.append(f + n_pos_comps)
            self.sample_to_Component_Idces_Combined.append(v)

    def dist_sample(self, sample_idx):
        all_pos_mu = self.all_components_Mu[0]
        all_neg_mu = self.all_components_Mu[1]
        all_pos_cov = self.all_components_Cov[0]
        all_neg_cov = self.all_components_Cov[1]
        sample_pos_mu = [all_pos_mu[i] for i in self.sample_to_Component_Idces[sample_idx][0]]
        sample_neg_mu = [all_neg_mu[i] for i in self.sample_to_Component_Idces[sample_idx][1]]
        sample_pos_cov = [all_pos_cov[i] for i in self.sample_to_Component_Idces[sample_idx][0]]
        sample_neg_cov = [all_neg_cov[i] for i in self.sample_to_Component_Idces[sample_idx][1]]
        sample_all_mu = sample_pos_mu + sample_neg_mu
        sample_all_cov = sample_pos_cov + sample_neg_cov


        sample_all_proportions = self.sample_to_component_Proportions[sample_idx]  # a tuple of two list
        sample_all_proportions = sample_all_proportions[0] + sample_all_proportions[1]

        assert (len(sample_all_mu) == len(sample_all_cov))
        assert (len(sample_all_mu) == len(sample_all_proportions))
        assert (math.isclose(sum(sample_all_proportions),1))

        comps = []

        for i in range(len(sample_all_mu)):
            comp = mvn(mean = sample_all_mu[i], cov = sample_all_cov[i])
            comps.append(comp)
        
        dist = mixture(comps, sample_all_proportions)
        return dist

    def responsibility(self, sample_idx, x):
        dist = self.dist_sample(sample_idx)
        component_resp = dist.responsibility(x)
        return component_resp    # responsibilities corresponding to componenents present in the sample
 

        



def GaussianMixtureDataGenerator2(num_samples, n_pos_comps, n_neg_comps , dim, sample_to_pos_comp_idces, sample_to_pos_comps_mix_prop, 
                                  sample_to_neg_comp_idces, sample_to_neg_comps_mix_prop, sample_sizes):
    
    assert (n_pos_comps == n_neg_comps) # For now we have to operate in these premises

    total_comps = n_pos_comps + n_neg_comps
    irr_vec = [0.01, 0.9, True]
    aucpn_range = [0.8, 0.85]
    
    NMix = NMixPar(dim, total_comps//2)
    NMix.perturb2SatisfyMetrics(aucpn_range, irr_vec)
    dg = NMix.dg
    p_comps = dg.dist_p.comps
    n_comps = dg.dist_n.comps

    all_samples = []
    all_sample_labels = []
    sample_to_component_Responsibilities = []
    component_mixProp_in_sample = []
    sample_to_Component_Idces = []

    pos_Mu = [component.mean for component in p_comps]
    neg_Mu = [component.mean for component in n_comps]
    pos_Cov = [component.cov for component in p_comps]
    neg_Cov = [component.cov for component in n_comps]

    

    for i in range(num_samples):
        s2pc = sample_to_pos_comp_idces[i]
        s2nc = sample_to_neg_comp_idces[i]

        sample_pos_comps = [p_comps[j] for j in s2pc]
        sample_neg_comps = [n_comps[j] for j in s2nc]
        sample_comps = sample_pos_comps + sample_neg_comps
        sample_comps_mix_prop = sample_to_pos_comps_mix_prop[i] + sample_to_neg_comps_mix_prop[i]

        assert (math.isclose( sum(sample_comps_mix_prop) , 1 ) )

        dist = mixture(sample_comps, sample_comps_mix_prop)
        x,y = dist.rvsCompInfo(sample_sizes[i])

        responsibility = dist.responsibility(x)
        all_samples.append(x)
        all_sample_labels.append(y)
        sample_to_component_Responsibilities.append(responsibility)
        component_mixProp_in_sample.append((sample_to_pos_comps_mix_prop[i], sample_to_neg_comps_mix_prop[i]))
        sample_to_Component_Idces.append( (s2pc, s2nc) )
        
    
    return MultiSampleGaussianMixData(all_samples, all_sample_labels, sample_to_component_Responsibilities, sample_to_Component_Idces,
                                      component_mixProp_in_sample, pos_Mu, neg_Mu, pos_Cov, neg_Cov)
        




    
def GaussianMixtureDataGenerator(num_samples=1, num_components=2, dim=2):
    #Dimension
    dim = 2
    #Number of components in positves and negatives each
    num_comps = 2
    # Setting the thrid entry of irr_vec to False enforces pairwise (between pair of any two compnents)
    # mutual irreducibility
    # setting the thrid entry of irr_vec to True enforces the strong irreducibility, where each 
    # component is irreducible w.r.t. all other components considered together.
    # The second entry of irr_vec is the the balanced posterior (responsibility) threshold for irreducibility.
    # The first argument is the proportion of points that ought to satisfy the posterior threshold
    # for irreducibility to hold true. Using 0.01 and 0.9 is easier to satisfy then 0.05 and 0.95. 
    # You may play with this, but note that the stronger irreducibility criteria, the more difficult 
    # it is to be satisfied and you might end up in infinite loop since no parameters could be 
    # found to satisfy both the irreducibility and auc criteria.
    irr_vec = [0.01, 0.9, True]
    # aucpn_range contains the desired range of AUCPN to be satisfied between the paired positive 
    # and negative component.
    # Once the constraints are satisfied between the paired components. The paired components are
    # moved together further or close to the other paired components that the AUC constraints can be satisfied between unpaired components
    # to the extent possible.
    #The minimum AUC range for 1 dimensional datasets is [0.7, 0.75].
    #If the AUC range is smaller than that irreducibility can't be achieved when using 
    #irr_vec = [0.01, 0.9, False]
    aucpn_range = [0.8, 0.85]
    #Dataset size
    n = 20000
    #proportion of positives
    alpha = 0.5
    NMix = NMixPar(dim, num_comps//2)
    NMix.perturb2SatisfyMetrics(aucpn_range, irr_vec)
    dg = NMix.dg
    comps = dg.dist_p.comps + dg.dist_n.comps
    dist = mixture(comps, [1/num_comps for i in range(num_comps)])
    X = []
    Y = []
    Responsibilities = []
    Proportions = []
    Mu = [component.mean for component in dist.comps]
    Cov = [component.cov for component in dist.comps]
    sample2Component = []
    dirichlet_alpha = np.array([1/num_comps for i in range(num_comps)])

    for i in range(num_samples):
        prop = dirichlet.rvs(alpha = dirichlet_alpha, size=1)[0]
        dist.mixProp=prop
        dist.mixProp[0]= 0.2
        dist.mixProp[1]= 0.8
        x,y = dist.rvsCompInfo(n)
        responsibility = dist.responsibility(x)
        X.append(x)
        Y.append(y)
        Responsibilities.append(responsibility)
        Proportions.append(prop)
        s2c = [i for i in range(num_comps)]
        sample2Component.append(s2c)
    return X, Y, Responsibilities, sample2Component, Mu, Cov, Proportions




def compute_true_resp(Mu, Cov, Proportions, sample):

    assert (len(Mu) == len(Cov))
    assert (len(Proportions) == len(Mu))

    allcomps = []
    for i in range(len(Mu)):
        comp_i = mvn(mean = Mu[i], cov = Cov[i])
        allcomps.append(comp_i)
    
    dist = mixture(allcomps, Proportions)

    true_resps = []
    for i in range(len(Mu)):
        comp_resp = dist.component_posterior(i, sample)
        true_resps.append(comp_resp)

    return true_resps