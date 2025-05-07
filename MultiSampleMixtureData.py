from DataGen.data.randomParameters import NormalMixPNParameters2 as NMixPar
from scipy.stats import dirichlet
from DataGen.data.utils import AUCFromDistributions
from scipy.stats import dirichlet
import numpy as np
from DataGen.data.distributions import mixture


    
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



# def transform(x):
