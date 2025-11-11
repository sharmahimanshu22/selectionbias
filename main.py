# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from MultiSampleMixtureData import GaussianMixtureDataGenerator, MultiSampleGaussianMixData, GaussianMixtureDataGenerator2
from trainer import trainer
from models import FullyConnectedAutoencoder as Model
from torch import optim
import numpy as np
import torch
import copy
from loss import GaussianMixAutoEncoderLoss
from gaussianMixLatentSpace import GaussianMixLatentSpace
from torch.utils.tensorboard import SummaryWriter
from plots import VisualizeInputData
import matplotlib.pyplot as plt
import io
from PIL import Image
from scipy.stats import multivariate_normal as mvn
#from plots import ResposibilitiesError
from DataGen.data.distributions import mixture
import pandas as pd

import pdb
import os
import math
from DataGen.plots.CIEllipse import CIEllipse
import json
import pickle
import sys





def transform3_single(e,f):
    return [math.exp(e[0])*math.cos(e[1]), math.exp(e[0])*math.sin(e[1])]

def transform3(X , y):
    z = np.array([e[0] for e in y])
    Xt = np.array([ transform3_single(e,f) for e,f in zip(X, z)])
    return Xt

def transform2_single(e, f):
    if f == 0:
        return [e[0] + e[1],e[0] - e[1]] 
    if f == 1:
        return [e[0] + e[1],e[0] - e[1]] 

def transform2(X , y):
    z = np.array([e[0] for e in y])
    Xt = np.array([ transform2_single(e,f) for e,f in zip(X, z)])
    return Xt

def transform1_single(e, f):
    if f == 0:
        return [e[0]*e[0] + e[1]*e[1],e[0]*e[0] - e[1]*e[1]] 
    if f == 1:
        return [e[0]*e[0] + e[1]*e[1],e[0]*e[0] - e[1]*e[1]] 

def transform1(X , y):
    z = np.array([e[0] for e in y])
    Xt = np.array([ transform1_single(e,f) for e,f in zip(X, z)])
    return Xt
    

def savedata(dirname, X, Y, Responsibilities_true, sample2Component, Mu, Cov, Proportions):
    os.makedirs(dirname, exist_ok=True)
    np.savetxt(dirname + "/xarray.txt",X[0])
    np.savetxt(dirname + "/yarray.txt", Y[0], fmt="%d")
    np.savetxt(dirname + "/responsibilities.txt", Responsibilities_true[0])
    np.savetxt(dirname + "/sample2comp.txt", sample2Component, fmt="%d")
    np.savetxt(dirname + "/mu.txt" , Mu)
    np.savetxt(dirname + "/cov1.txt", Cov[0])
    np.savetxt(dirname + "/cov2.txt", Cov[1])
    np.savetxt(dirname + "/proportions.txt", Proportions)
    
def get_fixed_data(dirname = "fixeddata"):
    X = [np.loadtxt(dirname +"/xarray.txt")]
    Y = [np.loadtxt(dirname + "/yarray.txt", dtype=int)]
    Y = [np.array([ [e] for e in Y[0]])] 
    Responsibilities_true = [np.loadtxt(dirname + "/responsibilities.txt")]
    sample2Component = [np.loadtxt(dirname + "/sample2comp.txt", dtype=int)]
    Mu = np.loadtxt(dirname+"/mu.txt")
    Cov1 = np.loadtxt(dirname+"/cov1.txt")
    Cov2 = np.loadtxt(dirname+"/cov2.txt")
    Cov = [Cov1, Cov2]
    Proportions = np.loadtxt(dirname+"/proportions.txt")
    return X, Y, Responsibilities_true, sample2Component, Mu, Cov, Proportions

def visualizeinput(X,y):

    
    for i, (x,y) in enumerate(zip(X, Y)):
        if len(X) == 2:
            plt.subplot(2,1,i)
        if len(X) == 3:
            plt.subplot(2,2,i)
        plt.title('Sample '+str(i))
        for c in np.unique(y):
            xx= x[(y==c).flatten()]
            ix = np.random.choice(xx.shape[0], 100, replace=True)
            plt.scatter(xx[ix, 0], xx[ix, 1], label='C'+str(c), alpha=0.5) 
            cov = np.cov(xx.T)
            mean = np.mean(xx, axis=0)
            CIEllipse(mean, cov, plt.gca(), n_std=1.0, facecolor='none', edgecolor='k')
    plt.legend()
    
    '''
    z = np.array([e[0] for e in y])
    #print("Zhere: ", z)
    comp1data = X[z==0][:200]
    comp2data = X[z==1][:200]
    #print("compdatalen")
    #print(len(comp1data))
    #print(len(comp2data))
    plt.scatter(comp1data[:,0], comp1data[:,1], color='r', s = 2)
    plt.scatter(comp2data[:,0], comp2data[:,1], color='b', s = 2)
    '''
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    # Convert the buffer to a PIL Image
    image = Image.open(buf)
    # Convert the PIL Image to a NumPy array
    image = np.array(image)
    # Convert the NumPy array to a PyTorch tensor
    image = torch.tensor(image).permute(2, 0, 1)  # Change the order of dimensions to [C, H, W]
    return image




class HyperParameters:
    def __init__(self, sample_sizes, test_size, batch_size, gmls_sigma, encoded_dim_autoencoder, num_layers_autoencoder, width_autoencoder, 
                 learning_rate, n_epochs,warmup_epochs):
        self.sample_sizes_training = [e*(1-test_size) for e in sample_sizes] # a hyper parameter dependent on input sample sizes. isn't that bad ?
        self.batch_size = batch_size
        self.gmls_sigma = gmls_sigma
        self.encoded_dim_autoencoder = encoded_dim_autoencoder
        self.num_layers_autoencoder = num_layers_autoencoder
        self.width_autoencoder = width_autoencoder
        self.test_size = test_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.warmup_epochs = warmup_epochs

        

class Context:
    def __init__(self, n_samples, n_pos_compos, n_neg_comps, dim, sample_to_pos_comp_idces,sample_to_pos_comps_mix_prop,
                 sample_to_neg_comp_idces,sample_to_neg_comps_mix_prop, sample_sizes, frozendata):
        self.n_samples = n_samples
        self.n_pos_comps = n_pos_compos
        self.n_neg_comps = n_neg_comps
        self.input_dim = dim
        self.sample_to_pos_comp_idces = sample_to_pos_comp_idces
        self.sample_to_pos_comps_mix_prop = sample_to_pos_comps_mix_prop
        self.sample_to_neg_comp_idces = sample_to_neg_comp_idces
        self.sample_to_neg_comps_mix_prop = sample_to_neg_comps_mix_prop
        self.sample_sizes = sample_sizes
        self.frozendata = frozendata



def get_hyperparameters(input_context):
    sample_sizes = input_context.sample_sizes
    test_size = 0.2
    batch_size = 100
    gmls_sigma = 0.1
    encoded_dim_autoencoder = 2
    num_layers_autoencoder = 10
    width_autoencoder = 10
    learning_rate = 0.001
    n_epochs = 100
    warmup_epochs = 20
    return HyperParameters(sample_sizes,test_size, batch_size,  gmls_sigma, encoded_dim_autoencoder, num_layers_autoencoder, width_autoencoder, learning_rate,
                           n_epochs, warmup_epochs)


def get_input_context():   
    frozendata = True 
    n_samples = 1
    n_pos_comps = 1
    n_neg_comps = 1
    input_dim = 2
    sample_to_pos_comp_idces = [[0]]
    sample_to_pos_comps_mix_prop = [[0.8]]
    sample_to_neg_comp_idces = [[0]]
    sample_to_neg_comps_mix_prop = [[0.2]]
    sample_sizes = [20000]
    return Context(n_samples, n_pos_comps,n_neg_comps, input_dim, sample_to_pos_comp_idces, 
                   sample_to_pos_comps_mix_prop, sample_to_neg_comp_idces, sample_to_neg_comps_mix_prop, sample_sizes, frozendata)


def get_our_input_data(context):
    if context.frozendata:
        return MultiSampleGaussianMixData.load_from("checksave")

    msgmd = GaussianMixtureDataGenerator2(context.n_samples, context.n_pos_comps, context.n_neg_comps, 
                                         context.input_dim, context.sample_to_pos_comp_idces, context.sample_to_pos_comps_mix_prop, 
                                         context.sample_to_neg_comp_idces, context.sample_to_neg_comps_mix_prop, context.sample_sizes)
    msgmd.save_this("checksave")
    return msgmd

def initialize_our_gaussian_mix_latent_space(context: Context, hp : HyperParameters):
    Mu0 = [np.ones((context.input_dim,)) + np.random.normal(size=(context.input_dim,))*0.1 for _ in range(context.n_pos_comps + context.n_neg_comps)]
    Cov0 = [np.eye(context.input_dim) for _ in range(context.n_pos_comps + context.n_neg_comps)]
    sample_to_ncomps = [ len(e) + len(f) for e,f in zip(context.sample_to_pos_comp_idces, context.sample_to_neg_comp_idces)]
    Proportions0 = [np.ones(e) / e for e in sample_to_ncomps]
    
    sample_to_Component_Idces_Combined = []
    for i in range(context.n_samples):
        p = copy.copy(context.sample_to_pos_comp_idces[i])
        n = copy.copy(context.sample_to_neg_comp_idces[i])
        n = [e + context.n_pos_comps for e in n]
        v = p + n # concatenate lists
        sample_to_Component_Idces_Combined.append(v)

    # We might need to make a separte context for hyper parameters of our models and latent space 

    return GaussianMixLatentSpace(Mu0, Cov0, Sample2Component=sample_to_Component_Idces_Combined, 
                                  Proportions=Proportions0, Sample_Sizes=hp.sample_sizes_training, 
                                  points_per_component=hp.batch_size, sigma=hp.gmls_sigma)

def main():
    context = get_input_context()
    hp = get_hyperparameters(context)
    writer = SummaryWriter()

    msgmd = get_our_input_data(context)   # data
    
    gaussianMixLatentSpace = initialize_our_gaussian_mix_latent_space(context, hp) # gaussian mix latent space
    model = Model(input_dim=context.input_dim, encoded_dim=hp.encoded_dim_autoencoder, width=hp.width_autoencoder, num_layers=hp.num_layers_autoencoder)
    parameters = list(model.parameters()) + list(gaussianMixLatentSpace.parameters())
    optimizer = optim.Adam(parameters, lr=hp.learning_rate)
    loss_func = GaussianMixAutoEncoderLoss(gamma=1.0, eta=0.0)

    best_state, last_state = trainer(msgmd, model, gaussianMixLatentSpace, loss_func, optimizer, 
                                     test_size=hp.test_size, batch_size=hp.batch_size, epochs=hp.n_epochs, 
                                     warmup_epochs= hp.warmup_epochs, writer=writer)




    sys.exit(0)




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
     # Training parameters
    input_dim = 2
    num_components = 2
    num_samples = 1
    test_size = 0.2
    encoded_dim = 2
    num_layers = 10
    width = 10
    learning_rate = 0.001
    batch_size = 100
    epochs = 150
    warmup_epochs = 20
    sigma = 0.1
    points_per_component = batch_size
    writer = SummaryWriter()
    
    #pdb.set_trace()
    # Currently the Gaussian GaussianMixtureDataGenerator assumes that all samples 
    # have the same components. This is fine for now. But when moving to P, N and U data
    # this will need to change.

    usefixed = True
    
    if usefixed:
        X, Y, Responsibilities_true, sample2Component, Mu, Cov, Proportions = get_fixed_data("frozendata")
    else:
        X, Y, Responsibilities_true, sample2Component, Mu, Cov, Proportions, \
            = GaussianMixtureDataGenerator(num_samples=num_samples, num_components=num_components, dim=input_dim)
        #savedata("fixeddata", X, Y, Responsibilities_true, sample2Component, Mu, Cov, Proportions)
        #print(sample2Component)


    gmd = GaussianMixData(X, Y, Responsibilities_true, sample2Component, Mu, Cov, Proportions)

    

    print(true_resps)

    sys.exit(0)
    




    writer.add_image("InitialInput", visualizeinput(X, Y))
    writer.flush()
    
    #df = pd.dataFrame("Xx", "Xy", "y", "resp_true_1", "resp_true_2" , "sample2comp", "mu", "cov", "Proportions")
    
    # x^2 + y^2, x^2 - y^2
    X = [transform3(X[0], Y[0])]

    #X = [ np.array([  [e[0]*e[0] + e[1]*e[1],e[0]*e[0] - e[1]*e[1]] for e in X[0]  ]) ]
    #Xt = [transform(X[0], Y[0])]

    
    writer.add_image("TransformedInput", visualizeinput(X, Y))
    writer.flush()

    #VisualizeInputData(X,Y)
    print("did visualization")
    Sample_Sizes = [x.shape[0]*(1-test_size) for x in X]
    Proportions0 = copy.deepcopy(Proportions)
    # Mu0 = copy.deepcopy(Mu)
    # Cov0 = copy.deepcopy(Cov)
    Mu0 = [np.ones((input_dim,)) + np.random.normal(size=(input_dim,))*0.1 for _ in range(num_components)]
    Cov0 = [np.eye(input_dim) for _ in range(num_components)]
    #Uncomment the line below to learn the proportions
    Proportions0 = [torch.ones(len(s2c)) / len(s2c) for s2c in sample2Component]
    ### Need to implement initialization of Mu and Cov. Perhaps use the 
    ### model outputs on the entire dataset. Do K-means clustering or GMM on it to learn 
    ### mus and covs.
    gaussianMixLatentSpace = GaussianMixLatentSpace(Mu0, Cov0, Sample2Component=sample2Component, 
                                                    Proportions=Proportions0, Sample_Sizes=Sample_Sizes, 
                                                    points_per_component=batch_size,sigma=sigma)
    # Initialize model, loss function, and optimizer
    model = Model(input_dim=input_dim, encoded_dim=encoded_dim, width=width, num_layers=num_layers)
    parameters = list(model.parameters()) + list(gaussianMixLatentSpace.parameters())
    optimizer = optim.Adam(parameters, lr=learning_rate)
    loss_func = GaussianMixAutoEncoderLoss(gamma=1.0, eta=0.0)
    best_state, last_state = trainer(X, model, gaussianMixLatentSpace, loss_func, optimizer, 
                                     test_size=test_size, batch_size=batch_size, epochs=epochs, 
                                     warmup_epochs= warmup_epochs, writer=writer, Y=Y)
    exit


    model.load_state_dict(best_state['model'])
    gaussianMixLatentSpace.load_state_dict(best_state['gaussianMixLatentSpace'])
    with torch.no_grad():
        Component_Responsibilities = [gaussianMixLatentSpace.compute_responsibilities(
                model(torch.tensor(x, dtype=torch.float32))[0],sample_index=sample_index)[1] 
                for sample_index, x in enumerate(X)]
    # matchedComponents = gaussianMixLatentSpace.matchComponents(x, resp, ) 
    # writer.add_image('Responsibilities Error', 
    #                 ResposibilitiesError(Responsibilities_true, Component_Responsibilities), epochs)
    # [print('Proportions True')]
    writer.close()

    pred_post = last_state['gaussianMixLatentSpace']

    comp1true = mvn(mean = Mu[0], cov = Cov[0])
    comp2true = mvn(mean = Mu[1], cov = Cov[1])
    mix_prop_true = Proportions
    dist = mixture([comp1true, comp2true], mix_prop_true)
    
    comp1 = mvn(mean=pred_post['Mu.0'], cov=pred_post['Cov.0'])
    comp2 = mvn(mean=pred_post['Mu.1'], cov=pred_post['Cov.1'])
    mix_proportions_best_state = pred_post['Proportions.0']
    dist2 = mixture([comp1, comp2], mix_proportions_best_state)


    mix_proportions_gmm = torch.tensor(best_state['gmm']['mProp'][0])
    
    
    true_comp1 = dist.component_posterior(0,X[0])
    true_comp2 = dist.component_posterior(1,X[0])
    pred_comp1 = dist2.component_posterior(0, model(torch.tensor(X[0], dtype=torch.float32))[0].detach().numpy())
    pred_comp2 = dist2.component_posterior(1, model(torch.tensor(X[0], dtype=torch.float32))[0].detach().numpy())

    _, component_responsibilities_mix_proportions_best_state, _ = gaussianMixLatentSpace.compute_responsibilities(
        model(torch.tensor(X[0], dtype=torch.float32))[0], proportion = mix_proportions_best_state)

    component_responsibilities_mix_proportions_best_state_comp1 = component_responsibilities_mix_proportions_best_state[0].detach().numpy().flatten()
    component_responsibilities_mix_proportions_best_state_comp2 = component_responsibilities_mix_proportions_best_state[1].detach().numpy().flatten()
    
    _, component_responsibilities_mix_proportions_gmm, _ = gaussianMixLatentSpace.compute_responsibilities(
        model(torch.tensor(X[0], dtype=torch.float32))[0], proportion = mix_proportions_gmm)
    component_responsibilities_mix_proportions_gmm_comp1 = component_responsibilities_mix_proportions_gmm[0].detach().numpy().flatten()
    component_responsibilities_mix_proportions_gmm_comp2 = component_responsibilities_mix_proportions_gmm[1].detach().numpy().flatten()

    
    #print(component_responsibilities_mix_proportions_best_state_comp1, " \ncomponent_responsibilities_mix_proportions_best_state")
    pars = {"Mu0": Mu[0], "Mu1": Mu[1], "Cov0": Cov[0],  "Cov1": Cov[1], "ProportionsTrue":  mix_prop_true, "mix_proportions_best_state" : mix_proportions_best_state , "mix_proportions_gmm" : mix_proportions_gmm}

    with open('saved_dictionary.pkl', 'wb') as f:
        pickle.dump(pars, f)
    
    
    data = {'truecomp1': true_comp1, 'truecomp2': true_comp2, 'predcomp1': pred_comp1, 'predcomp2': pred_comp2,
            'component_responsibilities_mix_proportions_best_state_comp1':component_responsibilities_mix_proportions_best_state_comp1,
            'component_responsibilities_mix_proportions_best_state_comp2':component_responsibilities_mix_proportions_best_state_comp2,
            'component_responsibilities_mix_proportions_gmm_comp1' : component_responsibilities_mix_proportions_gmm_comp1,
            'component_responsibilities_mix_proportions_gmm_comp2' : component_responsibilities_mix_proportions_gmm_comp2
            }
    df = pd.DataFrame(data)
    df.to_csv("posteriors.txt", index=False)
