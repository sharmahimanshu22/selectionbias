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
import argparse
from configurations import *
from plots import *


def get_argparer():
    parser = argparse.ArgumentParser()
    parser.add_argument('--usefrozen', action='store_true', help='Use stored data')
    parser.add_argument("-dd", "--datadir",type=str, help="directory to load data from")
    parser.add_argument("-sd", "--storedir", type=str, help="directory to save data in")
    parser.add_argument("-ld", "--logdir", type=str, help="directory to log data in")
    return parser





def get_our_input_data(context):
    if context.frozendata:
        return MultiSampleGaussianMixData.load_from(context.loaddir)

    msgmd = GaussianMixtureDataGenerator2(context.n_samples, context.n_pos_comps, context.n_neg_comps, 
                                         context.input_dim, context.sample_to_pos_comp_idces, context.sample_to_pos_comps_mix_prop, 
                                         context.sample_to_neg_comp_idces, context.sample_to_neg_comps_mix_prop, context.sample_sizes)
    msgmd.save_this(context.savedir)
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
    args = get_argparer().parse_args()
    context = get_input_context_one_sample_two_comps(args)
    hp = get_hyperparameters(context)
    writer = SummaryWriter(args.logdir)

    msgmd = get_our_input_data(context)   # data
    visualizeinput(msgmd, writer)
    
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
