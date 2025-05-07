# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from MultiSampleMixtureData import GaussianMixtureDataGenerator
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

#from plots import ResposibilitiesError

import pdb

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
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
    X, Y, Responsibilities_true, sample2Component, Mu, Cov, Proportions \
    = GaussianMixtureDataGenerator(num_samples=num_samples, num_components=num_components, dim=input_dim)
    VisualizeInputData(X,Y)
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
   
