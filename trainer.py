from models import FullyConnectedAutoencoder as Model
from loss import GaussianMixAutoEncoderLoss
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler
from plots import VisualizeEmbeddings
#from plots import VisualizeGMMFit
import copy

import torch
import numpy as np

def trainer(X, model, gaussianMixLatentSpace, loss_func,  optimizer, test_size=0.2, 
batch_size=0.2, epochs=1000, warmup_epochs = 20, writer=None, Y = None):
    #input_dim = X[0].shape[1]
    if type(batch_size) == float:
        batch_size = int(batch_size*X.shape[0])
    if test_size > 0.0:
        Testing = True
    else:
        Testing = False
    #pdb.set_trace()
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    if Testing:
        for x,y in zip(X,Y):
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
            X_train.append(x_train)
            X_test.append(x_test)
            Y_train.append(y_train)
            Y_test.append(y_test)
    else:
        X_train = X
        X_test = X
        Y_train = Y
        Y_test = Y

    X_train = [torch.tensor(x_train, dtype=torch.float) for x_train in X_train]
    X_test = [torch.tensor(x_test, dtype=torch.float) for x_test in X_test]
    # Y_train = [torch.tensor(y_train, dtype=torch.float) for y_train in Y_train]
    # Y_test = [torch.tensor(y_test, dtype=torch.float) for y_test in Y_test]

    Sample2Component = gaussianMixLatentSpace.Sample2Component

    max_data_size = max([x_train.shape[0] for x_train in X_train])  # Maximum number of data points in a dataset
    iterations_per_epoch = max_data_size // batch_size  # Number of iterations per epoch

    #Train_dataset = [TensorDataset(torch.tensor(x_train, dtype=torch.float)) for x_train in X_train]
    # Train_loader = [DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
    #                 for train_dataset in Train_dataset]
    #Test_dataset = [TensorDataset(torch.tensor(x_test, dtype=torch.float)) for x_test in X_test]
    # Test_loader = [DataLoader(test_dataset, batch_size=X_test.shape[0], shuffle=True) for test_dataset in Test_dataset]
    

   

    
    best_state = None
    lossTestBest = np.inf
    loss_func = GaussianMixAutoEncoderLoss(gamma=1.0, eta=1.0)
    bestEpoch = 0
    #Component_Responsibilities = [[torch.ones(x.shape[0], 1)/len(s2c) for _ in s2c] for x, s2c in zip(X_train, Sample2Component)]
    for epoch in range(epochs):
        for i in range(iterations_per_epoch):
            # Batch_Indices = [[WeightedRandomSampler(responsibility.flatten(),batch_size, replacement=True) for responsibility 
            #        in component_responsibilities] for component_responsibilities in Component_Responsibilities]
            Batch_Indices = [gaussianMixLatentSpace.batchIndices(z=model(x)[0], sample_index=s) for s, x in enumerate(X_train)]
            X_Batch = [torch.vstack([x[list(ix)] for ix in batch_indices]) for x, batch_indices in zip(X_train, Batch_Indices)]
            if epoch < warmup_epochs:
                if i == 0: 
                    print("Epoch " + str(epoch) + ". Warmup epochs are on")
                loss, metrics, Images  = loss_func(X_Batch, model, gaussianMixLatentSpace, warmup=True)
            else:
                if i == 0: 
                    print("Epoch " + str(epoch) + ". Warmup epochs are over")
                #gaussianMixLatentSpace.fitGMM([model(x)[0] for x in  X_Batch])
                if epoch == warmup_epochs and i == 0:
                    num_iter = 1000
                else:
                    num_iter = 100
                gaussianMixLatentSpace.fitGMM([model(x[torch.randint(0,x.shape[0],(2000,))])[0] for x in  X_train],
                                               num_iter =  num_iter)
                loss, metrics, Images  = loss_func(X_Batch, model, gaussianMixLatentSpace, warmup=False)
            writer.add_scalars('metrics/train/update', metrics, epoch*iterations_per_epoch + i)
            with torch.no_grad():
                loss_test, metrics_test = loss_func.validation_loss(X_test, model, gaussianMixLatentSpace)
                writer.add_scalars('metrics/test/update', metrics_test, epoch*iterations_per_epoch + i)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if lossTestBest >= loss_test:
                lossTestBest = loss_test
                best_state = {'model':copy.deepcopy(model.state_dict()), 
                              'gaussianMixLatentSpace':copy.deepcopy(gaussianMixLatentSpace.state_dict())}
                bestEpoch = epoch
            if i == 0:
                metricsEpoch = copy.deepcopy(metrics)
                metricsEpoch_test = copy.deepcopy(metrics_test)
            else:
                for key in metrics.keys():
                    metricsEpoch[key] += metrics[key]
                for key in metrics_test.keys():
                    metricsEpoch_test[key] += metrics_test[key]
            #if (epoch+1) % 20 == 0 and i == 0:
            #if True:
            if i == 0:
                [writer.add_image('Sample '+ str(s) + ' embeddings', 
                    VisualizeEmbeddings(x, y, s, model, gaussianMixLatentSpace), 
                    epoch) for x, y, s in zip(X_test, Y_test, range(len(X_test)))]
                # Images = [VisualizeMatching(z, matched_gaussianpoints, y) for z, y, gpoints in zip(Z, Y_Batch, Matched_GPoints)]
                # [writer.add_image('Matching '+str(i), img, epoch) for i, img in enumerate(Images)]
            #Uncomment the line below to learn the proportions
            [gaussianMixLatentSpace.update_proportions(z = model(x)[0],sample_index = s) for s, x in enumerate(X_train)]
            #For some reason the new responsibilities tend to get smaller and smaller for one of the components
            # Component_Responsibilities = [gaussianMixLatentSpace.compute_responsibilities(z = model(x)[0],sample_index = s)[1] for s, x in enumerate(X_train)]
            gaussianMixLatentSpace.resample()
        metricsEpoch = {key: value/iterations_per_epoch for key, value in metricsEpoch.items()}
        metricsEpoch_test = {key: value/iterations_per_epoch for key, value in metricsEpoch_test.items()}
        writer.add_scalars('metrics/train/Epoch', metricsEpoch, epoch)
        writer.add_scalars('metrics/test/Epoch', metricsEpoch_test, epoch)
        # Log the image to TensorBoard
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, Test Loss: {loss_test.item():.4f}")
            writer.flush()
            print("Proportions: " + str(gaussianMixLatentSpace.Proportions[0].data))
            [print(mu.data) for mu in gaussianMixLatentSpace.Mu]
            [print(cov.data) for cov in gaussianMixLatentSpace.Cov]
        #print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, Test Loss: {loss_test.item():.4f}")

       

    print("Best Epoch: " + str(bestEpoch))
    last_state = {'model':copy.deepcopy(model.state_dict()), 
                              'gaussianMixLatentSpace':copy.deepcopy(gaussianMixLatentSpace.state_dict())}
    return best_state, last_state



