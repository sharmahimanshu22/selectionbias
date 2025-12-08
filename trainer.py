from models import FullyConnectedAutoencoder as Model
from loss import GaussianMixAutoEncoderLoss
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler
from plots import VisualizeEmbeddings
#from plots import VisualizeGMMFit
import copy
import sys
from packageutils import *

import torch
import numpy as np
import ast


class SampleLogger:
    
    def __init__(self, dirname, sample_index):
        self.dirname = dirname
        self.gmlsmu = []
        self.gmlscov = []
        self.gmmmu = []
        self.gmmcov = []
        self.gmlsproportions = []
        self.gmmproportions = []
        self.sample_index = sample_index

    def addGMMmu(self, mu_comps):
        self.gmmmu.append(mu_comps)
        print(mu_comps, "gmmmu")

    def addGMLSmu(self, mu_comps):
        self.gmlsmu.append(mu_comps)
        print(mu_comps, "gmlsmu")
    
    def addGMMcov(self, cov_comps):
        self.gmmcov.append(cov_comps)

    def addGMLScov(self, cov_comps):
        self.gmlscov.append(cov_comps)
    
    def addGMMproportions(self, props):
        self.gmmproportions.append(props)
        print(props, "gmmprops")
    
    def addGMLSproportions(self, props):
        self.gmlsproportions.append(props)
        print(props, "gmlsprops")

    def save(self):
        with open(os.path.join(self.dirname, "gmm_mu_" + str(self.sample_index) + ".txt"), 'a'  ) as f:
            for mus in self.gmmmu:
                for m in mus:
                    mstring = np.array2string(m, separator=',')
                    f.write(mstring)
                    f.write('\t')
                f.write('\n')
            f.close()

        with open(os.path.join(self.dirname, "gmm_cov_" + str(self.sample_index) + ".txt"), 'a'  ) as f:
            for covs in self.gmmcov:
                for cov in covs:
                    covstring = np.array2string(cov, separator=',').replace('\n', ',')
                    f.write(covstring)
                    f.write('\t')
                f.write('\n')
            f.close()

        with open(os.path.join(self.dirname, "gmls_cov_" + str(self.sample_index) + ".txt"), 'a'  ) as f:
            for covs in self.gmlscov:
                for cov in covs:
                    covstring = np.array2string(cov, separator=',').replace('\n', ',')
                    f.write(covstring)
                    f.write('\t')
                f.write('\n')
            f.close()

        with open(os.path.join(self.dirname, "gmls_mu_" + str(self.sample_index) + ".txt") , 'a' ) as f:
            for mus in self.gmlsmu:
                for m in mus:
                    mstring = np.array2string(m, separator=',')
                    f.write(mstring)
                    f.write('\t')
                f.write('\n')
            f.close()

        with open(os.path.join(self.dirname, "gmm_props_" + str(self.sample_index) + ".txt") , 'a' ) as f:
            for p in self.gmmproportions:
                pstring = ','.join([str(e) for e in p])
                f.write(pstring)
                f.write('\n')
            f.close()

        with open(os.path.join(self.dirname, "gmls_props_" + str(self.sample_index) + ".txt") , 'a' ) as f:
            for p in self.gmlsproportions:
                pstring = ','.join([str(e) for e in p])
                f.write(pstring)
                f.write('\n')
            f.close()

        


def trainer(msgmd, model, gaussianMixLatentSpace, loss_func,  optimizer, test_size=0.2, 
batch_size=0.2, epochs=1000, warmup_epochs = 20, writer=None, dirname = None):
    X = msgmd.X
    Y = msgmd.y
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

    testdatasubdir = os.path.join(dirname, "testdata")
    os.makedirs(testdatasubdir,exist_ok=True)
    for s in range(len(msgmd.X)):
        true_resp_all_comps_sample = msgmd.responsibility_wrt_sample_comps(s,X_test[s])
        np.savetxt(os.path.join(testdatasubdir,"xtest_sample_" + str(s)+ ".txt"), X_test[s])
        np.savetxt(os.path.join(testdatasubdir,"trueposteriors_sample_" + str(s) + ".txt"), true_resp_all_comps_sample)

    sampledataloggers = [SampleLogger(dirname, i) for i in range(len(X_train))]


    # 16000
    max_data_size = max([x_train.shape[0] for x_train in X_train])  # Maximum number of data points in a dataset
    # 100
    iterations_per_epoch = max_data_size // batch_size  # Number of iterations per epoch

    best_state = None
    lossTestBest = np.inf
    loss_func = GaussianMixAutoEncoderLoss(gamma=1.0, eta=1.0)
    bestEpoch = 0

    for epoch in range(epochs):
        #print(iterations_per_epoch, batch_size)
        for i in range(iterations_per_epoch):
            #print(i)
            Batch_Indices = [gaussianMixLatentSpace.batchIndices(z=model(x)[0], sample_index=s) for s, x in enumerate(X_train)]
            X_Batch = [torch.vstack([x[list(ix)] for ix in batch_indices]) for x, batch_indices in zip(X_train, Batch_Indices)]
            #above X_Batch has 200 points from original input. They were selected by weighted sampling the model encoded values, 
            # based on responsibilities assigned to the encoded values. 100 values for sampled based on responsibility for first 
            # component and 100 values were sampled based on responsibility of second component. Responsibility computation
            
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
                              'gaussianMixLatentSpace':copy.deepcopy(gaussianMixLatentSpace.state_dict()),
                              'gmm':copy.deepcopy(gaussianMixLatentSpace.GMM.__dict__)}
                bestEpoch = epoch
            if i == 0:
                metricsEpoch = copy.deepcopy(metrics)
                metricsEpoch_test = copy.deepcopy(metrics_test)
            else:
                for key in metrics.keys():
                    metricsEpoch[key] += metrics[key]
                for key in metrics_test.keys():
                    metricsEpoch_test[key] += metrics_test[key]
            if i == 0:
                [writer.add_image('Sample '+ str(s) + ' embeddings', 
                    VisualizeEmbeddings(x, y, s, model, gaussianMixLatentSpace), 
                    epoch) for x, y, s in zip(X_test, Y_test, range(len(X_test)))]
                produce_posterior_report_for_all_samples(gaussianMixLatentSpace, model, msgmd, X_test, writer, epoch, dirname)

            #Uncomment the line below to learn the proportions
            [gaussianMixLatentSpace.update_proportions(z = model(x)[0],sample_index = s) for s, x in enumerate(X_train)]
            #For some reason the new responsibilities tend to get smaller and smaller for one of the components
            gaussianMixLatentSpace.resample()

        for i in range(len(X_train)):
            sampledataloggers[i].addGMLSmu([gaussianMixLatentSpace.Mu[j].numpy() for j in gaussianMixLatentSpace.Sample2Component[i]])
            sampledataloggers[i].addGMLScov([gaussianMixLatentSpace.Cov[j].numpy() for j in gaussianMixLatentSpace.Sample2Component[i]])
            sampledataloggers[i].addGMLSproportions(gaussianMixLatentSpace.Proportions[i].numpy() )

            sampledataloggers[i].addGMMmu([gaussianMixLatentSpace.GMM.compDist[j].mu for j in gaussianMixLatentSpace.GMM.cMemPerSample[i] ] )
            sampledataloggers[i].addGMMcov([gaussianMixLatentSpace.GMM.compDist[j].cov for j in gaussianMixLatentSpace.GMM.cMemPerSample[i] ] )
            sampledataloggers[i].addGMMproportions(gaussianMixLatentSpace.GMM.mProp[i] )


        
        metricsEpoch = {key: value/iterations_per_epoch for key, value in metricsEpoch.items()}
        metricsEpoch_test = {key: value/iterations_per_epoch for key, value in metricsEpoch_test.items()}
        writer.add_scalars('metrics/train/Epoch', metricsEpoch, epoch)
        writer.add_scalars('metrics/test/Epoch', metricsEpoch_test, epoch)

        # Log the image to TensorBoard
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, Test Loss: {loss_test.item():.4f}")
            writer.flush()
            for s in sampledataloggers:
                s.save()
            #writer.add_scalar("Sample1 GMLS Proporition", gaussianMixLatentSpace.Proportions[0].data)
            
            """ print("Proportions1: " + str(gaussianMixLatentSpace.Proportions[0].data))
            print("Proportions2: " + str(gaussianMixLatentSpace.Proportions[1].data))
            print("Proportions1 and 2 GMM: " + str(gaussianMixLatentSpace.GMM.mProp))
            print("Mu1 GMM", gaussianMixLatentSpace.GMM.compDist[0].mu)
            print("Mu2 GMM", gaussianMixLatentSpace.GMM.compDist[1].mu)
            print("Cov1 GMM", gaussianMixLatentSpace.GMM.compDist[0].cov)
            print("Cov2 GMM", gaussianMixLatentSpace.GMM.compDist[1].cov)

            [print(mu.data) for mu in gaussianMixLatentSpace.Mu]
            [print(cov.data) for cov in gaussianMixLatentSpace.Cov] """

       
    
    print("Best Epoch: " + str(bestEpoch))
    last_state = {'model':copy.deepcopy(model.state_dict()), 
                              'gaussianMixLatentSpace':copy.deepcopy(gaussianMixLatentSpace.state_dict())}
    
    return best_state, last_state



