import numpy as np
import torch
from MultiSampleMixtureData import *
import sys
from plots import *

    




def mean_squared_dist(la, lb):
    assert (len(la) == len(lb))
    return sum([ (la[i] - lb[i])**2 for i in range(len(la))])


# works on a single sample
def match_comps(true_resp_all_comps, resp_all_comps):
    assert (len(true_resp_all_comps) == len(resp_all_comps))
    n = len(true_resp_all_comps[0])
    for i in range(len(true_resp_all_comps)):
        assert (len(true_resp_all_comps[i]) == n) # assert that length of sample is constant 

    matching_comps = []
    for i in range(len(true_resp_all_comps)):
        matching_comp = None
        mse = np.inf
        for j in range(len(resp_all_comps)):
            v = mean_squared_dist(true_resp_all_comps[i], resp_all_comps[j])
            if v < mse:
                matching_comp = j
                mse = v
        matching_comps.append((i, matching_comp))

    return matching_comps
        

def produce_posterior_report_for_sample(sample_idx, gaussianMixLatentSpace, model, msgmd : MultiSampleGaussianMixData, test_data, writer, epoch):

    latent_embedding = model(torch.tensor(test_data[sample_idx], dtype=torch.float32))[0]
    _,pred_resp_all_comps,_ = gaussianMixLatentSpace.compute_responsibilities(latent_embedding.detach()
            , proportion = torch.tensor(gaussianMixLatentSpace.GMM.mProp[0]) , sample_index=sample_idx)
    
    true_resp_all_comps = msgmd.responsibility(sample_idx,test_data[sample_idx])

    first_comp_pred = pred_resp_all_comps[0].detach().numpy()
    second_comp_pred = pred_resp_all_comps[1].detach().numpy()
    first_comp_true = true_resp_all_comps[:,0]
    second_comp_true = true_resp_all_comps[:,1]


    image = plot_all_posteriors([first_comp_true,second_comp_true], [first_comp_pred,second_comp_pred])
    writer.add_image("Responsibility true vs predicted" ,image, epoch)






    
    

