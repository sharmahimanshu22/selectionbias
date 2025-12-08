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
        

def produce_posterior_report_for_all_samples(gaussianMixLatentSpace, model, msgmd, test_data, writer, epoch, dirname):

    
    for s in range(len(gaussianMixLatentSpace.Sample2Component)):
        latent_embedding = model(torch.tensor(test_data[s], dtype=torch.float32))[0]
        #print(test_data[s], "test_data")
        #print(latent_embedding.detach(), "latent_embedding")
        _,pred_resp_all_comps_gmls,_ = gaussianMixLatentSpace.compute_responsibilities(latent_embedding.detach()
                                                                                       , sample_index=s)
        pred_resp_all_comps_gmm = gaussianMixLatentSpace.GMM.responsibility(latent_embedding.detach(),
                                                                                sample_index=s)
        #
        #print(pred_resp_all_comps_gmm, "comp post gmm")
        
        with open(os.path.join(dirname,"allpredposteriors_gmls_sample_" + str(s) + ".txt"), 'a') as f:
            for ci in range(len(pred_resp_all_comps_gmls)):
                component_posterior = pred_resp_all_comps_gmls[ci].detach().numpy()
                f.write(",".join([str(e[0]) for e in component_posterior]))
                f.write("\n")
            f.write("\n")
            f.close()
        with open(os.path.join(dirname,"allpredposteriors_gmm_sample_" + str(s) + ".txt"), 'a') as f:
            for ci in range(len(pred_resp_all_comps_gmm[0])):
                component_posterior = pred_resp_all_comps_gmm[:,ci]
                f.write(",".join([str(e) for e in component_posterior]))
                f.write("\n")
            f.write("\n")
            f.close()

        #fig = plot_all_posteriors(test_data, [first_comp_true,second_comp_true], [first_comp_pred,second_comp_pred])
        #print("plot done")
        #writer.add_figure("Responsibilitytruevspredicted/resps" ,fig, epoch)