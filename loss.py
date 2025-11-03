import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt
from plots import VisualizeMatching


class GaussianMixAutoEncoderLoss(nn.Module):
    def __init__(self, gamma: float = 1, eta: float = 1, theta: float = 10):
        """
        Initialize the GaussianLoss class.

        Args:
            gamma (float): Weight for the Gaussian component matching loss.
            eta (float): Weight for the variance regularization term.
        """
        super(GaussianMixAutoEncoderLoss, self).__init__()
        self.eta = 10.0
        return  
    
    def forward(self, X: list, model, gaussianMixLatentSpace, warmup=False):
        """
        Compute a loss function using input tensors, embeddings, reconstructions, and transformed Gaussian samples.

        Args:
            X (list): A list containing tensors, each representing a batch of DD dimension data points.
            Z (list): A list containing tensors, each representing embeddings of points in X, possibly of a different dimension.
            XR (list): A list containing tensors, each representing the reconstruction of X from Z, with the same dimension as X.
            gaussians_all (list): A list of all transformed Gaussian samples as tensors.
            gaussians_decoded_all (list): A list of all gaussian samples decoded.

        Returns:
            torch.Tensor: A scalar loss value.
        """
        StandardAutoencoderLoss = []
        GaussianLatentLoss = []
        GaussianDecodedLoss = []
        GuassianResponsibilityLoss = []
        MinDistanceLoss = []
        CovarianceLoss = []
        Images = []
        for sample_index, x in enumerate(X):
            z, z_decoded = model(x)
            StandardAutoencoderLoss.append(torch.nn.functional.mse_loss(x, z_decoded, reduction='sum')) 
            #cov = torch.cov(z.transpose(0, 1))
            if not warmup:
                Gaussianpoints = gaussianMixLatentSpace(sample_index)
                Gaussianpoints_Responsibilities,_, pairwise_distances = gaussianMixLatentSpace.compute_responsibilities(z, Gaussianpoints, equal_proportion=True)
                # Compute the reconstruction loss                
                num_components = len(Gaussianpoints_Responsibilities)
                gaussianpoints_per_component = Gaussianpoints[0].shape[0]
                scaling_factor = num_components*gaussianpoints_per_component
                Gaussianpoints_weight = [responsibility.mean(dim=0) for responsibility in Gaussianpoints_Responsibilities]
                gPoint_weight_loss = ((torch.hstack(Gaussianpoints_weight)-1/(num_components*gaussianpoints_per_component))**2).sum()
                GuassianResponsibilityLoss.append(self.eta*scaling_factor*gPoint_weight_loss)
            else:
                GuassianResponsibilityLoss.append(torch.zeros(1))


        standardAutoencoderLoss = torch.stack(StandardAutoencoderLoss).sum()
        gaussianResponsibilityLoss = torch.stack(GuassianResponsibilityLoss).sum()
        gaussianResponsibilityLoss = self.eta * gaussianResponsibilityLoss
        loss = standardAutoencoderLoss + gaussianResponsibilityLoss
        metrics = {'TotalLoss': loss.item(), 'StandardAutoencoderLoss': standardAutoencoderLoss.item(), 
        'GaussianResponsibilityLoss': gaussianResponsibilityLoss.item()}
        #metrics = {'Total Loss': loss.item(), 'StandardAutoencoderLoss': standardAutoencoderLoss.item()}
        return loss, metrics, Images
    

    def validation_loss(self, X, model, gaussianMixLatentSpace):
        """
        Compute the validation loss.
        Args:
            X (list): A list containing tensors, each representing a batch of DD dimension data points.
            Z (list): A list containing tensors, each representing embeddings of points in X, possibly of a different dimension.
            XR (list): A list containing tensors, each representing the reconstruction of X from Z, with the same dimension as X.
            gaussians_all (list): A list of all transformed Gaussian samples as tensors.
            gaussians_decoded_all (list): A list of all gaussian samples decoded.

        Returns:
            torch.Tensor: A scalar loss value.
        """ 
        StandardAutoencoderLoss = []
        GuassianResponsibilityLoss = []
        for sample_index, x in enumerate(X):
            z, z_decoded = model(x)
            # Compute the reconstruction loss
            StandardAutoencoderLoss.append(torch.nn.functional.mse_loss(x, z_decoded, reduction='sum'))
            
        standardAutoencoderLoss = torch.stack(StandardAutoencoderLoss).sum()
        loss = standardAutoencoderLoss
        metrics = {'TotalLoss': loss.item(), 'StandardAutoencoderLoss': standardAutoencoderLoss.item()
        }
        return loss, metrics
        



    # didn't see this method being used yet
    def matchGaussianPoints(self, z: torch.Tensor, Gaussians: list, Gaussians_decoded: list, Gaussianpoints_Responsibilities):
        """
        Select the Gaussian component and the point within that component proportional to the responsibilities.

        Args:
            z (torch.Tensor): A tensor representing embeddings of input points.
            gaussians (list): A list of transformed Gaussian samples as tensors.
            responsibilities (list): A list of tensors containing the responsibilities giving probability
            that a point in z is generated from the gaussian point.

        Returns:
            torch.Tensor: A tensor of matched Gaussian points corresponding to points in z.
            numpy vector: A numpy vector of matched component indices.
            numpy vector: A numpy vector of matched gaussian point indices.
        """
        Gaussianpoints_Responsibilities = [responsibility + torch.tensor(10**-6) for responsibility in Gaussianpoints_Responsibilities]
        Temp = [responsibility.sum(dim=1, keepdim=True) for responsibility in Gaussianpoints_Responsibilities]
        denominator = torch.hstack(Temp).sum(dim=1, keepdim=True)
        Gaussianpoints_Responsibilities = [responsibility/denominator for responsibility in Gaussianpoints_Responsibilities]
        matched_component_index = np.zeros(z.shape[0], dtype=int)
        matched_gaussianpoint_index = np.zeros(z.shape[0], dtype=int)
        matched_gaussianpoints = torch.zeros_like(z)
        matched_gaussianpoints_decoded = torch.zeros((z.shape[0], Gaussians_decoded[0].shape[1]))
        for i in range(Gaussianpoints_Responsibilities[0].shape[0]):
            component_responsibility = [responsibility[i, :].sum() for responsibility in Gaussianpoints_Responsibilities]
            component = torch.multinomial(torch.tensor(component_responsibility), num_samples=1).item()
            gaussianpoint_responsibilities = Gaussianpoints_Responsibilities[component][i, :]
            gaussianpoint_index = torch.multinomial(gaussianpoint_responsibilities, num_samples=1).item()
            matched_component_index[i] = component
            matched_gaussianpoint_index[i] = gaussianpoint_index
            matched_gaussianpoints[i] = Gaussians[component][gaussianpoint_index]
            matched_gaussianpoints_decoded[i] = Gaussians_decoded[component][gaussianpoint_index]
            Gaussianpoints_Responsibilities[component][:, gaussianpoint_index] = 0
        return matched_gaussianpoints, matched_gaussianpoints_decoded, matched_component_index, matched_gaussianpoint_index


