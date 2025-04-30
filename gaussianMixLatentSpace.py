import torch
import numpy as np
import torch.nn as nn
from typing import Optional
from DataLoader import WeightedRandomSampler2
from GMM.gmm import GMM
from GMM.compModel import MVN

class GaussianMixLatentSpace(nn.Module):
    def __init__(self, Mu, Cov, Sample2Component, Proportions, Sample_Sizes, points_per_component=100, sigma=0.1):
        """
        Initialize the GaussianSampler class.

        Args:
            Mu (list): A list containing the means for Gaussian samples.
            Cov (list): A list containing the covariances for Gaussian samples.
            Sample2Component (list): A list of lists, where each inner list contains the indices of the Gaussian components
                that correspond to a sample.
            Proportions (list): A list of component proportions for each sample.
            Sample_Sizes (list): A list containing the sizes of each sample.
            points_per_component (int): Number of points to sample per Gaussian component. Defaults to 100.
            sigma (float): Kernel bandwidth for converting pairwise distances to probabilities. Defaults to 0.1.
        """
        super(GaussianMixLatentSpace, self).__init__()
        self.points_per_component = points_per_component
        self.dim =  Mu[0].shape[0]    # Gaussian Dimension
        self.num_components = len(Mu) # Number of Gaussian components.
        self.Gaussianpoints = [self._sample_gaussian() for _ in range(self.num_components)]
        self.Mu = nn.ParameterList([nn.Parameter(torch.tensor(mu, dtype=torch.float32), requires_grad=True) for mu in Mu])
        self.Cov = nn.ParameterList([nn.Parameter(torch.tensor(cov, dtype=torch.float32), requires_grad=False) for cov in Cov])
        self.Sample2Component = Sample2Component
        #Proportions = [torch.ones(len(s2c)) / len(s2c) for s2c in Sample2Component]
        self.Proportions = nn.ParameterList([nn.Parameter(torch.tensor(proportion, dtype=torch.float32), 
                                                          requires_grad=False) for proportion in Proportions])
        self.Proportions_min = [points_per_component/sample_size for sample_size in Sample_Sizes]
        self.GMM = self.initializeGMM() 

        self.sigma = sigma

    def initializeGMM(self):
        compDist = [MVN(dim=self.dim, unitCov=False) for _ in range(self.num_components)]
        gmm = GMM(self.num_components, self.dim, compDist=compDist, nMix=len(self.Sample2Component),
                   cMemPerSample=self.Sample2Component, equalPropComps=False, identicalCov=True)
        return gmm

    def _sample_gaussian(self):
        """
        Sample points from a standard Gaussian distribution.

        Returns:
            np.ndarray: An array of shape (m, DD) with points sampled from a standard Gaussian.
        """
        #np.random.normal(loc=0.0, scale=1.0, size=(self.points_per_component, self.dim))
        return torch.randn((self.points_per_component, self.dim))

    def resample(self):
        """
        Resample all Gaussian points.
        """
        self.Gaussianpoints = [self._sample_gaussian() for _ in range(self.num_components)]

    def transform_gaussians(self, sample_index: Optional[int] = None):
        """
        Transform all standard gaussian samples using the means and covariances.
        Returns:
            list: A list of transformed Gaussian samples as tensors.
        """
        Transformed_Gaussianpoints = []
        for gaussianpoints, mu, cov in zip(self.Gaussianpoints, self.Mu, self.Cov):
            cov_cholesky = torch.linalg.cholesky(cov)
            transformed_gaussianpoints = gaussianpoints @ cov_cholesky.T + mu 
            Transformed_Gaussianpoints.append(transformed_gaussianpoints)
        Transformed_Gaussianpoints = [Transformed_Gaussianpoints[component] for component in 
                                      self.Sample2Component[sample_index]] if sample_index is not None else Transformed_Gaussianpoints
        return Transformed_Gaussianpoints
    
    def compute_pairwise_distances(self, z: torch.Tensor, gaussians: list):
        """
        Compute the pairwise distances between embeddings and transformed Gaussian points.

        Args:
            z (torch.Tensor): A tensor representing embeddings of input point x.
            gaussians (list): A list of transformed Gaussian samples as tensors.

        Returns:
            list: A list of tensors containing the pairwise distances between embeddings and gaussian points.
        """
        distances = []
        for gaussian in gaussians:
            pairwise_dist = torch.cdist(z, gaussian) / self.sigma
            distances.append(pairwise_dist)
        return distances
    
    def compute_responsibilities(self, z: torch.Tensor, Gaussianpoints: Optional[list] = None, 
                                 proportion: Optional[torch.Tensor] = None, sample_index: Optional[int] = None, 
                                 equal_proportion: bool = False):
        if Gaussianpoints is None:
            Gaussianpoints = self.transform_gaussians()
            if sample_index is not None:
                Gaussianpoints = [Gaussianpoints[i] for i in self.Sample2Component[sample_index]]
        if proportion is None and not equal_proportion and sample_index is not None:
            proportion = self.Proportions[sample_index].data
        elif equal_proportion:
            proportion = torch.ones(len(Gaussianpoints))/len(Gaussianpoints)
        assert len(Gaussianpoints) == len(proportion), "Number of components and size of proportion must be equal."
        pairwise_distances = self.compute_pairwise_distances(z, Gaussianpoints)
        point_responsibilities, component_responsibilities = self._compute_responsibilities(pairwise_distances , proportion)
        #point_responsibilities, component_responsibilities = self._compute_responsibilities(z, Gaussianpoints, proportion)
        return point_responsibilities, component_responsibilities, pairwise_distances

    # def compute_densities(self, z: torch.Tensor, Gaussianpoints: list):
    #     pairwise_distances = self.compute_pairwise_distances(z, Gaussianpoints)
    #     point_densities = []
    #     component_densities = []
    #     for pairwise_dist in pairwise_distances:
    #         probability = torch.exp(-pairwise_dist)
    #         point_densities.append(probability)
    #         component_densities.append(probability.sum(dim=1, keepdim=True))
    #     return point_densities, component_densities
    
    # def _compute_responsibilities(self, z: torch.Tensor, Gaussianpoints: list, proportion: torch.Tensor):
    #     """
    #     Compute the responsibilities for each Gaussian sample.

    #     Args:
    #         z (torch.Tensor): A tensor representing embeddings of a set of input points x.
    #         gaussians: A list of transformed Gaussian samples as tensors.
    #         proportion: A probability vector for Gaussian proportions. Defaults to equal proportions.

    #     Returns:
    #         list: A list containing the responsibilities representing the probability that an embedding is 
    #         generated by a Gaussian point.
    #     """
    #     point_densities, component_densities = self.compute_densities(z, Gaussianpoints)
    #     point_densities = [density*p for density, p in zip(point_densities, proportion)]
    #     component_densities = [density*p for density, p in zip(component_densities, proportion)]
    #     denominator = torch.hstack(component_densities).sum(dim=1, keepdim=True)
    #     point_responsibilities = [density/denominator for density in point_densities]
    #     component_responsibilities = [density/denominator for density in component_densities]
    #     return point_responsibilities, component_responsibilities

    def _compute_responsibilities(self, pairwise_distances, proportion):
        exponents = [-distances + torch.log(prop) for (prop, distances) in zip (proportion, pairwise_distances)]
        max_exponent = torch.max(torch.hstack(exponents), dim=1, keepdim=True)[0]
        exponents = [exponent - max_exponent for exponent in exponents]
        logSumExponent = torch.log(torch.sum(torch.exp(torch.hstack(exponents)), dim=1, keepdim=True))
        point_responsibilities = [torch.exp(exponent - logSumExponent) for exponent in exponents]
        component_responsibilities = [torch.sum(pResp, dim=1, keepdim=True) for pResp in point_responsibilities]
        return point_responsibilities, component_responsibilities

    
    # def update_proportions(self, z: torch.Tensor, sample_index: int, num_iter: int = 10):
    #     s2c = self.Sample2Component[sample_index]
    #     Gaussianpoints = self.transform_gaussians(sample_index=sample_index)
    #     proportion = self.Proportions[sample_index].data
    #     prop_min = self.Proportions_min[sample_index]
    #     _, component_densities = self.compute_densities(z, Gaussianpoints)
    #     for _ in range(num_iter):
    #         weighted_densities = [density*p for density, p in zip(component_densities, proportion)]
    #         denominator = torch.hstack(weighted_densities).sum(dim=1, keepdim=True)
    #         component_responsibilities = [density/denominator for density in weighted_densities]
    #         proportion = torch.hstack(component_responsibilities).mean(dim=0)
    #         if torch.any(proportion < prop_min):
    #             index = proportion < prop_min
    #             num_index = index.sum()
    #             proportion[index] = prop_min
    #             proportion[~index] = (1-prop_min*num_index)*proportion[~index]/torch.sum(proportion[~index])
    #     self.Proportions[sample_index].data = proportion
    #     return proportion, component_responsibilities

    def update_proportions(self, z: torch.Tensor, sample_index: int, num_iter: int = 10):
        s2c = self.Sample2Component[sample_index]
        Gaussianpoints = self.transform_gaussians(sample_index=sample_index)
        proportion = self.Proportions[sample_index].data
        prop_min = self.Proportions_min[sample_index]
        pairwise_distances = self.compute_pairwise_distances(z, Gaussianpoints)
        for _ in range(num_iter):
            _, component_responsibilities = self._compute_responsibilities(pairwise_distances, proportion)
            proportion = torch.hstack(component_responsibilities).mean(dim=0)
            if torch.any(proportion < prop_min):
                index = proportion < prop_min
                num_index = index.sum()
                proportion[index] = prop_min
                proportion[~index] = (1-prop_min*num_index)*proportion[~index]/torch.sum(proportion[~index])
        self.Proportions[sample_index].data = proportion
        #print('Proportions', proportion)
        return proportion, component_responsibilities
    
    def fitGMM(self, Z: list, num_iter: int = 25):
        Z_np = [z.detach().numpy() for z in Z]
        self.GMM.refit(Z_np, maxIter=num_iter)
        for i in range(self.num_components):
            self.Mu[i].data = 0.0*self.Mu[i].data + 1.0*self.GMM.compDist[i].mu.astype(np.float32)
            self.Cov[i].data = 0.0*self.Cov[i].data + 1.0*self.GMM.compDist[i].cov.astype(np.float32)

    
    def batchIndices(self, z: torch.Tensor, sample_index: int):
        """
        Sample points from the Gaussian mixture model.

        Args:
            z (torch.Tensor): A tensor representing embeddings of a set of input points x.
            sample_index (int): The index of the sample to use for sampling.

        Returns:
            list: A list of sampled points from the Gaussian mixture model.
        """
       
        _, component_responsibilities, _ = self.compute_responsibilities(z, sample_index=sample_index)
        # batchSizes =[int(min(resp.sum().item(),  self.points_per_component)) for resp in component_responsibilities]
        # batchIndices = [WeightedRandomSampler(responsibility.flatten(), batchSize, replacement=True) 
        #                 for (batchSize, responsibility) in zip(batchSizes, component_responsibilities)] 
        # batchIndices = [batchIX + list(WeightedRandomSampler(np.ones(z.shape[0]), 
        #                                                      max(self.points_per_component-batchSize,0), 
        #                                                      replacement=True)) 
        #                                                      for batchSize, batchIX in zip(batchSizes, batchIndices)]

        batchIndices = [WeightedRandomSampler2(resp.flatten(), self.points_per_component, replacement=True) for resp in component_responsibilities]

        return batchIndices

    
    def forward(self, sample_index: Optional[int] = None):
        """
        Call transform_gaussians method to transform all standard Gaussian samples using the given means and covariances.
        """
        Transformed_Gaussianpoints = self.transform_gaussians(sample_index=sample_index)
        return Transformed_Gaussianpoints
