import pdb

import numpy as np
from scipy.special import logsumexp
from sklearn.metrics import pairwise_distances

def responsibilityAndLogp(x, comps, pi):
    logpdfs_c = np.array([comp.logpdf(x) for comp in comps]).T
    #pdb.set_trace()
    logpi = np.log(pi)
    logr = logpdfs_c + logpi
    logpdf = logsumexp(logr, axis=1)
    r = np.exp(logr - logpdf[:, None])
    return r, logpdf

def matchComponents(x, r1, r2):
    assert r1.shape[1] == r2.shape[1]
    m1 = np.vstack([np.sum(x*r[:,None], axis=0, keepdims=True)/np.sum(r) for r in r1.T])
    m2 = np.vstack([np.sum(x*r[:,None], axis=0, keepdims=True)/np.sum(r) for r in r2.T])
    #pdb.set_trace()
    dist = pairwise_distances(m1, m2)
    perm1 = np.ones(r1.shape[1]) * np.nan
    perm2 = np.ones(r2.shape[1]) * np.nan
    i = 0
    while np.any(np.isfinite(dist)):
        ix = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
        dist[ix[0], :] = np.inf
        dist[:, ix[1]] = np.inf
        perm1[i] = ix[0]
        perm2[i] = ix[1]
        i += 1
    perm1 = perm1.astype('int64')
    perm2 = perm2.astype('int64')
    return perm1, perm2




def ellipse_radius_along_vector(cov, n_std, v):
    """
    Computes the distance from the center to the ellipse boundary along a given vector.
    
    Args:
        cov (np.ndarray): 2x2 covariance matrix
        n_std (float): number of standard deviations (scales the ellipse)
        v (np.ndarray): 2D direction vector

    Returns:
        float: distance from center to ellipse boundary along direction v
    """
    # Normalize the vector
    v = np.asarray(v)
    v = v / np.linalg.norm(v)
    
    # Invert covariance matrix
    cov_inv = np.linalg.inv(cov)
    
    # The ellipse equation is (x.T @ cov_inv @ x) = (n_std)^2
    # Substitute x = r * v
    # Solve for r: r = n_std / sqrt(v.T @ cov_inv @ v)
    
    denominator = np.sqrt(v.T @ cov_inv @ v)
    r = n_std / denominator
    
    return r

# # Example usage
# cov = np.array([[3, 1], 
#                 [1, 2]])
# n_std = 2.0
# v = np.array([1, 2])  # arbitrary direction

# radius = ellipse_radius_along_vector(cov, n_std, v)
# print(f"Radius along vector {v}: {radius:.3f}")


