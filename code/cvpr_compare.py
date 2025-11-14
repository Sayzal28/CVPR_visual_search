import numpy as np
import scipy.io as sio
import os

# Cache for weighted Euclidean distance (diagonal covariance)
mahalanobis_pca_cache = {'variances': None, 'path': None}
# Cache for full matrix Mahalanobis distance (full covariance)
mahalanobis_full_cache = {'inv_cov': None, 'path': None}

def cvpr_compare(F1, F2, method='euclidean', variance_path=None, inv_cov_path=None):
    """
    Compares two feature vectors (F1, F2) using a specified distance metric.
    
    Args:
        F1 (np.array): The first feature vector.
        F2 (np.array): The second feature vector.
        method (str): The distance metric to use.
        variance_path (str, optional): Path to .mat file for 'mahalanobis_pca'.
        inv_cov_path (str, optional): Path to .mat file for 'mahalanobis_full_pca'.

    Returns:
        float: The calculated distance.
    """
    # Ensure feature vectors are 1D
    F1 = F1.ravel()
    F2 = F2.ravel()
    
    # Small epsilon to prevent division by zero
    epsilon = 1e-10
    
    if method == 'euclidean':        # Standard L2 norm
        dst = np.sqrt(np.sum((F1 - F2) ** 2))
        
    elif method == 'l1':        # Manhattan distance
        dst = np.sum(np.abs(F1 - F2))
        
    elif method == 'chi_squared':        # Chi-Squared distance, common for histograms
        denominator = F1 + F2
        denominator[denominator == 0] = 1e-10  # Avoid division by zero
        dst = 0.5 * np.sum((F1 - F2) ** 2 / denominator)
    
    elif method == 'cosine':        # Cosine distance (1 - similarity)
        similarity = np.dot(F1, F2) / ((np.linalg.norm(F1) * np.linalg.norm(F2)) + epsilon)
        similarity = np.clip(similarity, 0.0, 1.0)  # Ensure valid range
        dst = 1.0 - similarity
        
    elif method == 'mahalanobis_pca':
        # Weighted Euclidean distance (assumes diagonal covariance)
        if variance_path is None:
            raise ValueError("Must provide 'variance_path' for 'mahalanobis_pca' method.")
            
        # Load variances from cache or file
        if mahalanobis_pca_cache['variances'] is None or mahalanobis_pca_cache['path'] != variance_path:
            if not os.path.exists(variance_path):
                raise FileNotFoundError(f"Error: {variance_path} not found.")
            mat = sio.loadmat(variance_path)
            mahalanobis_pca_cache['variances'] = mat['variances'].flatten()
            mahalanobis_pca_cache['path'] = variance_path
            
        variances = mahalanobis_pca_cache['variances']
        
        # Calculate weighted distance
        squared_diffs = (F1 - F2) ** 2
        weighted_diffs = squared_diffs / (variances + epsilon)
        dst = np.sqrt(np.sum(weighted_diffs))

    elif method == 'mahalanobis_full_pca':
        # Full Mahalanobis distance using the inverse covariance matrix
        if inv_cov_path is None:
            raise ValueError("Must provide 'inv_cov_path' for 'mahalanobis_full_pca' method.")
        
        # Load inverse covariance matrix from cache or file
        if mahalanobis_full_cache['inv_cov'] is None or mahalanobis_full_cache['path'] != inv_cov_path:
            if not os.path.exists(inv_cov_path):
                raise FileNotFoundError(f"Error: {inv_cov_path} not found.")
            mat = sio.loadmat(inv_cov_path)
            mahalanobis_full_cache['inv_cov'] = mat['inv_cov']
            mahalanobis_full_cache['path'] = inv_cov_path
            
        inv_cov = mahalanobis_full_cache['inv_cov']
        diff = F1 - F2
        
        # Standard Mahalanobis formula: sqrt(diff.T * C_inv * diff)
        dst = np.sqrt(np.dot(np.dot(diff.T, inv_cov), diff))
        
    else:
        raise ValueError(f"Unknown distance method: {method}")
    
    return dst
