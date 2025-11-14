import os
import numpy as np
import cv2
from skimage.feature import local_binary_pattern
import scipy.io as sio

def rgb_histogram_feature(img, Q):
    """
    Computes a 3D RGB color histogram.
    'img' is assumed to be a normalized (0-1) BGR image.
    
    Args:
        img (np.array): Normalized input image.
        Q (int): Number of quantization bins per channel.

    Returns:
        np.array: A 1D normalized histogram of size (1, Q^3).
    """
    # Quantize image channels
    img_quantized = np.floor(img * Q).astype(np.int32)
    img_quantized = np.clip(img_quantized, 0, Q-1)
    
    # Flatten channels
    R = img_quantized[:, :, 0].ravel()
    G = img_quantized[:, :, 1].ravel()
    B = img_quantized[:, :, 2].ravel()
    
    # Combine channels into a single bin index
    bin_indices = R * (Q**2) + G * Q + B
    
    # Calculate histogram
    histogram, _ = np.histogram(bin_indices, bins=np.arange(Q**3 + 1))
    
    # Normalize histogram
    histogram = histogram.astype(np.float64)
    if np.sum(histogram) > 0:
        histogram = histogram / np.sum(histogram)
        
    return histogram.reshape(1, -1)

def hsv_feature(img, h_bins=16, s_bins=8, v_bins=8):
    """
    Computes a concatenated 1D HSV color histogram.
    'img' is assumed to be a normalized (0-1) BGR image.

    Returns:
        np.array: A 1D normalized histogram of size (1, h_bins + s_bins + v_bins).
    """
    # Convert to 8-bit uint and then to HSV
    img_uint8 = (img * 255).astype(np.uint8)
    img_hsv = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2HSV)
    
    # Calculate histograms for each channel
    h_hist, _ = np.histogram(img_hsv[:,:,0].ravel(), bins=h_bins, range=(0, 180))
    s_hist, _ = np.histogram(img_hsv[:,:,1].ravel(), bins=s_bins, range=(0, 256))
    v_hist, _ = np.histogram(img_hsv[:,:,2].ravel(), bins=v_bins, range=(0, 256))
    
    # Normalize each histogram individually
    if np.sum(h_hist) > 0: h_hist = h_hist.astype(np.float64) / np.sum(h_hist)
    if np.sum(s_hist) > 0: s_hist = s_hist.astype(np.float64) / np.sum(s_hist)
    if np.sum(v_hist) > 0: v_hist = v_hist.astype(np.float64) / np.sum(v_hist)
    
    # Concatenate normalized histograms
    F = np.concatenate((h_hist, s_hist, v_hist))
    
    return F.reshape(1, -1) 

def lbp_feature(img, radius=1):
    """
    Computes a global LBP (Local Binary Pattern) histogram.
    'img' is assumed to be a normalized (0-1) BGR image.

    Returns:
        np.array: A 1D normalized histogram of size (1, (8*radius) + 2).
    """
    # Convert to 8-bit uint and then to grayscale
    img_uint8 = (img * 255).astype(np.uint8)
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2GRAY)
    
    # Define LBP parameters
    n_points = 8 * radius
    
    # Compute the LBP image using 'uniform' method
    lbp = local_binary_pattern(gray, P=n_points, R=radius, method='uniform')
    
    # Calculate histogram of LBP codes
    n_bins = n_points + 2
    hist, _ = np.histogram(lbp.ravel(),
                           bins=n_bins,
                           range=(0, n_bins))
    
    # Normalize histogram
    if np.sum(hist) > 0:
        hist = hist.astype(np.float64) / np.sum(hist)
        
    return hist.reshape(1, -1)

def eoh_feature(patch, bins=8):
    """
    Computes an EOH (Edge Orientation Histogram) feature for an image patch.
    'patch' is assumed to be a normalized (0-1) BGR image.
    
    Returns:
        np.array: A 1D normalized histogram of size (1, bins).
    """
    # Convert to 8-bit uint and then to grayscale
    if patch.ndim == 3 and patch.shape[2] == 3:
        patch_uint8 = (patch * 255).astype(np.uint8)
        gray = cv2.cvtColor(patch_uint8, cv2.COLOR_BGR2GRAY)
    else:
        # Handle if patch is already grayscale
        if patch.ndim == 2:
             patch_uint8 = (patch * 255).astype(np.uint8)
             gray = patch_uint8
        else:
             return np.zeros((1, bins)) # Invalid input
             
    # Calculate gradients using Sobel operators
    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate magnitude and orientation
    mag = np.sqrt(dx**2 + dy**2)
    orient = np.arctan2(dy, dx)
    
    # Threshold to keep only strong edges
    threshold = np.mean(mag) * 0.5
    strong_orient = orient[mag > threshold]
    
    if strong_orient.size == 0:
        return np.zeros((1, bins)) # No strong edges found
        
    # Shift orientations from [-pi, pi] to [0, 2*pi]
    strong_orient_shifted = strong_orient + np.pi
    
    # Quantize orientations into bins
    bin_indices = (strong_orient_shifted / (2 * np.pi)) * bins
    bin_indices = np.floor(bin_indices).astype(int)
    bin_indices[bin_indices == bins] = bins - 1  # Handle edge case
    
    # Calculate histogram
    hist, _ = np.histogram(bin_indices, bins=bins, range=(0, bins))
    
    # Normalize histogram
    hist = hist.astype(np.float64)
    if np.sum(hist) > 0:
        hist = hist / np.sum(hist)
        
    return hist.reshape(1, -1)

def generate_spatial_grid_feature(img, grid_rows=4, grid_cols=4, feature_types=['color_hist', 'texture'], color_Q=4, texture_bins=8):
    """
    Computes a spatial grid feature by dividing the image and extracting
    features from each cell, then concatenating them.
    
    Args:
        img (np.array): Normalized (0-1) BGR input image.
        grid_rows (int): Number of rows in the grid.
        grid_cols (int): Number of columns in the grid.
        feature_types (list): List of features to extract (e.g., 'color_hist', 'texture').
        color_Q (int): 'Q' parameter for 'color_hist'.
        texture_bins (int): 'bins' parameter for 'texture' (EOH).

    Returns:
        np.array: A 1D concatenated feature vector for the entire grid.
    """
    h, w, _ = img.shape
    cell_h = h // grid_rows
    cell_w = w // grid_cols
    
    all_cell_features = []
    
    # Pre-calculate size of an empty feature vector
    d_size = 0
    if 'avg_color' in feature_types: d_size += 3
    if 'color_hist' in feature_types: d_size += color_Q**3
    if 'texture' in feature_types: d_size += texture_bins
    empty_feat = np.zeros((1, d_size))

    # Iterate over the grid
    for r in range(grid_rows):
        for c in range(grid_cols):
            # Extract the patch
            patch = img[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w]
            
            if patch.shape[0] == 0 or patch.shape[1] == 0:
                all_cell_features.append(empty_feat) # Append empty vector for 0-size patch
                continue
                
            cell_feats = []
            
            # --- Extract requested features for the patch ---
            if 'avg_color' in feature_types:
                avg_bgr = cv2.mean(patch)[:3]
                avg_rgb = np.array([avg_bgr[2], avg_bgr[1], avg_bgr[0]])
                cell_feats.append(avg_rgb.reshape(1, -1))
                
            if 'color_hist' in feature_types:
                color_hist = rgb_histogram_feature(patch, color_Q)
                cell_feats.append(color_hist)
                
            if 'texture' in feature_types:
                texture_hist = eoh_feature(patch, texture_bins)
                cell_feats.append(texture_hist)
                
            # Concatenate features for this cell
            if cell_feats:
                all_cell_features.append(np.concatenate(cell_feats, axis=1))
            else:
                 all_cell_features.append(empty_feat) # Append empty if no features selected
                 
    # Concatenate all cell features into one final vector
    return np.concatenate(all_cell_features, axis=1)
