import os
import numpy as np
import cv2
import scipy.io as sio
from extractRandom import (
    rgb_histogram_feature, 
    hsv_feature, 
    lbp_feature
)

def compute_and_save_global_descriptor(method, params, dataset_folder, out_folder):
    """
    Computes and saves global descriptors for all images in the dataset.
    
    Args:
        method (str): The descriptor type ('rgb', 'hsv', 'lbp').
        params (dict): Parameters for the chosen descriptor.
        dataset_folder (str): Path to the MSRC dataset root.
        out_folder (str): Base directory to save descriptor files.
    """
    
    # 1. Determine descriptor function and output folder based on method
    if method == 'rgb':
        Q = params.get('Q', 8)
        descriptor_func = lambda img: rgb_histogram_feature(img, Q=Q)
        out_subfolder = f'global_rgb_histo_Q{Q}'
        print(f"Computing 'rgb' descriptors with Q={Q}...")
        
    elif method == 'hsv':
        h_bins = params.get('h_bins', 16)
        s_bins = params.get('s_bins', 8)
        v_bins = params.get('v_bins', 8)
        descriptor_func = lambda img: hsv_feature(img, h_bins, s_bins, v_bins)
        out_subfolder = f'global_hsv_histo_H{h_bins}_S{s_bins}_V{v_bins}'
        print(f"Computing 'hsv' descriptors (H={h_bins}, S={s_bins}, V={v_bins})...")
        
    elif method == 'lbp':
        radius = params.get('radius', 1)
        descriptor_func = lambda img: lbp_feature(img, radius)
        out_subfolder = f'global_lbp_histo_R{radius}'
        print(f"Computing 'lbp' descriptors (Radius={radius})...")
        
    else:
        raise ValueError(f"Unknown global descriptor method: {method}")

    # Create the final output directory
    output_path = os.path.join(out_folder, out_subfolder)
    os.makedirs(output_path, exist_ok=True)
    
    # 2. Iterate through images and process them
    image_dir = os.path.join(dataset_folder, 'Images')
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".bmp")]
    
    descriptor_dim = 0
    
    for i, filename in enumerate(image_files):
        # Print progress update
        if (i+1) % 100 == 0:
            print(f"... processing image {i+1}/{len(image_files)}")
            
        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Warning: Could not read {filename}. Skipping.")
            continue
            
        # Normalize image to [0, 1] range
        img_normalized = img.astype(np.float64) / 255.0
        fout = os.path.join(output_path, filename.replace('.bmp', '.mat'))

        # Compute the feature
        F = descriptor_func(img_normalized)
        
        # Store descriptor dimension from the first valid feature
        if i == 0:
            descriptor_dim = F.shape[1]

        # Save descriptor to .mat file
        sio.savemat(fout, {'F': F})

    print(f"\nFinished! Descriptors saved to {output_path}")
    print(f"Each descriptor has {descriptor_dim} dimensions")
    
    return output_path
