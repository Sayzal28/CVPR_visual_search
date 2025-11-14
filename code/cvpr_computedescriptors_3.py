import os
import numpy as np
import cv2
import scipy.io as sio
import argparse
from extractRandom import generate_spatial_grid_feature

def generate_spatial_descriptors(args):
    """
    Iterates through the dataset and computes spatial grid descriptors
    for each image, saving them to .mat files.
    """
    
    # 1. Create a descriptive output folder name based on parameters
    feature_str = "_".join(sorted(args.features))
    out_subfolder = f"spatial_grid_{args.grid_rows}x{args.grid_cols}_{feature_str}"
    
    # Add feature-specific parameters to the folder name
    if 'color_hist' in args.features:
        out_subfolder += f"_Q{args.color_Q}"
    if 'texture' in args.features:
        out_subfolder += f"_T{args.texture_bins}"
            
    output_path = os.path.join(args.out_folder, out_subfolder)
    os.makedirs(output_path, exist_ok=True)

    # 2. Print a summary of the computation job
    print(f"--- Starting Requirement 3 Descriptor Computation ---")
    print(f"Grid Size: {args.grid_rows}x{args.grid_cols}")
    print(f"Cell Features: {args.features}")
    if 'color_hist' in args.features:
        print(f"Color Q: {args.color_Q}")
    if 'texture' in args.features:
        print(f"Texture Bins: {args.texture_bins}")
    print(f"Saving to: {output_path}\n")

    # Get list of all images
    image_dir = os.path.join(args.dataset_folder, 'Images')
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".bmp")]
    
    final_descriptor_size = 0

    # 3. Process each image
    for i, filename in enumerate(image_files):
        if (i+1) % 50 == 0:
            print(f"Processing file {i+1}/{len(image_files)}: {filename}")
            
        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Warning: Could not read image {filename}. Skipping.")
            continue
            
        # Normalize image to [0, 1] for feature functions
        img_normalized = img.astype(np.float64) / 255.0
        fout = os.path.join(output_path, filename.replace('.bmp', '.mat'))

        # 4. Call the unified spatial grid feature extractor
        F = generate_spatial_grid_feature(
            img_normalized,
            grid_rows=args.grid_rows,
            grid_cols=args.grid_cols,
            feature_types=args.features,
            color_Q=args.color_Q,
            texture_bins=args.texture_bins
        )
        
        # Store descriptor size from the first image
        if i == 0:
            final_descriptor_size = F.shape[1]

        # Save the feature vector
        sio.savemat(fout, {'F': F})

    print(f"\n--- Finished! ---")
    print(f"Descriptors saved to: {output_path}")
    print(f"Final descriptor size (dimensions): {final_descriptor_size}")


if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='Compute Spatial Grid Descriptors (Requirement 3)')
    
    parser.add_argument('--dataset_folder', type=str, required=True, help='Path to MSRC_ObjCategImageDatabase_v2')
    parser.add_argument('--out_folder', type=str, required=True, help='Base folder to save descriptors (e.g., "descriptors")')
    
    # Flexible argument for choosing one or more features
    parser.add_argument('--features', nargs='+', 
                        default=['color_hist', 'texture'],
                        choices=['avg_color', 'color_hist', 'texture'],
                        help='One or more features to extract per cell. E.g., --features texture --features color_hist')
    
    # Grid geometry arguments
    parser.add_argument('--grid_rows', type=int, default=4, help='Number of grid rows (default: 4)')
    parser.add_argument('--grid_cols', type=int, default=4, help='Number of grid columns (default: 4)')
    
    # Feature-specific parameter arguments
    parser.add_argument('--color_Q', type=int, default=4,
                        help='Quantization for "color_hist" (default: 4). Total bins = Q^3')
    parser.add_argument('--texture_bins', type=int, default=8,
                        help='Number of bins for "texture" (EOH) (default: 8)')

    args = parser.parse_args()
    
    # Validate dataset path before starting
    if not os.path.isdir(os.path.join(args.dataset_folder, 'Images')):
        print(f"Error: 'Images' subfolder not found in {args.dataset_folder}")
    else:
        generate_spatial_descriptors(args)
